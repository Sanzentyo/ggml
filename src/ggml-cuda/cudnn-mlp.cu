#include "cudnn-mlp.cuh"

#include <cudnn_frontend.h>
#include <cudnn_graph.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

constexpr int64_t FC1_W_UID = 1;
constexpr int64_t INPUT_UID = 2;
constexpr int64_t FC1_BIAS_UID = 3;
constexpr int64_t FC1_OUT_UID = 4;

struct cudnn_handle_deleter {
    void operator()(std::remove_pointer_t<cudnnHandle_t>* handle) const {
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }
};

using cudnn_handle_ptr =
    std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, cudnn_handle_deleter>;

struct graph_key {
    int64_t input_dim = 0;
    int64_t hidden_dim = 0;
    int64_t cols = 0;
    ggml_type input_type = GGML_TYPE_BF16;
    ggml_type output_type = GGML_TYPE_BF16;

    bool operator==(const graph_key& other) const {
        return std::tie(input_dim, hidden_dim, cols, input_type, output_type) ==
               std::tie(other.input_dim,
                        other.hidden_dim,
                        other.cols,
                        other.input_type,
                        other.output_type);
    }
};

struct graph_key_hash {
    size_t operator()(const graph_key& key) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        mix(static_cast<uint64_t>(key.input_dim));
        mix(static_cast<uint64_t>(key.hidden_dim));
        mix(static_cast<uint64_t>(key.cols));
        mix(static_cast<uint64_t>(key.input_type));
        mix(static_cast<uint64_t>(key.output_type));
        return h;
    }
};

struct graph_entry {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
};

static std::vector<int64_t> matrix_stride(int64_t rows, int64_t cols) {
    return {rows * cols, 1, rows};
}

static std::vector<int64_t> ggml_weight_as_transposed_stride(int64_t rows, int64_t cols) {
    return {rows * cols, cols, 1};
}

static fe::DataType_t cudnn_dtype_from_ggml(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return fe::DataType_t::HALF;
        case GGML_TYPE_BF16:
            return fe::DataType_t::BFLOAT16;
        case GGML_TYPE_F32:
            return fe::DataType_t::FLOAT;
        default:
            GGML_ABORT("unsupported cuDNN MLP tensor type");
    }
}

static std::shared_ptr<fe::graph::Graph> create_graph(const graph_key& key) {
    constexpr int64_t batch = 1;
    const auto input_dtype = cudnn_dtype_from_ggml(key.input_type);
    const auto output_dtype = cudnn_dtype_from_ggml(key.output_type);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(input_dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto fc1_w = graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("fc1_w")
            .set_uid(FC1_W_UID)
            .set_dim({batch, key.hidden_dim, key.input_dim})
            .set_stride(ggml_weight_as_transposed_stride(key.hidden_dim, key.input_dim))
            .set_data_type(input_dtype));
    auto input = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("input")
                                   .set_uid(INPUT_UID)
                                   .set_dim({batch, key.input_dim, key.cols})
                                   .set_stride(matrix_stride(key.input_dim, key.cols))
                                   .set_data_type(input_dtype));
    auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("fc1_bias")
                                  .set_uid(FC1_BIAS_UID)
                                  .set_dim({batch, key.hidden_dim, 1})
                                  .set_stride({key.hidden_dim, 1, key.hidden_dim})
                                  .set_data_type(fe::DataType_t::FLOAT));

    auto fc1 = graph->matmul(fc1_w,
                             input,
                             fe::graph::Matmul_attributes().set_name("fc1").set_compute_data_type(
                                 fe::DataType_t::FLOAT));
    fc1->set_data_type(fe::DataType_t::FLOAT);

    auto fc1_biased = graph->pointwise(fc1,
                                       bias,
                                       fe::graph::Pointwise_attributes()
                                           .set_name("fc1_bias_add")
                                           .set_mode(fe::PointwiseMode_t::ADD)
                                           .set_compute_data_type(fe::DataType_t::FLOAT));
    fc1_biased->set_data_type(fe::DataType_t::FLOAT);

    auto out = graph->pointwise(fc1_biased,
                                fe::graph::Pointwise_attributes()
                                    .set_name("fc1_gelu_erf")
                                    .set_mode(fe::PointwiseMode_t::GELU_FWD)
                                    .set_compute_data_type(fe::DataType_t::FLOAT));
    out->set_output(true)
        .set_uid(FC1_OUT_UID)
        .set_dim({batch, key.hidden_dim, key.cols})
        .set_stride(matrix_stride(key.hidden_dim, key.cols))
        .set_data_type(output_dtype);

    return graph;
}

static std::shared_ptr<graph_entry> get_graph(cudnnHandle_t handle, const graph_key& key) {
    static std::mutex mutex;
    static std::unordered_map<graph_key, std::shared_ptr<graph_entry>, graph_key_hash> cache;

    std::lock_guard lock(mutex);
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }

    auto entry = std::make_shared<graph_entry>();
    entry->graph = create_graph(key);
    auto status = entry->graph->build(handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        return nullptr;
    }
    auto workspace_status = entry->graph->get_workspace_size(entry->workspace_size);
    if (!workspace_status.is_good()) {
        return nullptr;
    }
    cache.emplace(key, entry);
    return entry;
}

static cudnnHandle_t get_handle(int device) {
    thread_local std::unordered_map<int, cudnn_handle_ptr> handles;
    if (auto it = handles.find(device); it != handles.end()) {
        return it->second.get();
    }

    cudnnHandle_t raw = nullptr;
    if (cudnnCreate(&raw) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    auto [it, inserted] = handles.emplace(device, cudnn_handle_ptr(raw));
    GGML_UNUSED(inserted);
    return it->second.get();
}

static const char* output_suffix(ggml_type output_type) {
    return output_type == GGML_TYPE_F32 ? "F32" : "BF16";
}

static bool should_log(ggml_type output_type) {
    if (std::getenv("GGML_CUDA_PROFILE_CUDNN_MLP_FC1_GELU") != nullptr) {
        return true;
    }
    if (output_type == GGML_TYPE_F32) {
        return std::getenv("GGML_CUDA_PROFILE_CUDNN_MLP_FC1_GELU_F32") != nullptr;
    }
    return std::getenv("GGML_CUDA_PROFILE_CUDNN_MLP_FC1_GELU_BF16") != nullptr;
}

static bool reject(const char* reason,
                   const ggml_tensor* mm_node,
                   const ggml_tensor* bias,
                   const ggml_tensor* dst,
                   ggml_type output_type) {
    if (should_log(output_type)) {
        const ggml_tensor* src0 = mm_node != nullptr ? mm_node->src[0] : nullptr;
        const ggml_tensor* src1 = mm_node != nullptr ? mm_node->src[1] : nullptr;
        std::fprintf(stderr,
                     "GGML_CUDA_CUDNN_MLP_FC1_GELU_%s reject reason=%s mm=%s src0=%s "
                     "src1=%s bias=%s dst=%s src0_type=%s src1_type=%s bias_type=%s "
                     "dst_type=%s dst_ne=[%lld,%lld,%lld,%lld]\n",
                     output_suffix(output_type),
                     reason,
                     mm_node != nullptr ? mm_node->name : "<null>",
                     src0 != nullptr ? src0->name : "<null>",
                     src1 != nullptr ? src1->name : "<null>",
                     bias != nullptr ? bias->name : "<null>",
                     dst != nullptr ? dst->name : "<null>",
                     src0 != nullptr ? ggml_type_name(src0->type) : "<null>",
                     src1 != nullptr ? ggml_type_name(src1->type) : "<null>",
                     bias != nullptr ? ggml_type_name(bias->type) : "<null>",
                     dst != nullptr ? ggml_type_name(dst->type) : "<null>",
                     dst != nullptr ? (long long) dst->ne[0] : -1LL,
                     dst != nullptr ? (long long) dst->ne[1] : -1LL,
                     dst != nullptr ? (long long) dst->ne[2] : -1LL,
                     dst != nullptr ? (long long) dst->ne[3] : -1LL);
    }
    return false;
}

static bool is_supported_shape(const ggml_tensor* mm_node,
                               const ggml_tensor* bias,
                               const ggml_tensor* dst,
                               ggml_type output_type) {
    if (mm_node == nullptr || bias == nullptr || dst == nullptr || mm_node->op != GGML_OP_MUL_MAT) {
        return reject("null-or-op", mm_node, bias, dst, output_type);
    }
    const ggml_tensor* src0 = mm_node->src[0];
    const ggml_tensor* src1 = mm_node->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return reject("null-src", mm_node, bias, dst, output_type);
    }
    const bool dst_type_ok = (output_type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) ||
                             (output_type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const bool input_type_ok =
        src0->type == src1->type && (src0->type == GGML_TYPE_BF16 ||
                                     (output_type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F16));
    if (!input_type_ok || bias->type != GGML_TYPE_F32 || !dst_type_ok) {
        return reject("type", mm_node, bias, dst, output_type);
    }
    if (std::strstr(src0->name, "vit.blocks.") == nullptr ||
        std::strstr(src0->name, ".mlp.lin1.") == nullptr) {
        return reject("not-vit-mlp-fc1", mm_node, bias, dst, output_type);
    }
    const int64_t src1_cols = ggml_nelements(src1) / src1->ne[0];
    const int64_t dst_cols = ggml_nelements(dst) / dst->ne[0];
    const bool shape_ok = src0->ne[0] == src1->ne[0] && src0->ne[1] == dst->ne[0] &&
                          src0->ne[2] == 1 && src0->ne[3] == 1 && src1_cols == dst_cols &&
                          bias->ne[0] == dst->ne[0] && ggml_nelements(bias) == dst->ne[0];
    if (!shape_ok) {
        return reject("shape", mm_node, bias, dst, output_type);
    }
    const bool stride_ok = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                           ggml_is_contiguous(dst) && ggml_is_contiguous(bias);
    if (!stride_ok) {
        return reject("stride", mm_node, bias, dst, output_type);
    }
    return true;
}

}  // namespace

static bool ggml_cuda_cudnn_mlp_fc1_gelu(ggml_backend_cuda_context& ctx,
                                         const ggml_tensor* mm_node,
                                         const ggml_tensor* bias,
                                         ggml_tensor* dst,
                                         ggml_type output_type,
                                         const char* enable_env,
                                         const char* unsafe_env) {
    if (std::getenv(enable_env) == nullptr || std::getenv(unsafe_env) == nullptr) {
        return false;
    }
    if (!is_supported_shape(mm_node, bias, dst, output_type)) {
        return false;
    }

    const ggml_tensor* src0 = mm_node->src[0];
    const ggml_tensor* src1 = mm_node->src[1];
    const graph_key key{
        src0->ne[0],
        src0->ne[1],
        ggml_nelements(dst) / dst->ne[0],
        src0->type,
        output_type,
    };

    ggml_cuda_set_device(ctx.device);
    cudnnHandle_t handle = get_handle(ctx.device);
    if (handle == nullptr) {
        return reject("handle", mm_node, bias, dst, output_type);
    }
    if (cudnnSetStream(handle, ctx.stream()) != CUDNN_STATUS_SUCCESS) {
        return reject("stream", mm_node, bias, dst, output_type);
    }

    auto entry = get_graph(handle, key);
    if (!entry) {
        return reject("graph", mm_node, bias, dst, output_type);
    }

    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    void* workspace_ptr = nullptr;
    if (entry->workspace_size > 0) {
        workspace_ptr = workspace.alloc(static_cast<size_t>(entry->workspace_size));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    const bool profile = should_log(output_type);
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, ctx.stream()));
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {FC1_W_UID, src0->data},
        {INPUT_UID, src1->data},
        {FC1_BIAS_UID, bias->data},
        {FC1_OUT_UID, dst->data},
    };
    auto status = entry->graph->execute(handle, variant_pack, workspace_ptr);
    if (!status.is_good()) {
        if (profile) {
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        return reject("execute", mm_node, bias, dst, output_type);
    }
    CUDA_CHECK(cudaGetLastError());

    if (profile) {
        CUDA_CHECK(cudaEventRecord(stop, ctx.stream()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float execute_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&execute_ms, start, stop));
        std::fprintf(stderr,
                     "GGML_CUDA_CUDNN_MLP_FC1_GELU_%s success mm=%s dst=%s "
                     "input_type=%s shape=[%lld,%lld,%lld] workspace=%lld execute_ms=%.6f\n",
                     output_suffix(output_type),
                     mm_node->name,
                     dst->name,
                     ggml_type_name(key.input_type),
                     (long long) key.input_dim,
                     (long long) key.hidden_dim,
                     (long long) key.cols,
                     (long long) entry->workspace_size,
                     execute_ms);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return true;
}

bool ggml_cuda_cudnn_mlp_fc1_gelu_bf16(ggml_backend_cuda_context& ctx,
                                       const ggml_tensor* mm_node,
                                       const ggml_tensor* bias,
                                       ggml_tensor* dst) {
    return ggml_cuda_cudnn_mlp_fc1_gelu(ctx,
                                        mm_node,
                                        bias,
                                        dst,
                                        GGML_TYPE_BF16,
                                        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16",
                                        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_BF16_UNSAFE_RUN");
}

bool ggml_cuda_cudnn_mlp_fc1_gelu_f32(ggml_backend_cuda_context& ctx,
                                      const ggml_tensor* mm_node,
                                      const ggml_tensor* bias,
                                      ggml_tensor* dst) {
    return ggml_cuda_cudnn_mlp_fc1_gelu(ctx,
                                        mm_node,
                                        bias,
                                        dst,
                                        GGML_TYPE_F32,
                                        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_F32",
                                        "GGML_CUDA_ENABLE_CUDNN_MLP_FC1_GELU_F32_UNSAFE_RUN");
}
