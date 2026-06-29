#include "convert.cuh"
#include "cudnn-sdpa.cuh"

#include <cudnn_frontend.h>
#include <cudnn_graph.h>

#include <cmath>
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

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

template <typename src_t>
static __global__ void cudnn_sdpa_bhsd_to_ggml_dhsb_f32(const src_t* __restrict__ src,
                                                        float* __restrict__ dst,
                                                        const int64_t batch,
                                                        const int64_t heads,
                                                        const int64_t q_seq,
                                                        const int64_t head_dim) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n = batch * heads * q_seq * head_dim;
    if (i >= n) {
        return;
    }

    const int64_t d = i % head_dim;
    const int64_t t = i / head_dim;
    const int64_t s = t % q_seq;
    const int64_t u = t / q_seq;
    const int64_t h = u % heads;
    const int64_t b = u / heads;

    dst[d + h * head_dim + s * heads * head_dim + b * q_seq * heads * head_dim] =
        ggml_cuda_cast<float>(src[i]);
}

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
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t q_seq = 0;
    int64_t kv_seq = 0;
    int64_t head_dim = 0;
    int io_dtype = 0;
    uint32_t scale_bits = 0;

    bool operator==(const graph_key& other) const {
        return std::tie(batch, heads, q_seq, kv_seq, head_dim, io_dtype, scale_bits) ==
               std::tie(other.batch,
                        other.heads,
                        other.q_seq,
                        other.kv_seq,
                        other.head_dim,
                        other.io_dtype,
                        other.scale_bits);
    }
};

struct graph_key_hash {
    size_t operator()(const graph_key& key) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        mix(static_cast<uint64_t>(key.batch));
        mix(static_cast<uint64_t>(key.heads));
        mix(static_cast<uint64_t>(key.q_seq));
        mix(static_cast<uint64_t>(key.kv_seq));
        mix(static_cast<uint64_t>(key.head_dim));
        mix(static_cast<uint64_t>(key.io_dtype));
        mix(static_cast<uint64_t>(key.scale_bits));
        return h;
    }
};

struct graph_entry {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
};

static fe::DataType_t cudnn_io_dtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return fe::DataType_t::HALF;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            return fe::DataType_t::BFLOAT16;
        default:
            return fe::DataType_t::NOT_SET;
    }
}

static bool allow_f32_to_bf16() {
    return std::getenv("GGML_CUDA_CUDNN_SDPA_ALLOW_F32_TO_BF16") != nullptr;
}

static bool diagnose_rejects(int64_t head_dim) {
    if (head_dim == 32) {
        return std::getenv("GGML_CUDA_DIAG_CUDNN_SDPA_HEAD32_REJECT") != nullptr;
    }
    if (head_dim == 64) {
        return std::getenv("GGML_CUDA_DIAG_CUDNN_SDPA_HEAD64_REJECT") != nullptr;
    }
    return false;
}

static bool should_log_reject(const ggml_tensor* dst, int64_t head_dim) {
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    return diagnose_rejects(head_dim) && q != nullptr && k != nullptr && v != nullptr &&
           (q->ne[0] == head_dim || k->ne[0] == head_dim || v->ne[0] == head_dim);
}

static bool reject_shape(const ggml_tensor* dst, int64_t head_dim, const char* reason) {
    if (should_log_reject(dst, head_dim)) {
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* v = dst->src[2];
        fprintf(stderr,
                "GGML_CUDA_CUDNN_SDPA_HEAD%lld_REJECT reason=%s Q=[%lld,%lld,%lld,%lld] "
                "K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld] "
                "dst=[%lld,%lld,%lld,%lld] q_contig=%d k_contig=%d v_contig=%d dst_contig=%d "
                "q_type=%s k_type=%s v_type=%s dst_type=%s\n",
                (long long) head_dim,
                reason,
                (long long) q->ne[0],
                (long long) q->ne[1],
                (long long) q->ne[2],
                (long long) q->ne[3],
                (long long) k->ne[0],
                (long long) k->ne[1],
                (long long) k->ne[2],
                (long long) k->ne[3],
                (long long) v->ne[0],
                (long long) v->ne[1],
                (long long) v->ne[2],
                (long long) v->ne[3],
                (long long) dst->ne[0],
                (long long) dst->ne[1],
                (long long) dst->ne[2],
                (long long) dst->ne[3],
                ggml_is_contiguous(q) ? 1 : 0,
                ggml_is_contiguous(k) ? 1 : 0,
                ggml_is_contiguous(v) ? 1 : 0,
                ggml_is_contiguous(dst) ? 1 : 0,
                ggml_type_name(q->type),
                ggml_type_name(k->type),
                ggml_type_name(v->type),
                ggml_type_name(dst->type));
    }
    return false;
}

static int cudnn_io_dtype_id(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return 1;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            return 2;
        default:
            return 0;
    }
}

static std::shared_ptr<fe::graph::Graph> create_graph(const graph_key& key,
                                                      fe::DataType_t io_dtype,
                                                      float scale) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const std::vector<int64_t> q_dims = {key.batch, key.heads, key.q_seq, key.head_dim};
    const std::vector<int64_t> kv_dims = {key.batch, key.heads, key.kv_seq, key.head_dim};
    const std::vector<int64_t> q_stride = {
        key.heads * key.q_seq * key.head_dim, key.q_seq * key.head_dim, key.head_dim, 1};
    const std::vector<int64_t> kv_stride = {
        key.heads * key.kv_seq * key.head_dim, key.kv_seq * key.head_dim, key.head_dim, 1};

    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim(q_dims)
                               .set_stride(q_stride)
                               .set_data_type(io_dtype));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim(kv_dims)
                               .set_stride(kv_stride)
                               .set_data_type(io_dtype));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim(kv_dims)
                               .set_stride(kv_stride)
                               .set_data_type(io_dtype));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("ggml_sdpa").set_attn_scale(scale);
    auto [o, stats] = graph->sdpa(q, k, v, sdpa_options);
    GGML_UNUSED(stats);
    o->set_output(true)
        .set_dim(q_dims)
        .set_stride(
            {key.heads * key.q_seq * key.head_dim, key.q_seq * key.head_dim, key.head_dim, 1})
        .set_data_type(io_dtype)
        .set_uid(O_UID);

    return graph;
}

static std::shared_ptr<graph_entry> get_graph(cudnnHandle_t handle,
                                              const graph_key& key,
                                              fe::DataType_t io_dtype,
                                              float scale) {
    static std::mutex mutex;
    static std::unordered_map<graph_key, std::shared_ptr<graph_entry>, graph_key_hash> cache;

    std::lock_guard lock(mutex);
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }

    auto entry = std::make_shared<graph_entry>();
    entry->graph = create_graph(key, io_dtype, scale);
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

static bool is_supported_shape(const ggml_tensor* dst, int64_t head_dim) {
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    const ggml_tensor* mask = dst->src[3];
    const ggml_tensor* sinks = dst->src[4];

    if (mask != nullptr || sinks != nullptr) {
        return reject_shape(dst, head_dim, "mask-or-sinks");
    }
    if (q->ne[0] != head_dim || k->ne[0] != head_dim || v->ne[0] != head_dim ||
        dst->ne[0] != head_dim) {
        return reject_shape(dst, head_dim, "head-dim");
    }
    const bool allow_head64_window =
        head_dim == 64 && std::getenv("GGML_CUDA_CUDNN_SDPA_HEAD64_ALLOW_WINDOW") != nullptr;
    const int64_t min_q_seq = allow_head64_window ? 576 : 1024;
    if (q->ne[1] < min_q_seq) {
        return reject_shape(dst, head_dim, "short-q-seq");
    }
    if (q->ne[2] != k->ne[2] || q->ne[2] != v->ne[2] || q->ne[3] != k->ne[3] ||
        q->ne[3] != v->ne[3]) {
        return reject_shape(dst, head_dim, "batch-or-heads");
    }
    if (k->ne[1] != v->ne[1] || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1] ||
        dst->ne[3] != q->ne[3]) {
        return reject_shape(dst, head_dim, "dst-shape");
    }
    const bool can_pack_v =
        !ggml_is_contiguous(v) && (v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_BF16);
    if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) ||
        (!ggml_is_contiguous(v) && !can_pack_v) || !ggml_is_contiguous(dst)) {
        return reject_shape(dst, head_dim, "non-contiguous");
    }
    if (q->type != k->type || q->type != v->type) {
        return reject_shape(dst, head_dim, "mixed-types");
    }
    if (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_BF16 && q->type != GGML_TYPE_F16) {
        return reject_shape(dst, head_dim, "unsupported-type");
    }
    if (q->type == GGML_TYPE_F32 && !allow_f32_to_bf16()) {
        return reject_shape(dst, head_dim, "f32-to-bf16-disabled");
    }
    if (dst->type != GGML_TYPE_F32) {
        return reject_shape(dst, head_dim, "dst-not-f32");
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    const auto* params = reinterpret_cast<const float*>(dst->op_params);
    std::memcpy(&max_bias, params + 1, sizeof(float));
    std::memcpy(&logit_softcap, params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return reject_shape(dst, head_dim, "bias-or-softcap");
    }
    return true;
}

}  // namespace

static bool ggml_cuda_cudnn_sdpa_head(ggml_backend_cuda_context& ctx,
                                      ggml_tensor* dst,
                                      int64_t head_dim,
                                      const char* enable_env,
                                      const char* unsafe_env,
                                      const char* profile_env) {
    if (std::getenv(enable_env) == nullptr) {
        return false;
    }
    if (std::getenv(unsafe_env) == nullptr) {
        return false;
    }
    if (!is_supported_shape(dst, head_dim)) {
        return false;
    }

    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    const fe::DataType_t io_dtype = cudnn_io_dtype(q->type);
    if (io_dtype == fe::DataType_t::NOT_SET) {
        return false;
    }

    float scale = 0.0f;
    std::memcpy(&scale, reinterpret_cast<const float*>(dst->op_params), sizeof(float));
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));

    ggml_cuda_set_device(ctx.device);
    cudnnHandle_t handle = get_handle(ctx.device);
    if (handle == nullptr) {
        return false;
    }
    if (cudnnSetStream(handle, ctx.stream()) != CUDNN_STATUS_SUCCESS) {
        return false;
    }

    const graph_key key{
        q->ne[3],
        q->ne[2],
        q->ne[1],
        k->ne[1],
        q->ne[0],
        cudnn_io_dtype_id(q->type),
        scale_bits,
    };
    auto entry = get_graph(handle, key, io_dtype, scale);
    if (!entry) {
        return false;
    }

    bool profile_enabled = std::getenv(profile_env) != nullptr;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_after_q = nullptr;
    cudaEvent_t profile_after_k = nullptr;
    cudaEvent_t profile_after_v = nullptr;
    cudaEvent_t profile_after_execute = nullptr;
    cudaEvent_t profile_after_output = nullptr;
    auto destroy_profile_events = [&] {
        for (cudaEvent_t event : {profile_start,
                                  profile_after_q,
                                  profile_after_k,
                                  profile_after_v,
                                  profile_after_execute,
                                  profile_after_output}) {
            if (event != nullptr) {
                CUDA_CHECK(cudaEventDestroy(event));
            }
        }
    };
    if (profile_enabled) {
        if (cudaEventCreate(&profile_start) != cudaSuccess ||
            cudaEventCreate(&profile_after_q) != cudaSuccess ||
            cudaEventCreate(&profile_after_k) != cudaSuccess ||
            cudaEventCreate(&profile_after_v) != cudaSuccess ||
            cudaEventCreate(&profile_after_execute) != cudaSuccess ||
            cudaEventCreate(&profile_after_output) != cudaSuccess) {
            destroy_profile_events();
            profile_enabled = false;
        }
    }
    auto record_profile = [&](cudaEvent_t event) {
        if (profile_enabled) {
            CUDA_CHECK(cudaEventRecord(event, ctx.stream()));
        }
    };
    bool recorded_after_q = false;
    bool recorded_after_k = false;
    bool recorded_after_v = false;
    record_profile(profile_start);

    const int64_t q_count = q->ne[0] * q->ne[1] * q->ne[2] * q->ne[3];
    const int64_t kv_count = k->ne[0] * k->ne[1] * k->ne[2] * k->ne[3];
    const void* q_data = q->data;
    const void* k_data = k->data;
    const bool pack_v = !ggml_is_contiguous(v);
    const void* v_data = v->data;
    ggml_cuda_pool_alloc<nv_bfloat16> q_bf16(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> k_bf16(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> v_bf16(ctx.pool());
    if (q->type == GGML_TYPE_F32) {
        const to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        const to_bf16_nc_cuda_t to_bf16_nc =
            pack_v ? ggml_get_to_bf16_nc_cuda(GGML_TYPE_F32) : nullptr;
        if (to_bf16 == nullptr || (pack_v && to_bf16_nc == nullptr)) {
            destroy_profile_events();
            return false;
        }
        q_data = q_bf16.alloc(static_cast<size_t>(q_count));
        k_data = k_bf16.alloc(static_cast<size_t>(kv_count));
        v_data = v_bf16.alloc(static_cast<size_t>(kv_count));
        to_bf16(q->data, q_bf16.get(), q_count, ctx.stream());
        record_profile(profile_after_q);
        recorded_after_q = true;
        to_bf16(k->data, k_bf16.get(), kv_count, ctx.stream());
        record_profile(profile_after_k);
        recorded_after_k = true;
        if (pack_v) {
            to_bf16_nc(v->data,
                       v_bf16.get(),
                       v->ne[0],
                       v->ne[1],
                       v->ne[2],
                       v->ne[3],
                       v->nb[1] / ggml_type_size(v->type),
                       v->nb[2] / ggml_type_size(v->type),
                       v->nb[3] / ggml_type_size(v->type),
                       ctx.stream());
        } else {
            to_bf16(v->data, v_bf16.get(), kv_count, ctx.stream());
        }
        record_profile(profile_after_v);
        recorded_after_v = true;
    } else if (q->type == GGML_TYPE_BF16 && pack_v) {
        const to_bf16_nc_cuda_t to_bf16_nc = ggml_get_to_bf16_nc_cuda(GGML_TYPE_BF16);
        if (to_bf16_nc == nullptr) {
            destroy_profile_events();
            return false;
        }
        v_data = v_bf16.alloc(static_cast<size_t>(kv_count));
        record_profile(profile_after_q);
        record_profile(profile_after_k);
        recorded_after_q = true;
        recorded_after_k = true;
        to_bf16_nc(v->data,
                   v_bf16.get(),
                   v->ne[0],
                   v->ne[1],
                   v->ne[2],
                   v->ne[3],
                   v->nb[1] / ggml_type_size(v->type),
                   v->nb[2] / ggml_type_size(v->type),
                   v->nb[3] / ggml_type_size(v->type),
                   ctx.stream());
        record_profile(profile_after_v);
        recorded_after_v = true;
    }
    if (!recorded_after_q) {
        record_profile(profile_after_q);
    }
    if (!recorded_after_k) {
        record_profile(profile_after_k);
    }
    if (!recorded_after_v) {
        record_profile(profile_after_v);
    }

    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    void* workspace_ptr = nullptr;
    if (entry->workspace_size > 0) {
        workspace_ptr = workspace.alloc(static_cast<size_t>(entry->workspace_size));
    }

    ggml_cuda_pool_alloc<nv_bfloat16> out_bhsd_bf16(ctx.pool());
    ggml_cuda_pool_alloc<half> out_bhsd_f16(ctx.pool());
    void* out_bhsd_ptr = nullptr;
    if (io_dtype == fe::DataType_t::BFLOAT16) {
        out_bhsd_ptr = out_bhsd_bf16.alloc(static_cast<size_t>(q_count));
    } else if (io_dtype == fe::DataType_t::HALF) {
        out_bhsd_ptr = out_bhsd_f16.alloc(static_cast<size_t>(q_count));
    } else {
        return false;
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {Q_UID, const_cast<void*>(q_data)},
        {K_UID, const_cast<void*>(k_data)},
        {V_UID, const_cast<void*>(v_data)},
        {O_UID, out_bhsd_ptr},
    };

    auto status = entry->graph->execute(handle, variant_pack, workspace_ptr);
    if (!status.is_good()) {
        destroy_profile_events();
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    record_profile(profile_after_execute);

    constexpr int threads = 256;
    const int blocks = (q_count + threads - 1) / threads;
    if (io_dtype == fe::DataType_t::BFLOAT16) {
        cudnn_sdpa_bhsd_to_ggml_dhsb_f32<<<blocks, threads, 0, ctx.stream()>>>(
            out_bhsd_bf16.get(),
            static_cast<float*>(dst->data),
            q->ne[3],
            q->ne[2],
            q->ne[1],
            q->ne[0]);
    } else {
        cudnn_sdpa_bhsd_to_ggml_dhsb_f32<<<blocks, threads, 0, ctx.stream()>>>(
            out_bhsd_f16.get(),
            static_cast<float*>(dst->data),
            q->ne[3],
            q->ne[2],
            q->ne[1],
            q->ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
    record_profile(profile_after_output);

    if (profile_enabled) {
        CUDA_CHECK(cudaEventSynchronize(profile_after_output));
        float q_ms = 0.0f;
        float k_ms = 0.0f;
        float v_ms = 0.0f;
        float execute_ms = 0.0f;
        float output_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&q_ms, profile_start, profile_after_q));
        CUDA_CHECK(cudaEventElapsedTime(&k_ms, profile_after_q, profile_after_k));
        CUDA_CHECK(cudaEventElapsedTime(&v_ms, profile_after_k, profile_after_v));
        CUDA_CHECK(cudaEventElapsedTime(&execute_ms, profile_after_v, profile_after_execute));
        CUDA_CHECK(cudaEventElapsedTime(&output_ms, profile_after_execute, profile_after_output));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, profile_start, profile_after_output));
        fprintf(stderr,
                "GGML_CUDA_CUDNN_SDPA_HEAD%lld Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] "
                "V=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld] type=%s scale=%.9f "
                "workspace=%lld v_pack=%d out=contiguous_bhsd_transpose q_ms=%.6f k_ms=%.6f "
                "v_ms=%.6f execute_ms=%.6f out_ms=%.6f total_ms=%.6f\n",
                (long long) head_dim,
                (long long) q->ne[0],
                (long long) q->ne[1],
                (long long) q->ne[2],
                (long long) q->ne[3],
                (long long) k->ne[0],
                (long long) k->ne[1],
                (long long) k->ne[2],
                (long long) k->ne[3],
                (long long) v->ne[0],
                (long long) v->ne[1],
                (long long) v->ne[2],
                (long long) v->ne[3],
                (long long) dst->ne[0],
                (long long) dst->ne[1],
                (long long) dst->ne[2],
                (long long) dst->ne[3],
                ggml_type_name(q->type),
                scale,
                (long long) entry->workspace_size,
                pack_v ? 1 : 0,
                q_ms,
                k_ms,
                v_ms,
                execute_ms,
                output_ms,
                total_ms);
    }
    destroy_profile_events();
    return true;
}

bool ggml_cuda_cudnn_sdpa_head32(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    return ggml_cuda_cudnn_sdpa_head(ctx,
                                     dst,
                                     32,
                                     "GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32",
                                     "GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD32_UNSAFE_RUN",
                                     "GGML_CUDA_PROFILE_CUDNN_SDPA_HEAD32");
}

bool ggml_cuda_cudnn_sdpa_head64(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    return ggml_cuda_cudnn_sdpa_head(ctx,
                                     dst,
                                     64,
                                     "GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD64",
                                     "GGML_CUDA_ENABLE_CUDNN_SDPA_HEAD64_UNSAFE_RUN",
                                     "GGML_CUDA_PROFILE_CUDNN_SDPA_HEAD64");
}
