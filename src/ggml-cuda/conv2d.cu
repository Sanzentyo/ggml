#include "conv2d.cuh"
#include "convert.cuh"

#ifdef GGML_CUDA_USE_CUDNN
#include <cudnn.h>
#endif

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>

struct conv_params {
    const int64_t IW, IH;
    const int64_t OW, OH;
    const int64_t KW, KH;
    const int64_t ST_X, ST_Y;
    const int64_t PD_X, PD_Y;
    const int64_t DL_X, DL_Y;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
};

struct kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

__device__ __forceinline__ int64_t max64(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int64_t out_x,
                                                                 int64_t out_y,
                                                                 const conv_params& P) {
    kernel_bounds bounds;
    bounds.y_min = max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(
    int64_t out_coord, int64_t kern_coord, int64_t stride, int64_t dilation, int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int64_t input_index(
        int64_t n, int64_t c, int64_t y, int64_t x, const conv_params& P) {
        return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ static int64_t kernel_index(
        int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv_params& P) {
        return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ static int64_t output_index(
        int64_t n, int64_t c, int64_t y, int64_t x, const conv_params& P) {
        return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ static void unpack_indices(int64_t global_idx,
                                          const conv_params& P,
                                          int64_t& n,
                                          int64_t& c,
                                          int64_t& out_y,
                                          int64_t& out_x) {
        out_x = global_idx % P.OW;
        out_y = (global_idx / P.OW) % P.OH;
        c = (global_idx / (P.OW * P.OH)) % P.OC;
        n = global_idx / (P.OW * P.OH * P.OC);
    }
};

template <typename T, typename Layout>
static __global__ void conv2d_kernel(const float* __restrict__ input,
                                     const T* __restrict__ kernel,
                                     float* __restrict__ output,
                                     const conv_params P) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= P.TOTAL) {
        return;
    }

    int64_t n, c_out, out_y, out_x;
    Layout::unpack_indices(global_idx, P, n, c_out, out_y, out_x);

    float acc = 0.0f;

    for (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

        for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int64_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

            for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int64_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

                const float input_val = input[Layout::input_index(n, c_in, in_y, in_x, P)];
                const T kernel_val = kernel[Layout::kernel_index(c_out, c_in, ky, kx, P)];
                acc += (input_val * ggml_cuda_cast<float>(kernel_val));
            }
        }
    }

    // [N, OC, OH, OW]
    output[Layout::output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T>
static void conv2d_cuda(
    const float* X_D, const T* K_D, float* Y_D, const conv_params P, cudaStream_t st) {
    const int blocks = (P.TOTAL + CUDA_CONV2D_BLOCK_SIZE - 1) / CUDA_CONV2D_BLOCK_SIZE;
    conv2d_kernel<T, whcn_layout><<<blocks, CUDA_CONV2D_BLOCK_SIZE, 0, st>>>(X_D, K_D, Y_D, P);
}

static void conv2d_cuda_f16(
    const float* X_D, const half* K_D, float* Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<half>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(
    const float* X_D, const float* K_D, float* Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<float>(X_D, K_D, Y_D, P, st);
}

#ifdef GGML_CUDA_USE_CUDNN
namespace {

struct cudnn_handle_deleter {
    void operator()(std::remove_pointer_t<cudnnHandle_t>* handle) const {
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }
};

struct cudnn_tensor_desc_deleter {
    void operator()(std::remove_pointer_t<cudnnTensorDescriptor_t>* desc) const {
        if (desc != nullptr) {
            cudnnDestroyTensorDescriptor(desc);
        }
    }
};

struct cudnn_filter_desc_deleter {
    void operator()(std::remove_pointer_t<cudnnFilterDescriptor_t>* desc) const {
        if (desc != nullptr) {
            cudnnDestroyFilterDescriptor(desc);
        }
    }
};

struct cudnn_conv_desc_deleter {
    void operator()(std::remove_pointer_t<cudnnConvolutionDescriptor_t>* desc) const {
        if (desc != nullptr) {
            cudnnDestroyConvolutionDescriptor(desc);
        }
    }
};

using cudnn_handle_ptr =
    std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, cudnn_handle_deleter>;
using cudnn_tensor_desc_ptr =
    std::unique_ptr<std::remove_pointer_t<cudnnTensorDescriptor_t>, cudnn_tensor_desc_deleter>;
using cudnn_filter_desc_ptr =
    std::unique_ptr<std::remove_pointer_t<cudnnFilterDescriptor_t>, cudnn_filter_desc_deleter>;
using cudnn_conv_desc_ptr =
    std::unique_ptr<std::remove_pointer_t<cudnnConvolutionDescriptor_t>, cudnn_conv_desc_deleter>;

struct cudnn_conv2d_key {
    int device = 0;
    int iw = 0;
    int ih = 0;
    int ow = 0;
    int oh = 0;
    int kw = 0;
    int kh = 0;
    int ic = 0;
    int oc = 0;
    int batch = 0;
    int stride_x = 0;
    int stride_y = 0;
    int pad_x = 0;
    int pad_y = 0;
    int dilation_x = 0;
    int dilation_y = 0;
    int xw_dtype = 0;
    int y_dtype = 0;

    bool operator==(const cudnn_conv2d_key& other) const {
        return std::tie(device,
                        iw,
                        ih,
                        ow,
                        oh,
                        kw,
                        kh,
                        ic,
                        oc,
                        batch,
                        stride_x,
                        stride_y,
                        pad_x,
                        pad_y,
                        dilation_x,
                        dilation_y,
                        xw_dtype,
                        y_dtype) == std::tie(other.device,
                                             other.iw,
                                             other.ih,
                                             other.ow,
                                             other.oh,
                                             other.kw,
                                             other.kh,
                                             other.ic,
                                             other.oc,
                                             other.batch,
                                             other.stride_x,
                                             other.stride_y,
                                             other.pad_x,
                                             other.pad_y,
                                             other.dilation_x,
                                             other.dilation_y,
                                             other.xw_dtype,
                                             other.y_dtype);
    }
};

struct cudnn_conv2d_key_hash {
    size_t operator()(const cudnn_conv2d_key& key) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        mix(static_cast<uint64_t>(key.device));
        mix(static_cast<uint64_t>(key.iw));
        mix(static_cast<uint64_t>(key.ih));
        mix(static_cast<uint64_t>(key.ow));
        mix(static_cast<uint64_t>(key.oh));
        mix(static_cast<uint64_t>(key.kw));
        mix(static_cast<uint64_t>(key.kh));
        mix(static_cast<uint64_t>(key.ic));
        mix(static_cast<uint64_t>(key.oc));
        mix(static_cast<uint64_t>(key.batch));
        mix(static_cast<uint64_t>(key.stride_x));
        mix(static_cast<uint64_t>(key.stride_y));
        mix(static_cast<uint64_t>(key.pad_x));
        mix(static_cast<uint64_t>(key.pad_y));
        mix(static_cast<uint64_t>(key.dilation_x));
        mix(static_cast<uint64_t>(key.dilation_y));
        mix(static_cast<uint64_t>(key.xw_dtype));
        mix(static_cast<uint64_t>(key.y_dtype));
        return h;
    }
};

struct cudnn_conv2d_plan {
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_size = 0;
};

static cudnnHandle_t get_cudnn_handle(int device) {
    thread_local std::unordered_map<int, cudnn_handle_ptr> handles;
    auto it = handles.find(device);
    if (it != handles.end()) {
        return it->second.get();
    }

    ggml_cuda_set_device(device);
    cudnnHandle_t raw = nullptr;
    if (cudnnCreate(&raw) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    auto [inserted, _] = handles.emplace(device, cudnn_handle_ptr(raw));
    return inserted->second.get();
}

static cudnn_tensor_desc_ptr make_tensor_desc() {
    cudnnTensorDescriptor_t raw = nullptr;
    if (cudnnCreateTensorDescriptor(&raw) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    return cudnn_tensor_desc_ptr(raw);
}

static cudnn_filter_desc_ptr make_filter_desc() {
    cudnnFilterDescriptor_t raw = nullptr;
    if (cudnnCreateFilterDescriptor(&raw) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    return cudnn_filter_desc_ptr(raw);
}

static cudnn_conv_desc_ptr make_conv_desc() {
    cudnnConvolutionDescriptor_t raw = nullptr;
    if (cudnnCreateConvolutionDescriptor(&raw) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    return cudnn_conv_desc_ptr(raw);
}

static bool profile_cudnn_conv2d() {
    return std::getenv("GGML_CUDA_PROFILE_CUDNN_CONV2D") != nullptr;
}

static bool cudnn_conv2d_enabled() {
    return std::getenv("GGML_CUDA_DISABLE_CUDNN_CONV2D") == nullptr;
}

static bool cudnn_conv2d_f32_lowp_enabled() {
    const char * env = std::getenv("GGML_CUDA_ENABLE_CUDNN_CONV2D_F32_LOWP");
    return env != nullptr && env[0] != '0';
}

static bool should_log_cudnn_conv2d_reject() {
    return std::getenv("GGML_CUDA_DIAG_CUDNN_CONV2D_REJECT") != nullptr;
}

static std::optional<cudnnConvolutionFwdAlgo_t> forced_cudnn_conv2d_algo() {
    const char* env = std::getenv("GGML_CUDA_CUDNN_CONV2D_ALGO");
    if (env == nullptr || *env == '\0') {
        return std::nullopt;
    }
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0 || value >= CUDNN_CONVOLUTION_FWD_ALGO_COUNT) {
        return std::nullopt;
    }
    return static_cast<cudnnConvolutionFwdAlgo_t>(value);
}

static bool log_cudnn_conv2d_reject(const char* reason,
                                    cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    if (should_log_cudnn_conv2d_reject()) {
        if (status == CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "GGML_CUDA_CUDNN_CONV2D_REJECT reason=%s\n", reason);
        } else {
            std::fprintf(stderr,
                         "GGML_CUDA_CUDNN_CONV2D_REJECT reason=%s status=%s\n",
                         reason,
                         cudnnGetErrorString(status));
        }
    }
    return false;
}

static bool set_nchw_tensor_desc(
    const cudnnTensorDescriptor_t desc, cudnnDataType_t data_type, int n, int c, int h, int w) {
    const int stride_w = 1;
    const int stride_h = w;
    const int stride_c = h * w;
    const int stride_n = c * h * w;
    return cudnnSetTensor4dDescriptorEx(
               desc, data_type, n, c, h, w, stride_n, stride_c, stride_h, stride_w) ==
           CUDNN_STATUS_SUCCESS;
}

static bool get_cudnn_conv2d_plan(cudnnHandle_t handle,
                                  const cudnn_conv2d_key& key,
                                  const cudnnTensorDescriptor_t x_desc,
                                  const cudnnFilterDescriptor_t w_desc,
                                  const cudnnConvolutionDescriptor_t conv_desc,
                                  const cudnnTensorDescriptor_t y_desc,
                                  cudnn_conv2d_plan& plan) {
    thread_local std::unordered_map<cudnn_conv2d_key, cudnn_conv2d_plan, cudnn_conv2d_key_hash>
        plan_cache;
    if (auto it = plan_cache.find(key); it != plan_cache.end()) {
        plan = it->second;
        return true;
    }

    const auto forced_algo = forced_cudnn_conv2d_algo();
    const cudnnConvolutionFwdAlgo_t default_algos[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    };
    const cudnnConvolutionFwdAlgo_t forced_value =
        forced_algo.value_or(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    const cudnnConvolutionFwdAlgo_t forced_algos[] = {forced_value};
    const cudnnConvolutionFwdAlgo_t* algos = forced_algo ? forced_algos : default_algos;
    const int n_algos =
        forced_algo ? 1 : static_cast<int>(sizeof(default_algos) / sizeof(default_algos[0]));

    for (int i = 0; i < n_algos; ++i) {
        const cudnnConvolutionFwdAlgo_t algo = algos[i];
        size_t workspace_size = 0;
        const cudnnStatus_t workspace_status = cudnnGetConvolutionForwardWorkspaceSize(
            handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size);
        if (workspace_status == CUDNN_STATUS_SUCCESS) {
            plan.algo = algo;
            plan.workspace_size = workspace_size;
            plan_cache.emplace(key, plan);
            return true;
        }
    }

    return log_cudnn_conv2d_reject("no_supported_algorithm");
}

static bool try_cudnn_conv2d_lowp_input_lowp_kernel_f32_output(ggml_backend_cuda_context& ctx,
                                                               const ggml_tensor* kernel,
                                                               const ggml_tensor* input,
                                                               ggml_tensor* dst,
                                                               const conv_params& P) {
    if (!cudnn_conv2d_enabled()) {
        return false;
    }
    if ((kernel->type != GGML_TYPE_F16 && kernel->type != GGML_TYPE_BF16) ||
        input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return log_cudnn_conv2d_reject("dtype");
    }
    if (!ggml_is_contiguous(kernel) || !ggml_is_contiguous(input) || !ggml_is_contiguous(dst)) {
        return log_cudnn_conv2d_reject("non_contiguous");
    }
    if (P.B <= 0 || P.IC <= 0 || P.OC <= 0 || P.IW <= 0 || P.IH <= 0 || P.OW <= 0 || P.OH <= 0 ||
        P.KW <= 0 || P.KH <= 0) {
        return log_cudnn_conv2d_reject("bad_shape");
    }
    if (P.B > INT_MAX || P.IC > INT_MAX || P.OC > INT_MAX || P.IW > INT_MAX || P.IH > INT_MAX ||
        P.OW > INT_MAX || P.OH > INT_MAX || P.KW > INT_MAX || P.KH > INT_MAX || P.ST_X > INT_MAX ||
        P.ST_Y > INT_MAX || P.PD_X > INT_MAX || P.PD_Y > INT_MAX || P.DL_X > INT_MAX ||
        P.DL_Y > INT_MAX) {
        return log_cudnn_conv2d_reject("shape_overflow");
    }

    cudnnHandle_t handle = get_cudnn_handle(ctx.device);
    if (handle == nullptr) {
        return log_cudnn_conv2d_reject("create_handle");
    }
    const cudnnStatus_t stream_status = cudnnSetStream(handle, ctx.stream());
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_stream", stream_status);
    }

    auto x_desc = make_tensor_desc();
    auto y_desc = make_tensor_desc();
    auto w_desc = make_filter_desc();
    auto conv_desc = make_conv_desc();
    if (!x_desc || !y_desc || !w_desc || !conv_desc) {
        return log_cudnn_conv2d_reject("create_descriptor");
    }

    const int n = static_cast<int>(P.B);
    const int ic = static_cast<int>(P.IC);
    const int oc = static_cast<int>(P.OC);
    const int ih = static_cast<int>(P.IH);
    const int iw = static_cast<int>(P.IW);
    const int oh = static_cast<int>(P.OH);
    const int ow = static_cast<int>(P.OW);
    const int kh = static_cast<int>(P.KH);
    const int kw = static_cast<int>(P.KW);
    const int pad_h = static_cast<int>(P.PD_Y);
    const int pad_w = static_cast<int>(P.PD_X);
    const int stride_h = static_cast<int>(P.ST_Y);
    const int stride_w = static_cast<int>(P.ST_X);
    const int dilation_h = static_cast<int>(P.DL_Y);
    const int dilation_w = static_cast<int>(P.DL_X);
    const bool use_bf16 = kernel->type == GGML_TYPE_BF16;
    const cudnnDataType_t lowp_dtype = use_bf16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_HALF;
    const int lowp_dtype_key = use_bf16 ? 2 : 1;

    if (!set_nchw_tensor_desc(x_desc.get(), lowp_dtype, n, ic, ih, iw) ||
        !set_nchw_tensor_desc(y_desc.get(), CUDNN_DATA_FLOAT, n, oc, oh, ow)) {
        return log_cudnn_conv2d_reject("set_tensor_descriptor");
    }

    cudnnStatus_t status = cudnnSetFilter4dDescriptor(
        w_desc.get(), lowp_dtype, CUDNN_TENSOR_NCHW, oc, ic, kh, kw);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_filter_descriptor", status);
    }

    status = cudnnSetConvolution2dDescriptor(conv_desc.get(),
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_convolution_descriptor", status);
    }
    status = cudnnSetConvolutionMathType(conv_desc.get(), CUDNN_TENSOR_OP_MATH);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_math_type", status);
    }

    const cudnn_conv2d_key key = {ctx.device,
                                  iw,
                                  ih,
                                  ow,
                                  oh,
                                  kw,
                                  kh,
                                  ic,
                                  oc,
                                  n,
                                  stride_w,
                                  stride_h,
                                  pad_w,
                                  pad_h,
                                  dilation_w,
                                  dilation_h,
                                  lowp_dtype_key,
                                  0};
    cudnn_conv2d_plan plan;
    bool output_lowp = false;
    if (!get_cudnn_conv2d_plan(
            handle, key, x_desc.get(), w_desc.get(), conv_desc.get(), y_desc.get(), plan)) {
        if (!set_nchw_tensor_desc(y_desc.get(), lowp_dtype, n, oc, oh, ow)) {
            return log_cudnn_conv2d_reject("set_lowp_output_descriptor");
        }
        cudnn_conv2d_key lowp_key = key;
        lowp_key.y_dtype = lowp_dtype_key;
        if (!get_cudnn_conv2d_plan(
                handle,
                lowp_key,
                x_desc.get(),
                w_desc.get(),
                conv_desc.get(),
                y_desc.get(),
                plan)) {
            return false;
        }
        output_lowp = true;
    }

    const int64_t input_ne = P.B * P.IC * P.IH * P.IW;
    const int64_t output_ne = P.B * P.OC * P.OH * P.OW;
    const size_t lowp_size = use_bf16 ? sizeof(nv_bfloat16) : sizeof(half);
    ggml_cuda_pool_alloc<char> input_lowp(ctx.pool());
    void* input_lowp_ptr = input_lowp.alloc(static_cast<size_t>(input_ne) * lowp_size);
    if (use_bf16) {
        const to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_bf16 != nullptr);
        to_bf16(input->data, static_cast<nv_bfloat16*>(input_lowp_ptr), input_ne, ctx.stream());
    } else {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16(input->data, static_cast<half*>(input_lowp_ptr), input_ne, ctx.stream());
    }

    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    void* workspace_ptr = nullptr;
    if (plan.workspace_size > 0) {
        workspace_ptr = workspace.alloc(plan.workspace_size);
    }

    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool do_profile = profile_cudnn_conv2d();
    if (do_profile) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_stop));
        CUDA_CHECK(cudaEventRecord(profile_start, ctx.stream()));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    ggml_cuda_pool_alloc<char> output_lowp_alloc(ctx.pool());
    void* output_ptr = dst->data;
    if (output_lowp) {
        output_ptr = output_lowp_alloc.alloc(static_cast<size_t>(output_ne) * lowp_size);
    }
    status = cudnnConvolutionForward(handle,
                                     &alpha,
                                     x_desc.get(),
                                     input_lowp_ptr,
                                     w_desc.get(),
                                     kernel->data,
                                     conv_desc.get(),
                                     plan.algo,
                                     workspace_ptr,
                                     plan.workspace_size,
                                     &beta,
                                     y_desc.get(),
                                     output_ptr);

    if (do_profile) {
        CUDA_CHECK(cudaEventRecord(profile_stop, ctx.stream()));
        CUDA_CHECK(cudaEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        std::fprintf(stderr,
                     "GGML_CUDA_PROFILE_CUDNN_CONV2D ms=%.6f algo=%d workspace=%zu "
                     "xw_dtype=%s y_dtype=%s x=[%d,%d,%d,%d] w=[%d,%d,%d,%d] "
                     "y=[%d,%d,%d,%d]\n",
                     elapsed_ms,
                     static_cast<int>(plan.algo),
                     plan.workspace_size,
                     use_bf16 ? "bf16" : "f16",
                     output_lowp ? (use_bf16 ? "bf16" : "f16") : "f32",
                     n,
                     ic,
                     ih,
                     iw,
                     oc,
                     ic,
                     kh,
                     kw,
                     n,
                     oc,
                     oh,
                     ow);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_stop));
    }

    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("forward", status);
    }
    if (output_lowp) {
        const to_fp32_cuda_t to_fp32 =
            ggml_get_to_fp32_cuda(use_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F16);
        GGML_ASSERT(to_fp32 != nullptr);
        to_fp32(output_lowp_alloc.get(), static_cast<float*>(dst->data), output_ne, ctx.stream());
    }
    return true;
}

static bool try_cudnn_conv2d_f32_input_lowp_kernel_f32_output(ggml_backend_cuda_context& ctx,
                                                              const ggml_tensor* kernel,
                                                              const ggml_tensor* input,
                                                              ggml_tensor* dst,
                                                              const conv_params& P) {
    if (!cudnn_conv2d_enabled() || !cudnn_conv2d_f32_lowp_enabled()) {
        return false;
    }
    if ((kernel->type != GGML_TYPE_F16 && kernel->type != GGML_TYPE_BF16) ||
        input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return log_cudnn_conv2d_reject("dtype");
    }
    if (!ggml_is_contiguous(kernel) || !ggml_is_contiguous(input) || !ggml_is_contiguous(dst)) {
        return log_cudnn_conv2d_reject("non_contiguous");
    }
    if (P.B <= 0 || P.IC <= 0 || P.OC <= 0 || P.IW <= 0 || P.IH <= 0 || P.OW <= 0 || P.OH <= 0 ||
        P.KW <= 0 || P.KH <= 0) {
        return log_cudnn_conv2d_reject("bad_shape");
    }
    if (P.B > INT_MAX || P.IC > INT_MAX || P.OC > INT_MAX || P.IW > INT_MAX || P.IH > INT_MAX ||
        P.OW > INT_MAX || P.OH > INT_MAX || P.KW > INT_MAX || P.KH > INT_MAX || P.ST_X > INT_MAX ||
        P.ST_Y > INT_MAX || P.PD_X > INT_MAX || P.PD_Y > INT_MAX || P.DL_X > INT_MAX ||
        P.DL_Y > INT_MAX) {
        return log_cudnn_conv2d_reject("shape_overflow");
    }

    cudnnHandle_t handle = get_cudnn_handle(ctx.device);
    if (handle == nullptr) {
        return log_cudnn_conv2d_reject("create_handle");
    }
    const cudnnStatus_t stream_status = cudnnSetStream(handle, ctx.stream());
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_stream", stream_status);
    }

    auto x_desc = make_tensor_desc();
    auto y_desc = make_tensor_desc();
    auto w_desc = make_filter_desc();
    auto conv_desc = make_conv_desc();
    if (!x_desc || !y_desc || !w_desc || !conv_desc) {
        return log_cudnn_conv2d_reject("create_descriptor");
    }

    const int n = static_cast<int>(P.B);
    const int ic = static_cast<int>(P.IC);
    const int oc = static_cast<int>(P.OC);
    const int ih = static_cast<int>(P.IH);
    const int iw = static_cast<int>(P.IW);
    const int oh = static_cast<int>(P.OH);
    const int ow = static_cast<int>(P.OW);
    const int kh = static_cast<int>(P.KH);
    const int kw = static_cast<int>(P.KW);
    const int pad_h = static_cast<int>(P.PD_Y);
    const int pad_w = static_cast<int>(P.PD_X);
    const int stride_h = static_cast<int>(P.ST_Y);
    const int stride_w = static_cast<int>(P.ST_X);
    const int dilation_h = static_cast<int>(P.DL_Y);
    const int dilation_w = static_cast<int>(P.DL_X);
    const bool use_bf16 = kernel->type == GGML_TYPE_BF16;
    const cudnnDataType_t lowp_dtype = use_bf16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_HALF;
    const int mixed_dtype_key = use_bf16 ? 20 : 10;

    if (!set_nchw_tensor_desc(x_desc.get(), CUDNN_DATA_FLOAT, n, ic, ih, iw) ||
        !set_nchw_tensor_desc(y_desc.get(), CUDNN_DATA_FLOAT, n, oc, oh, ow)) {
        return log_cudnn_conv2d_reject("set_tensor_descriptor");
    }

    cudnnStatus_t status = cudnnSetFilter4dDescriptor(
        w_desc.get(), lowp_dtype, CUDNN_TENSOR_NCHW, oc, ic, kh, kw);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_filter_descriptor", status);
    }

    status = cudnnSetConvolution2dDescriptor(conv_desc.get(),
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_convolution_descriptor", status);
    }
    status = cudnnSetConvolutionMathType(conv_desc.get(), CUDNN_TENSOR_OP_MATH);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_math_type", status);
    }

    const cudnn_conv2d_key key = {ctx.device,
                                  iw,
                                  ih,
                                  ow,
                                  oh,
                                  kw,
                                  kh,
                                  ic,
                                  oc,
                                  n,
                                  stride_w,
                                  stride_h,
                                  pad_w,
                                  pad_h,
                                  dilation_w,
                                  dilation_h,
                                  mixed_dtype_key,
                                  0};
    cudnn_conv2d_plan plan;
    if (!get_cudnn_conv2d_plan(
            handle, key, x_desc.get(), w_desc.get(), conv_desc.get(), y_desc.get(), plan)) {
        return false;
    }

    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    void* workspace_ptr = nullptr;
    if (plan.workspace_size > 0) {
        workspace_ptr = workspace.alloc(plan.workspace_size);
    }

    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool do_profile = profile_cudnn_conv2d();
    if (do_profile) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_stop));
        CUDA_CHECK(cudaEventRecord(profile_start, ctx.stream()));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cudnnConvolutionForward(handle,
                                     &alpha,
                                     x_desc.get(),
                                     input->data,
                                     w_desc.get(),
                                     kernel->data,
                                     conv_desc.get(),
                                     plan.algo,
                                     workspace_ptr,
                                     plan.workspace_size,
                                     &beta,
                                     y_desc.get(),
                                     dst->data);

    if (do_profile) {
        CUDA_CHECK(cudaEventRecord(profile_stop, ctx.stream()));
        CUDA_CHECK(cudaEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        std::fprintf(stderr,
                     "GGML_CUDA_PROFILE_CUDNN_CONV2D ms=%.6f algo=%d workspace=%zu "
                     "x_dtype=f32 w_dtype=%s y_dtype=f32 x=[%d,%d,%d,%d] "
                     "w=[%d,%d,%d,%d] y=[%d,%d,%d,%d]\n",
                     elapsed_ms,
                     static_cast<int>(plan.algo),
                     plan.workspace_size,
                     use_bf16 ? "bf16" : "f16",
                     n,
                     ic,
                     ih,
                     iw,
                     oc,
                     ic,
                     kh,
                     kw,
                     n,
                     oc,
                     oh,
                     ow);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_stop));
    }

    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("forward", status);
    }
    return true;
}

static bool try_cudnn_conv2d_f32_input_f32_kernel_f32_output(ggml_backend_cuda_context& ctx,
                                                             const ggml_tensor* kernel,
                                                             const ggml_tensor* input,
                                                             ggml_tensor* dst,
                                                             const conv_params& P) {
    if (!cudnn_conv2d_enabled()) {
        return false;
    }
    if (kernel->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return log_cudnn_conv2d_reject("dtype");
    }
    if (!ggml_is_contiguous(kernel) || !ggml_is_contiguous(input) || !ggml_is_contiguous(dst)) {
        return log_cudnn_conv2d_reject("non_contiguous");
    }
    if (P.B <= 0 || P.IC <= 0 || P.OC <= 0 || P.IW <= 0 || P.IH <= 0 || P.OW <= 0 || P.OH <= 0 ||
        P.KW <= 0 || P.KH <= 0) {
        return log_cudnn_conv2d_reject("bad_shape");
    }
    if (P.B > INT_MAX || P.IC > INT_MAX || P.OC > INT_MAX || P.IW > INT_MAX || P.IH > INT_MAX ||
        P.OW > INT_MAX || P.OH > INT_MAX || P.KW > INT_MAX || P.KH > INT_MAX || P.ST_X > INT_MAX ||
        P.ST_Y > INT_MAX || P.PD_X > INT_MAX || P.PD_Y > INT_MAX || P.DL_X > INT_MAX ||
        P.DL_Y > INT_MAX) {
        return log_cudnn_conv2d_reject("shape_overflow");
    }

    cudnnHandle_t handle = get_cudnn_handle(ctx.device);
    if (handle == nullptr) {
        return log_cudnn_conv2d_reject("create_handle");
    }
    const cudnnStatus_t stream_status = cudnnSetStream(handle, ctx.stream());
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_stream", stream_status);
    }

    auto x_desc = make_tensor_desc();
    auto y_desc = make_tensor_desc();
    auto w_desc = make_filter_desc();
    auto conv_desc = make_conv_desc();
    if (!x_desc || !y_desc || !w_desc || !conv_desc) {
        return log_cudnn_conv2d_reject("create_descriptor");
    }

    const int n = static_cast<int>(P.B);
    const int ic = static_cast<int>(P.IC);
    const int oc = static_cast<int>(P.OC);
    const int ih = static_cast<int>(P.IH);
    const int iw = static_cast<int>(P.IW);
    const int oh = static_cast<int>(P.OH);
    const int ow = static_cast<int>(P.OW);
    const int kh = static_cast<int>(P.KH);
    const int kw = static_cast<int>(P.KW);
    const int pad_h = static_cast<int>(P.PD_Y);
    const int pad_w = static_cast<int>(P.PD_X);
    const int stride_h = static_cast<int>(P.ST_Y);
    const int stride_w = static_cast<int>(P.ST_X);
    const int dilation_h = static_cast<int>(P.DL_Y);
    const int dilation_w = static_cast<int>(P.DL_X);

    if (!set_nchw_tensor_desc(x_desc.get(), CUDNN_DATA_FLOAT, n, ic, ih, iw) ||
        !set_nchw_tensor_desc(y_desc.get(), CUDNN_DATA_FLOAT, n, oc, oh, ow)) {
        return log_cudnn_conv2d_reject("set_tensor_descriptor");
    }

    cudnnStatus_t status = cudnnSetFilter4dDescriptor(
        w_desc.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, oc, ic, kh, kw);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_filter_descriptor", status);
    }

    status = cudnnSetConvolution2dDescriptor(conv_desc.get(),
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_convolution_descriptor", status);
    }
    const cudnnMathType_t math_type = std::getenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F") == nullptr
                                         ? CUDNN_TENSOR_OP_MATH
                                         : CUDNN_FMA_MATH;
    status = cudnnSetConvolutionMathType(conv_desc.get(), math_type);
    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("set_math_type", status);
    }

    const cudnn_conv2d_key key = {ctx.device,
                                  iw,
                                  ih,
                                  ow,
                                  oh,
                                  kw,
                                  kh,
                                  ic,
                                  oc,
                                  n,
                                  stride_w,
                                  stride_h,
                                  pad_w,
                                  pad_h,
                                  dilation_w,
                                  dilation_h,
                                  0,
                                  0};
    cudnn_conv2d_plan plan;
    if (!get_cudnn_conv2d_plan(
            handle, key, x_desc.get(), w_desc.get(), conv_desc.get(), y_desc.get(), plan)) {
        return false;
    }

    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    void* workspace_ptr = nullptr;
    if (plan.workspace_size > 0) {
        workspace_ptr = workspace.alloc(plan.workspace_size);
    }

    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    const bool do_profile = profile_cudnn_conv2d();
    if (do_profile) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_stop));
        CUDA_CHECK(cudaEventRecord(profile_start, ctx.stream()));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cudnnConvolutionForward(handle,
                                     &alpha,
                                     x_desc.get(),
                                     input->data,
                                     w_desc.get(),
                                     kernel->data,
                                     conv_desc.get(),
                                     plan.algo,
                                     workspace_ptr,
                                     plan.workspace_size,
                                     &beta,
                                     y_desc.get(),
                                     dst->data);

    if (do_profile) {
        CUDA_CHECK(cudaEventRecord(profile_stop, ctx.stream()));
        CUDA_CHECK(cudaEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        std::fprintf(stderr,
                     "GGML_CUDA_PROFILE_CUDNN_CONV2D ms=%.6f algo=%d workspace=%zu "
                     "xw_dtype=f32 y_dtype=f32 math=%s x=[%d,%d,%d,%d] w=[%d,%d,%d,%d] "
                     "y=[%d,%d,%d,%d]\n",
                     elapsed_ms,
                     static_cast<int>(plan.algo),
                     plan.workspace_size,
                     math_type == CUDNN_FMA_MATH ? "fma" : "tensor_op",
                     n,
                     ic,
                     ih,
                     iw,
                     oc,
                     ic,
                     kh,
                     kw,
                     n,
                     oc,
                     oh,
                     ow);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_stop));
    }

    if (status != CUDNN_STATUS_SUCCESS) {
        return log_cudnn_conv2d_reject("forward", status);
    }
    return true;
}

}  // namespace
#endif

void ggml_cuda_op_conv2d(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* kernel = dst->src[0];
    const ggml_tensor* input = dst->src[1];
    float* K_D = (float*) kernel->data;
    const float* X_D = (const float*) input->data;
    float* Y_D = (float*) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_BF16 ||
                kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    cudaStream_t st = ctx.stream();

    const int32_t* p = (const int32_t*) dst->op_params;
    const int ST_X = p[0];  // stride_x
    const int ST_Y = p[1];  // stride_y
    const int PD_X = p[2];  // padding_x
    const int PD_Y = p[3];  // padding_y
    const int DL_X = p[4];  // dilation_x
    const int DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int IW = input->ne[0];   // input_w
    const int IH = input->ne[1];   // input_h
    const int OW = dst->ne[0];     // output_w
    const int OH = dst->ne[1];     // output_h
    const int KW = kernel->ne[0];  // kernel_w
    const int KH = kernel->ne[1];  // kernel_h
    const int IC = input->ne[2];   // input_channels
    const int OC = kernel->ne[3];  // ouptut_chanles
    const int B = input->ne[3];    // n_batches

    const int64_t total = B * OC * OH * OW;
    conv_params params = {
        IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total};

#ifdef GGML_CUDA_USE_CUDNN
    if (try_cudnn_conv2d_f32_input_f32_kernel_f32_output(ctx, kernel, input, dst, params)) {
        return;
    }
    if (try_cudnn_conv2d_f32_input_lowp_kernel_f32_output(ctx, kernel, input, dst, params)) {
        return;
    }
    if (try_cudnn_conv2d_lowp_input_lowp_kernel_f32_output(ctx, kernel, input, dst, params)) {
        return;
    }
#endif

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_cuda_f16(X_D, reinterpret_cast<const half*>(kernel->data), Y_D, params, st);
    } else if (kernel->type == GGML_TYPE_BF16) {
        conv2d_cuda<nv_bfloat16>(
            X_D, reinterpret_cast<const nv_bfloat16*>(kernel->data), Y_D, params, st);
    } else {
        conv2d_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}
