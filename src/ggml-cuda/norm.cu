#include "norm.cuh"
#include "convert.cuh"
#include <cstdlib>
#include <cstdint>

static int ggml_cuda_norm_1024_mode() {
    static const int mode = [] {
        const char * env = std::getenv("GGML_CUDA_NORM_1024_MODE");
        if (env != nullptr) {
            return std::atoi(env);
        }
        // Backward-compatible alias used while benchmarking the first warp-only variant.
        env = std::getenv("GGML_CUDA_NORM_1024_WARP");
        if (env != nullptr) {
            return std::atoi(env);
        }
        return 2;
    }();
    return mode;
}

static bool ggml_cuda_enable_norm_1024_affine_axis0() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_ENABLE_NORM_1024_AFFINE_AXIS0");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

template <int block_size,
          bool do_multiply = false,
          bool do_add = false,
          bool write_side = false,
          typename side_t = float>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps,
        const float * mul                  = nullptr,
        const int64_t mul_stride_row       = 0,
        const int64_t mul_stride_channel   = 0,
        const int64_t mul_stride_sample    = 0,
        const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
        const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
        const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
        const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
        const float * add                  = nullptr,
        const int64_t add_stride_row       = 0,
        const int64_t add_stride_channel   = 0,
        const int64_t add_stride_sample    = 0,
        const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
        const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
        const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
        const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
        side_t *      side_dst             = nullptr,
        const int64_t side_stride_row      = 0,
        const int64_t side_stride_channel  = 0,
        const int64_t side_stride_sample   = 0) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
    if constexpr (write_side) {
        side_dst += sample*side_stride_sample + channel*side_stride_channel + row*side_stride_row;
    }

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const uint32_t add_row     = fastmodulo(row, add_nrows_packed);
        const uint32_t add_channel = fastmodulo(channel, add_nchannels_packed);
        const uint32_t add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float norm = (x[col] - mean) * inv_std;
        if constexpr (do_multiply && do_add) {
            const uint32_t mul_col = fastmodulo(col, mul_ncols_packed);
            const uint32_t add_col = fastmodulo(col, add_ncols_packed);
            const float result = __fadd_rn(__fmul_rn(norm, mul[mul_col]), add[add_col]);
            dst[col] = result;
            if constexpr (write_side) {
                side_dst[col] = ggml_cuda_cast<side_t>(result);
            }
        } else if constexpr (do_multiply) {
            const uint32_t mul_col = fastmodulo(col, mul_ncols_packed);
            const float result = __fmul_rn(norm, mul[mul_col]);
            dst[col] = result;
            if constexpr (write_side) {
                side_dst[col] = ggml_cuda_cast<side_t>(result);
            }
        } else {
            dst[col] = norm;
            if constexpr (write_side) {
                side_dst[col] = ggml_cuda_cast<side_t>(norm);
            }
        }
    }
}

template <int block_size,
          bool do_multiply = false,
          bool do_add = false,
          bool preserve_residual = false,
          bool write_side = false,
          typename side_t = float,
          typename x1_t = float>
static __global__ void norm_residual_f32(
        const float * x0, const x1_t * x1, float * dst, const int ncols,
        const int64_t stride0_row, const int64_t stride0_channel, const int64_t stride0_sample,
        const int64_t stride1_row, const int64_t stride1_channel, const int64_t stride1_sample,
        const int64_t stride_dst_row, const int64_t stride_dst_channel, const int64_t stride_dst_sample,
        float * residual_dst,
        const int64_t stride_residual_row, const int64_t stride_residual_channel, const int64_t stride_residual_sample,
        const float eps,
        const float * mul                  = nullptr,
        const int64_t mul_stride_row       = 0,
        const int64_t mul_stride_channel   = 0,
        const int64_t mul_stride_sample    = 0,
        const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
        const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
        const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
        const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
        const float * add                  = nullptr,
        const int64_t add_stride_row       = 0,
        const int64_t add_stride_channel   = 0,
        const int64_t add_stride_sample    = 0,
        const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
        const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
        const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
        const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
        side_t *      side_dst             = nullptr,
        const int64_t stride_side_row      = 0,
        const int64_t stride_side_channel  = 0,
        const int64_t stride_side_sample   = 0) {
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x0  += sample*stride0_sample + channel*stride0_channel + row*stride0_row;
    x1  += sample*stride1_sample + channel*stride1_channel + row*stride1_row;
    dst += sample*stride_dst_sample + channel*stride_dst_channel + row*stride_dst_row;
    if constexpr (preserve_residual) {
        residual_dst += sample*stride_residual_sample + channel*stride_residual_channel + row*stride_residual_row;
    }
    if constexpr (write_side) {
        side_dst += sample*stride_side_sample + channel*stride_side_channel + row*stride_side_row;
    }

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const uint32_t add_row     = fastmodulo(row, add_nrows_packed);
        const uint32_t add_channel = fastmodulo(channel, add_nchannels_packed);
        const uint32_t add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __fadd_rn(x0[col], ggml_cuda_cast<float>(x1[col]));
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __fadd_rn(x0[col], ggml_cuda_cast<float>(x1[col]));
        if constexpr (preserve_residual) {
            residual_dst[col] = xi;
        }
        const float norm = (xi - mean) * inv_std;
        if constexpr (do_multiply && do_add) {
            const uint32_t mul_col = fastmodulo(col, mul_ncols_packed);
            const uint32_t add_col = fastmodulo(col, add_ncols_packed);
            const float result = __fadd_rn(__fmul_rn(norm, mul[mul_col]), add[add_col]);
            dst[col] = result;
            if constexpr (write_side) {
                side_dst[col] = (side_t) result;
            }
        } else if constexpr (do_multiply) {
            const uint32_t mul_col = fastmodulo(col, mul_ncols_packed);
            const float result = __fmul_rn(norm, mul[mul_col]);
            dst[col] = result;
            if constexpr (write_side) {
                side_dst[col] = (side_t) result;
            }
        } else {
            dst[col] = norm;
            if constexpr (write_side) {
                side_dst[col] = (side_t) norm;
            }
        }
    }
}

template <int block_size,
          bool preserve_residual = false,
          bool write_side = false,
          typename side_t = float,
          typename x1_t = float>
static __global__ void norm_residual_f32_1024_affine_axis0(
        const float * x0, const x1_t * x1, float * dst,
        const int64_t stride0_row, const int64_t stride0_channel, const int64_t stride0_sample,
        const int64_t stride1_row, const int64_t stride1_channel, const int64_t stride1_sample,
        const int64_t stride_dst_row, const int64_t stride_dst_channel, const int64_t stride_dst_sample,
        float * residual_dst,
        const int64_t stride_residual_row, const int64_t stride_residual_channel, const int64_t stride_residual_sample,
        const float eps,
        const float * mul,
        const float * add,
        side_t * side_dst = nullptr,
        const int64_t stride_side_row = 0,
        const int64_t stride_side_channel = 0,
        const int64_t stride_side_sample = 0) {
    constexpr int ncols = 1024;
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x0  += sample*stride0_sample + channel*stride0_channel + row*stride0_row;
    x1  += sample*stride1_sample + channel*stride1_channel + row*stride1_row;
    dst += sample*stride_dst_sample + channel*stride_dst_channel + row*stride_dst_row;
    if constexpr (preserve_residual) {
        residual_dst += sample*stride_residual_sample + channel*stride_residual_channel + row*stride_residual_row;
    }
    if constexpr (write_side) {
        side_dst += sample*stride_side_sample + channel*stride_side_channel + row*stride_side_row;
    }

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __fadd_rn(x0[col], ggml_cuda_cast<float>(x1[col]));
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __fadd_rn(x0[col], ggml_cuda_cast<float>(x1[col]));
        if constexpr (preserve_residual) {
            residual_dst[col] = xi;
        }
        const float norm = (xi - mean) * inv_std;
        const float result = __fadd_rn(__fmul_rn(norm, mul[col]), add[col]);
        dst[col] = result;
        if constexpr (write_side) {
            side_dst[col] = (side_t) result;
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024 || (ncols == 1024 && ggml_cuda_norm_1024_mode() == 1)) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else if (ncols == 1024 && ggml_cuda_norm_1024_mode() == 2) {
        const dim3 block_dims(256, 1, 1);
        norm_f32<256><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void norm_mul_f32_cuda(const float *  x,
                              const float *  mul,
                              const float *  add,
                              float *        dst,
                              const int      ncols,
                              const int      nrows,
                              const int      nchannels,
                              const int      nsamples,
                              const int64_t  stride_row,
                              const int64_t  stride_channel,
                              const int64_t  stride_sample,
                              const int64_t  mul_stride_row,
                              const int64_t  mul_stride_channel,
                              const int64_t  mul_stride_sample,
                              const uint32_t mul_ncols,
                              const uint32_t mul_nrows,
                              const uint32_t mul_nchannels,
                              const uint32_t mul_nsamples,
                              const int64_t  add_stride_row,
                              const int64_t  add_stride_channel,
                              const int64_t  add_stride_sample,
                              const uint32_t add_ncols,
                              const uint32_t add_nrows,
	                              const uint32_t add_nchannels,
	                              const uint32_t add_nsamples,
	                              const float    eps,
	                              cudaStream_t   stream,
	                              const ggml_type side_type = GGML_TYPE_COUNT,
	                              void *          side_dst = nullptr,
	                              const int64_t   side_stride_row = 0,
	                              const int64_t   side_stride_channel = 0,
	                              const int64_t   side_stride_sample = 0) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (add == nullptr) {
        if (ncols < 1024 || (ncols == 1024 && ggml_cuda_norm_1024_mode() == 1)) {
            const dim3 block_dims(WARP_SIZE, 1, 1);
            norm_f32<WARP_SIZE, true><<<blocks_num, block_dims, 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        } else if (ncols == 1024 && ggml_cuda_norm_1024_mode() == 2) {
            const dim3 block_dims(256, 1, 1);
            norm_f32<256, true><<<blocks_num, block_dims, 32 * sizeof(float2), stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            norm_f32<1024, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        }
	} else {
	    const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
	    const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
	    const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
	    const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);

#define GGML_CUDA_LAUNCH_NORM_MUL_ADD(BLOCK_SIZE, SHMEM_BYTES)                                 \
    do {                                                                                        \
        if (side_dst != nullptr && side_type == GGML_TYPE_BF16) {                                \
            norm_f32<BLOCK_SIZE, true, true, true, nv_bfloat16>                                  \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                               \
                    x,                                                                           \
                    dst,                                                                         \
                    ncols,                                                                       \
                    stride_row,                                                                  \
                    stride_channel,                                                              \
                    stride_sample,                                                               \
                    eps,                                                                         \
                    mul,                                                                         \
                    mul_stride_row,                                                              \
                    mul_stride_channel,                                                          \
                    mul_stride_sample,                                                           \
                    mul_ncols_packed,                                                            \
                    mul_nrows_packed,                                                            \
                    mul_nchannels_packed,                                                        \
                    mul_nsamples_packed,                                                         \
                    add,                                                                         \
                    add_stride_row,                                                              \
                    add_stride_channel,                                                          \
                    add_stride_sample,                                                           \
                    add_ncols_packed,                                                            \
                    add_nrows_packed,                                                            \
                    add_nchannels_packed,                                                        \
                    add_nsamples_packed,                                                         \
                    (nv_bfloat16 *) side_dst,                                                    \
                    side_stride_row,                                                             \
                    side_stride_channel,                                                         \
                    side_stride_sample);                                                         \
        } else if (side_dst != nullptr && side_type == GGML_TYPE_F16) {                           \
            norm_f32<BLOCK_SIZE, true, true, true, half>                                         \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                               \
                    x,                                                                           \
                    dst,                                                                         \
                    ncols,                                                                       \
                    stride_row,                                                                  \
                    stride_channel,                                                              \
                    stride_sample,                                                               \
                    eps,                                                                         \
                    mul,                                                                         \
                    mul_stride_row,                                                              \
                    mul_stride_channel,                                                          \
                    mul_stride_sample,                                                           \
                    mul_ncols_packed,                                                            \
                    mul_nrows_packed,                                                            \
                    mul_nchannels_packed,                                                        \
                    mul_nsamples_packed,                                                         \
                    add,                                                                         \
                    add_stride_row,                                                              \
                    add_stride_channel,                                                          \
                    add_stride_sample,                                                           \
                    add_ncols_packed,                                                            \
                    add_nrows_packed,                                                            \
                    add_nchannels_packed,                                                        \
                    add_nsamples_packed,                                                         \
                    (half *) side_dst,                                                           \
                    side_stride_row,                                                             \
                    side_stride_channel,                                                         \
                    side_stride_sample);                                                         \
        } else {                                                                                 \
            norm_f32<BLOCK_SIZE, true, true><<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(    \
                x,                                                                                \
                dst,                                                                              \
                ncols,                                                                            \
                stride_row,                                                                       \
                stride_channel,                                                                   \
                stride_sample,                                                                    \
                eps,                                                                              \
                mul,                                                                              \
                mul_stride_row,                                                                   \
                mul_stride_channel,                                                               \
                mul_stride_sample,                                                                \
                mul_ncols_packed,                                                                 \
                mul_nrows_packed,                                                                 \
                mul_nchannels_packed,                                                             \
                mul_nsamples_packed,                                                              \
                add,                                                                              \
                add_stride_row,                                                                   \
                add_stride_channel,                                                               \
                add_stride_sample,                                                                \
                add_ncols_packed,                                                                 \
                add_nrows_packed,                                                                 \
                add_nchannels_packed,                                                             \
                add_nsamples_packed);                                                             \
        }                                                                                         \
    } while (0)

	    if (ncols < 1024 || (ncols == 1024 && ggml_cuda_norm_1024_mode() == 1)) {
	        const dim3 block_dims(WARP_SIZE, 1, 1);
	        GGML_CUDA_LAUNCH_NORM_MUL_ADD(WARP_SIZE, 0);
	    } else if (ncols == 1024 && ggml_cuda_norm_1024_mode() == 2) {
	        const dim3 block_dims(256, 1, 1);
	        GGML_CUDA_LAUNCH_NORM_MUL_ADD(256, 32 * sizeof(float2));
	    } else {
	        const dim3 block_dims(1024, 1, 1);
	        GGML_CUDA_LAUNCH_NORM_MUL_ADD(
	            1024, block_dims.x > WARP_SIZE ? 32 * sizeof(float2) : 0);
	    }

#undef GGML_CUDA_LAUNCH_NORM_MUL_ADD
	}
}

static void norm_residual_mul_f32_cuda(const float *  x0,
                                       const void *   x1,
                                       const float *  mul,
                                       const float *  add,
                                       float *        dst,
                                       float *        residual_dst,
                                       const int      ncols,
                                       const int      nrows,
                                       const int      nchannels,
                                       const int      nsamples,
                                       const int64_t  stride0_row,
                                       const int64_t  stride0_channel,
                                       const int64_t  stride0_sample,
                                       const int64_t  stride1_row,
                                       const int64_t  stride1_channel,
                                       const int64_t  stride1_sample,
                                       const int64_t  stride_dst_row,
                                       const int64_t  stride_dst_channel,
                                       const int64_t  stride_dst_sample,
                                       const int64_t  stride_residual_row,
                                       const int64_t  stride_residual_channel,
                                       const int64_t  stride_residual_sample,
                                       const int64_t  mul_stride_row,
                                       const int64_t  mul_stride_channel,
                                       const int64_t  mul_stride_sample,
                                       const uint32_t mul_ncols,
                                       const uint32_t mul_nrows,
                                       const uint32_t mul_nchannels,
                                       const uint32_t mul_nsamples,
                                       const int64_t  add_stride_row,
                                       const int64_t  add_stride_channel,
                                       const int64_t  add_stride_sample,
                                       const uint32_t add_ncols,
                                       const uint32_t add_nrows,
                                       const uint32_t add_nchannels,
                                       const uint32_t add_nsamples,
                                       const float    eps,
                                       cudaStream_t   stream,
                                       const ggml_type x1_type = GGML_TYPE_F32,
                                       const ggml_type side_type = GGML_TYPE_COUNT,
                                       void *          side_dst = nullptr,
                                       const int64_t   stride_side_row = 0,
                                       const int64_t   stride_side_channel = 0,
                                       const int64_t   stride_side_sample = 0) {
    const dim3 blocks_num(nrows, nchannels, nsamples);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
    const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
    const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
    const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);

#define GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0_TYPED(                                      \
    BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, X1_T)                                                \
    do {                                                                                             \
        const X1_T * x1_typed = static_cast<const X1_T *>(x1);                                       \
        if (side_dst != nullptr && side_type == GGML_TYPE_BF16) {                                     \
            norm_residual_f32_1024_affine_axis0<                                                     \
                BLOCK_SIZE, PRESERVE_RESIDUAL, true, nv_bfloat16, X1_T>                              \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                                   \
                    x0,                                                                               \
                    x1_typed,                                                                         \
                    dst,                                                                              \
                    stride0_row,                                                                      \
                    stride0_channel,                                                                  \
                    stride0_sample,                                                                   \
                    stride1_row,                                                                      \
                    stride1_channel,                                                                  \
                    stride1_sample,                                                                   \
                    stride_dst_row,                                                                   \
                    stride_dst_channel,                                                               \
                    stride_dst_sample,                                                                \
                    residual_dst,                                                                     \
                    stride_residual_row,                                                              \
                    stride_residual_channel,                                                          \
                    stride_residual_sample,                                                           \
                    eps,                                                                              \
                    mul,                                                                              \
                    add,                                                                              \
                    (nv_bfloat16 *) side_dst,                                                         \
                    stride_side_row,                                                                  \
                    stride_side_channel,                                                              \
                    stride_side_sample);                                                              \
        } else if (side_dst != nullptr && side_type == GGML_TYPE_F16) {                               \
            norm_residual_f32_1024_affine_axis0<                                                     \
                BLOCK_SIZE, PRESERVE_RESIDUAL, true, half, X1_T>                                     \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                                    \
                    x0,                                                                               \
                    x1_typed,                                                                         \
                    dst,                                                                              \
                    stride0_row,                                                                      \
                    stride0_channel,                                                                  \
                    stride0_sample,                                                                   \
                    stride1_row,                                                                      \
                    stride1_channel,                                                                  \
                    stride1_sample,                                                                   \
                    stride_dst_row,                                                                   \
                    stride_dst_channel,                                                               \
                    stride_dst_sample,                                                                \
                    residual_dst,                                                                     \
                    stride_residual_row,                                                              \
                    stride_residual_channel,                                                          \
                    stride_residual_sample,                                                           \
                    eps,                                                                              \
                    mul,                                                                              \
                    add,                                                                              \
                    (half *) side_dst,                                                                \
                    stride_side_row,                                                                  \
                    stride_side_channel,                                                              \
                    stride_side_sample);                                                              \
        } else {                                                                                     \
            norm_residual_f32_1024_affine_axis0<                                                     \
                BLOCK_SIZE, PRESERVE_RESIDUAL, false, float, X1_T>                                   \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                                    \
                    x0,                                                                               \
                    x1_typed,                                                                         \
                    dst,                                                                              \
                    stride0_row,                                                                      \
                    stride0_channel,                                                                  \
                    stride0_sample,                                                                   \
                    stride1_row,                                                                      \
                    stride1_channel,                                                                  \
                    stride1_sample,                                                                   \
                    stride_dst_row,                                                                   \
                    stride_dst_channel,                                                               \
                    stride_dst_sample,                                                                \
                    residual_dst,                                                                     \
                    stride_residual_row,                                                              \
                    stride_residual_channel,                                                          \
                    stride_residual_sample,                                                           \
                    eps,                                                                              \
                    mul,                                                                              \
                    add);                                                                             \
        }                                                                                            \
    } while (0)

#define GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0(BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES) \
    do {                                                                                             \
        if (x1_type == GGML_TYPE_BF16) {                                                             \
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0_TYPED(                                  \
                BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, nv_bfloat16);                            \
        } else if (x1_type == GGML_TYPE_F16) {                                                       \
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0_TYPED(                                  \
                BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, half);                                   \
        } else {                                                                                     \
            GGML_ASSERT(x1_type == GGML_TYPE_F32);                                                   \
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0_TYPED(                                  \
                BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, float);                                  \
        }                                                                                            \
    } while (0)

#define GGML_CUDA_LAUNCH_NORM_RESIDUAL_TYPED(BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, X1_T)      \
    do {                                                                                            \
        const X1_T * x1_typed = static_cast<const X1_T *>(x1);                                      \
        if (side_dst != nullptr && side_type == GGML_TYPE_BF16) {                                    \
            norm_residual_f32<BLOCK_SIZE, true, true, PRESERVE_RESIDUAL, true, nv_bfloat16, X1_T>   \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                                  \
                    x0,                                                                             \
                    x1_typed,                                                                       \
                    dst,                                                                            \
                    ncols,                                                                          \
                    stride0_row,                                                                    \
                    stride0_channel,                                                                \
                    stride0_sample,                                                                 \
                    stride1_row,                                                                    \
                    stride1_channel,                                                                \
                    stride1_sample,                                                                 \
                    stride_dst_row,                                                                 \
                    stride_dst_channel,                                                             \
                    stride_dst_sample,                                                              \
                    residual_dst,                                                                   \
                    stride_residual_row,                                                            \
                    stride_residual_channel,                                                        \
                    stride_residual_sample,                                                         \
                    eps,                                                                            \
                    mul,                                                                            \
                    mul_stride_row,                                                                 \
                    mul_stride_channel,                                                             \
                    mul_stride_sample,                                                              \
                    mul_ncols_packed,                                                               \
                    mul_nrows_packed,                                                               \
                    mul_nchannels_packed,                                                           \
                    mul_nsamples_packed,                                                            \
                    add,                                                                            \
                    add_stride_row,                                                                 \
                    add_stride_channel,                                                             \
                    add_stride_sample,                                                              \
                    add_ncols_packed,                                                               \
                    add_nrows_packed,                                                               \
                    add_nchannels_packed,                                                           \
                    add_nsamples_packed,                                                            \
                    (nv_bfloat16 *) side_dst,                                                       \
                    stride_side_row,                                                                \
                    stride_side_channel,                                                            \
                    stride_side_sample);                                                            \
        } else if (side_dst != nullptr && side_type == GGML_TYPE_F16) {                              \
            norm_residual_f32<BLOCK_SIZE, true, true, PRESERVE_RESIDUAL, true, half, X1_T>          \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                                  \
                    x0,                                                                             \
                    x1_typed,                                                                       \
                    dst,                                                                            \
                    ncols,                                                                          \
                    stride0_row,                                                                    \
                    stride0_channel,                                                                \
                    stride0_sample,                                                                 \
                    stride1_row,                                                                    \
                    stride1_channel,                                                                \
                    stride1_sample,                                                                 \
                    stride_dst_row,                                                                 \
                    stride_dst_channel,                                                             \
                    stride_dst_sample,                                                              \
                    residual_dst,                                                                   \
                    stride_residual_row,                                                            \
                    stride_residual_channel,                                                        \
                    stride_residual_sample,                                                         \
                    eps,                                                                            \
                    mul,                                                                            \
                    mul_stride_row,                                                                 \
                    mul_stride_channel,                                                             \
                    mul_stride_sample,                                                              \
                    mul_ncols_packed,                                                               \
                    mul_nrows_packed,                                                               \
                    mul_nchannels_packed,                                                           \
                    mul_nsamples_packed,                                                            \
                    add,                                                                            \
                    add_stride_row,                                                                 \
                    add_stride_channel,                                                             \
                    add_stride_sample,                                                              \
                    add_ncols_packed,                                                               \
                    add_nrows_packed,                                                               \
                    add_nchannels_packed,                                                           \
                    add_nsamples_packed,                                                            \
                    (half *) side_dst,                                                              \
                    stride_side_row,                                                                \
                    stride_side_channel,                                                            \
                    stride_side_sample);                                                            \
        } else {                                                                                    \
            norm_residual_f32<BLOCK_SIZE, true, true, PRESERVE_RESIDUAL, false, float, X1_T>        \
                <<<blocks_num, block_dims, SHMEM_BYTES, stream>>>(                                  \
                    x0,                                                                             \
                    x1_typed,                                                                       \
                    dst,                                                                            \
                    ncols,                                                                          \
                    stride0_row,                                                                    \
                    stride0_channel,                                                                \
                    stride0_sample,                                                                 \
                    stride1_row,                                                                    \
                    stride1_channel,                                                                \
                    stride1_sample,                                                                 \
                    stride_dst_row,                                                                 \
                    stride_dst_channel,                                                             \
                    stride_dst_sample,                                                              \
                    residual_dst,                                                                   \
                    stride_residual_row,                                                            \
                    stride_residual_channel,                                                        \
                    stride_residual_sample,                                                         \
                    eps,                                                                            \
                    mul,                                                                            \
                    mul_stride_row,                                                                 \
                    mul_stride_channel,                                                             \
                    mul_stride_sample,                                                              \
                    mul_ncols_packed,                                                               \
                    mul_nrows_packed,                                                               \
                    mul_nchannels_packed,                                                           \
                    mul_nsamples_packed,                                                            \
                    add,                                                                            \
                    add_stride_row,                                                                 \
                    add_stride_channel,                                                             \
                    add_stride_sample,                                                              \
                    add_ncols_packed,                                                               \
                    add_nrows_packed,                                                               \
                    add_nchannels_packed,                                                           \
                    add_nsamples_packed);                                                           \
        }                                                                                           \
    } while (0)

#define GGML_CUDA_LAUNCH_NORM_RESIDUAL(BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES)                 \
    do {                                                                                            \
        if (x1_type == GGML_TYPE_BF16) {                                                            \
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_TYPED(                                                   \
                BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, nv_bfloat16);                           \
        } else if (x1_type == GGML_TYPE_F16) {                                                      \
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_TYPED(BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, half); \
        } else {                                                                                    \
            GGML_ASSERT(x1_type == GGML_TYPE_F32);                                                  \
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_TYPED(BLOCK_SIZE, PRESERVE_RESIDUAL, SHMEM_BYTES, float);\
        }                                                                                           \
    } while (0)

    const bool affine_axis0_1024 =
        ggml_cuda_enable_norm_1024_affine_axis0() &&
        ncols == 1024 &&
        mul_ncols == 1024 && mul_nrows == 1 && mul_nchannels == 1 && mul_nsamples == 1 &&
        add_ncols == 1024 && add_nrows == 1 && add_nchannels == 1 && add_nsamples == 1;

    if (ncols < 1024 || (ncols == 1024 && ggml_cuda_norm_1024_mode() == 1)) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        if (affine_axis0_1024 && residual_dst) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0(WARP_SIZE, true, 0);
        } else if (affine_axis0_1024) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0(WARP_SIZE, false, 0);
        } else if (residual_dst) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL(WARP_SIZE, true, 0);
        } else {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL(WARP_SIZE, false, 0);
        }
    } else if (ncols == 1024 && ggml_cuda_norm_1024_mode() == 2) {
        const dim3 block_dims(256, 1, 1);
        if (affine_axis0_1024 && residual_dst) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0(256, true, 32 * sizeof(float2));
        } else if (affine_axis0_1024) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0(256, false, 32 * sizeof(float2));
        } else if (residual_dst) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL(256, true, 32 * sizeof(float2));
        } else {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL(256, false, 32 * sizeof(float2));
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        if (affine_axis0_1024 && residual_dst) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0(1024, true, block_dims.x > WARP_SIZE ? 32 * sizeof(float2) : 0);
        } else if (affine_axis0_1024) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0(1024, false, block_dims.x > WARP_SIZE ? 32 * sizeof(float2) : 0);
        } else if (residual_dst) {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL(1024, true, block_dims.x > WARP_SIZE ? 32 * sizeof(float2) : 0);
        } else {
            GGML_CUDA_LAUNCH_NORM_RESIDUAL(1024, false, block_dims.x > WARP_SIZE ? 32 * sizeof(float2) : 0);
        }
    }

#undef GGML_CUDA_LAUNCH_NORM_RESIDUAL
#undef GGML_CUDA_LAUNCH_NORM_RESIDUAL_TYPED
#undef GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0
#undef GGML_CUDA_LAUNCH_NORM_RESIDUAL_1024_AFFINE_AXIS0_TYPED
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        rms_norm_f32<256, false><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024, false><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32<256, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32<1024, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32<256, true, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32<1024, true, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        l2_norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        l2_norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = norm_src->ne[0];
    const int64_t ne01 = norm_src->ne[1];
    const int64_t ne02 = norm_src->ne[2];
    const int64_t ne03 = norm_src->ne[3];

    const size_t ts0 = ggml_type_size(norm_src->type);
    GGML_ASSERT(norm_src->nb[0] == ts0);
    const int64_t s01 = norm_src->nb[1] / ts0;
    const int64_t s02 = norm_src->nb[2] / ts0;
    const int64_t s03 = norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                      ne00, ne01, ne02, ne03,
                      /*s00*/ s01, s02, s03,
                      /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                      mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                      /*add_s00*/ 0, 0, 0,
                      0, 0, 0, 0,
                      eps, stream);
}

void ggml_cuda_op_norm_fused_add(ggml_backend_cuda_context & ctx,
                                 ggml_tensor *               dst,
                                 ggml_tensor *               mul_tensor,
                                 ggml_tensor *               add_tensor,
                                 ggml_tensor *               cpy_tensor) {
    const ggml_tensor * norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float * add_d = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = norm_src->ne[0];
    const int64_t ne01 = norm_src->ne[1];
    const int64_t ne02 = norm_src->ne[2];
    const int64_t ne03 = norm_src->ne[3];

    const size_t ts0 = ggml_type_size(norm_src->type);
    GGML_ASSERT(norm_src->nb[0] == ts0);
    const int64_t s01 = norm_src->nb[1] / ts0;
    const int64_t s02 = norm_src->nb[2] / ts0;
    const int64_t s03 = norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    ggml_type side_type = GGML_TYPE_COUNT;
    void * side_dst = nullptr;
    int64_t side_s01 = 0;
    int64_t side_s02 = 0;
    int64_t side_s03 = 0;
    if (cpy_tensor != nullptr) {
        ggml_tensor * cpy_dst = cpy_tensor->src[1];
        if (cpy_tensor->src[0] == add_tensor && cpy_dst != nullptr &&
            (cpy_dst->type == GGML_TYPE_BF16 || cpy_dst->type == GGML_TYPE_F16) &&
            ggml_are_same_shape(cpy_dst, add_tensor) && ggml_is_contiguous(cpy_dst)) {
            side_type = cpy_dst->type;
            side_dst = cpy_dst->data;
            const size_t ts_side = ggml_type_size(cpy_dst->type);
            GGML_ASSERT(cpy_dst->nb[0] == ts_side);
            side_s01 = cpy_dst->nb[1] / ts_side;
            side_s02 = cpy_dst->nb[2] / ts_side;
            side_s03 = cpy_dst->nb[3] / ts_side;
        }
    }

    norm_mul_f32_cuda(src0_d, mul_d, add_d, dst_d,
                      ne00, ne01, ne02, ne03,
                      /*s00*/ s01, s02, s03,
                      /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                      mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                      /*add_s00*/ add_s01, add_s02, add_s03,
                      add_ncols, add_nrows, add_nchannels, add_nsamples,
                      eps, stream, side_type, side_dst, side_s01, side_s02, side_s03);
}

void ggml_cuda_op_add_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               add_input,
                                     ggml_tensor *               norm_tensor,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     const bool                  preserve_add_output,
                                     ggml_tensor *               cpy_tensor) {
    const ggml_tensor * add_src0 = add_input->src[0];
    const ggml_tensor * add_src1 = add_input->src[1];
    GGML_ASSERT(norm_tensor->src[0] == add_input);

    float eps = 0.0f;
    memcpy(&eps, norm_tensor->op_params, sizeof(float));

    const float * src0_d = (const float *) add_src0->data;
    const float * src1_d = (const float *) add_src1->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == norm_tensor) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == norm_tensor) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float * affine_add_d = nullptr;
    const ggml_tensor * affine_add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        affine_add_d = (float *) add_tensor->src[1]->data;
        affine_add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        affine_add_d = (float *) add_tensor->src[0]->data;
        affine_add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(add_src0->type == GGML_TYPE_F32);
    GGML_ASSERT(add_src1->type == GGML_TYPE_F32 || add_src1->type == GGML_TYPE_BF16 ||
                add_src1->type == GGML_TYPE_F16);
    GGML_ASSERT(add_input->type == GGML_TYPE_F32);
    GGML_ASSERT(norm_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = add_src0->ne[0];
    const int64_t ne01 = add_src0->ne[1];
    const int64_t ne02 = add_src0->ne[2];
    const int64_t ne03 = add_src0->ne[3];

    const size_t ts0 = ggml_type_size(add_src0->type);
    const size_t ts1 = ggml_type_size(add_src1->type);
    const size_t ts_dst = ggml_type_size(add_tensor->type);
    const size_t ts_residual = ggml_type_size(add_input->type);
    GGML_ASSERT(add_src0->nb[0] == ts0);
    GGML_ASSERT(add_src1->nb[0] == ts1);
    GGML_ASSERT(add_tensor->nb[0] == ts_dst);
    GGML_ASSERT(add_input->nb[0] == ts_residual);
    const int64_t s0_01 = add_src0->nb[1] / ts0;
    const int64_t s0_02 = add_src0->nb[2] / ts0;
    const int64_t s0_03 = add_src0->nb[3] / ts0;
    const int64_t s1_01 = add_src1->nb[1] / ts1;
    const int64_t s1_02 = add_src1->nb[2] / ts1;
    const int64_t s1_03 = add_src1->nb[3] / ts1;
    const int64_t sd_01 = add_tensor->nb[1] / ts_dst;
    const int64_t sd_02 = add_tensor->nb[2] / ts_dst;
    const int64_t sd_03 = add_tensor->nb[3] / ts_dst;
    const int64_t sr_01 = add_input->nb[1] / ts_residual;
    const int64_t sr_02 = add_input->nb[2] / ts_residual;
    const int64_t sr_03 = add_input->nb[3] / ts_residual;

    ggml_type side_type = GGML_TYPE_COUNT;
    void * side_dst = nullptr;
    int64_t ss_01 = 0;
    int64_t ss_02 = 0;
    int64_t ss_03 = 0;
    if (cpy_tensor != nullptr) {
        ggml_tensor * cpy_dst = cpy_tensor->src[1];
        if (cpy_tensor->src[0] == add_tensor && cpy_dst != nullptr &&
            (cpy_dst->type == GGML_TYPE_BF16 || cpy_dst->type == GGML_TYPE_F16) &&
            ggml_are_same_shape(cpy_dst, add_tensor) && ggml_is_contiguous(cpy_dst)) {
            side_type = cpy_dst->type;
            side_dst = cpy_dst->data;
            const size_t ts_side = ggml_type_size(cpy_dst->type);
            GGML_ASSERT(cpy_dst->nb[0] == ts_side);
            ss_01 = cpy_dst->nb[1] / ts_side;
            ss_02 = cpy_dst->nb[2] / ts_side;
            ss_03 = cpy_dst->nb[3] / ts_side;
        }
    }

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(affine_add_src->type);
    GGML_ASSERT(affine_add_src->nb[0] == ts_add);
    const int64_t add_s01 = affine_add_src->nb[1] / ts_add;
    const int64_t add_s02 = affine_add_src->nb[2] / ts_add;
    const int64_t add_s03 = affine_add_src->nb[3] / ts_add;

    const int add_ncols     = affine_add_src->ne[0];
    const int add_nrows     = affine_add_src->ne[1];
    const int add_nchannels = affine_add_src->ne[2];
    const int add_nsamples  = affine_add_src->ne[3];

    norm_residual_mul_f32_cuda(src0_d, src1_d, mul_d, affine_add_d, dst_d,
                               preserve_add_output ? (float *) add_input->data : nullptr,
                               ne00, ne01, ne02, ne03,
                               s0_01, s0_02, s0_03,
                               s1_01, s1_02, s1_03,
                               sd_01, sd_02, sd_03,
                               sr_01, sr_02, sr_03,
                               mul_s01, mul_s02, mul_s03,
                               mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                               add_s01, add_s02, add_s03,
                               add_ncols, add_nrows, add_nchannels, add_nsamples,
                               eps, stream, add_src1->type, side_type, side_dst, ss_01, ss_02, ss_03);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
