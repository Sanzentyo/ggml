#include "conv2d-transpose.cuh"
#include "convert.cuh"

#include <cstdlib>

static bool ggml_cuda_conv_transpose_k2s2_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_CONV_TRANSPOSE_K2S2");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}

static bool ggml_cuda_conv_transpose_k2s2_gemm_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_CONV_TRANSPOSE_K2S2_GEMM");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}

template <typename kernel_t, bool fuse_bias>
static __global__ void conv2d_transpose_kernel(const float * __restrict__ input,
                                               const kernel_t * __restrict__ kernel,
                                               const float * __restrict__ bias,
                                               float * __restrict__ conv_output,
                                               float * __restrict__ add_output,
                                               const int in_w,
                                               const int in_h,
                                               const int out_w,
                                               const int out_h,
                                               const int kernel_w,
                                               const int kernel_h,
                                               const int stride,
                                               const int c_in,
                                               const int c_out,
                                               const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_elements = out_w * out_h * c_out * batches;

    if (global_idx >= total_elements) {
        return;
    }

    const int out_x_idx = global_idx % out_w;
    const int out_y_idx = (global_idx / out_w) % out_h;
    const int c_idx     = (global_idx / (out_w * out_h)) % c_out;
    const int n_idx     = global_idx / (out_w * out_h * c_out);

    float accumulator = 0;
    // For each output idx, find the inputs that contribute to it by checking stride alignment and bounds

    for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int in_y = out_y_idx - kh;
            if (in_y < 0 || in_y % stride) {
                continue;
            }
            in_y /= stride;
            if (in_y >= in_h) {
                continue;
            }

            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_x = out_x_idx - kw;
                if (in_x < 0 || in_x % stride) {
                    continue;
                }
                in_x /= stride;
                if (in_x >= in_w) {
                    continue;
                }

                const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + (in_w) *in_y + in_x;
                const int kernel_idx =
                    (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx + (kernel_w) *kh + kw;

                float    input_val = input[input_idx];
                kernel_t kern_val  = kernel[kernel_idx];

                accumulator += input_val * ggml_cuda_cast<float>(kern_val);
            }
        }
    }

    const int output_idx = (out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + (out_w) * out_y_idx + out_x_idx;
    conv_output[output_idx] = accumulator;
    if constexpr (fuse_bias) {
        add_output[output_idx] = accumulator + bias[c_idx];
    }
}

template <typename kernel_t, bool fuse_bias>
static __global__ void conv2d_transpose_k2_s2_kernel(const float * __restrict__ input,
                                                     const kernel_t * __restrict__ kernel,
                                                     const float * __restrict__ bias,
                                                     float * __restrict__ conv_output,
                                                     float * __restrict__ add_output,
                                                     const int in_w,
                                                     const int in_h,
                                                     const int out_w,
                                                     const int out_h,
                                                     const int c_in,
                                                     const int c_out,
                                                     const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_w * out_h * c_out * batches;
    if (global_idx >= total_elements) {
        return;
    }

    const int out_x_idx = global_idx % out_w;
    const int out_y_idx = (global_idx / out_w) % out_h;
    const int c_idx     = (global_idx / (out_w * out_h)) % c_out;
    const int n_idx     = global_idx / (out_w * out_h * c_out);
    const int output_idx = (out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + (out_w) * out_y_idx + out_x_idx;

    const int in_x = out_x_idx >> 1;
    const int in_y = out_y_idx >> 1;
    if (in_x >= in_w || in_y >= in_h) {
        conv_output[output_idx] = 0.0f;
        if constexpr (fuse_bias) {
            add_output[output_idx] = bias[c_idx];
        }
        return;
    }

    const int kw = out_x_idx & 1;
    const int kh = out_y_idx & 1;
    float accumulator = 0.0f;
    for (int c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
        const int kernel_idx = 4 * c_out * c_in_idx + 4 * c_idx + 2 * kh + kw;
        accumulator += input[input_idx] * ggml_cuda_cast<float>(kernel[kernel_idx]);
    }

    conv_output[output_idx] = accumulator;
    if constexpr (fuse_bias) {
        add_output[output_idx] = accumulator + bias[c_idx];
    }
}

template <bool fuse_bias>
static __global__ void conv2d_transpose_k2_s2_scatter_kernel(const float * __restrict__ tmp,
                                                             const float * __restrict__ bias,
                                                             float * __restrict__ conv_output,
                                                             float * __restrict__ add_output,
                                                             const int in_w,
                                                             const int in_h,
                                                             const int out_w,
                                                             const int out_h,
                                                             const int c_out,
                                                             const int batches,
                                                             const int kh,
                                                             const int kw) {
    const int spatial = in_w * in_h;
    const int total = spatial * c_out * batches;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total) {
        return;
    }

    const int c_idx = global_idx % c_out;
    const int s_idx = (global_idx / c_out) % spatial;
    const int n_idx = global_idx / (spatial * c_out);
    const int in_x = s_idx % in_w;
    const int in_y = s_idx / in_w;
    const int out_x = 2 * in_x + kw;
    const int out_y = 2 * in_y + kh;
    if (out_x >= out_w || out_y >= out_h) {
        return;
    }

    const int out_spatial = out_w * out_h;
    const int output_idx = (out_spatial * c_out) * n_idx + out_spatial * c_idx + out_w * out_y + out_x;
    const float value = tmp[c_idx + c_out * s_idx + c_out * spatial * n_idx];
    conv_output[output_idx] = value;
    if constexpr (fuse_bias) {
        add_output[output_idx] = value + bias[c_idx];
    }
}

template <typename kernel_t>
static __global__ void conv2d_transpose_k2_s2_pack_kernel(const kernel_t * __restrict__ kernel,
                                                          kernel_t * __restrict__ packed,
                                                          const int c_in,
                                                          const int c_out,
                                                          const int kh,
                                                          const int kw) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = c_in * c_out;
    if (global_idx >= total) {
        return;
    }
    const int c_idx = global_idx % c_out;
    const int c_in_idx = global_idx / c_out;
    const int q = 2 * kh + kw;
    packed[c_idx + c_out * c_in_idx] = kernel[4 * c_out * c_in_idx + 4 * c_idx + q];
}

//input is (W, H, C_in, N), Kernel is (W, H, C_out, C_in)
void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const ggml_tensor * bias   = nullptr;

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    if (bias_add_node != nullptr) {
        GGML_ASSERT(bias_add_node->type == GGML_TYPE_F32);
        GGML_ASSERT(bias_add_node->op == GGML_OP_ADD);
        if (bias_add_node->src[0] == dst) {
            bias = bias_add_node->src[1];
        } else {
            GGML_ASSERT(bias_add_node->src[1] == dst);
            bias = bias_add_node->src[0];
        }
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(bias->ne[0] == 1 && bias->ne[1] == 1 && bias->ne[2] == kernel->ne[2] && bias->ne[3] == 1);
        GGML_ASSERT(ggml_are_same_shape(dst, bias_add_node));
    }

    const float * input_data  = (const float *) input->data;
    const float * bias_data   = bias != nullptr ? (const float *) bias->data : nullptr;
    float *       conv_output_data = (float *) dst->data;
    float *       add_output_data  = bias_add_node != nullptr ? (float *) bias_add_node->data : nullptr;
    const void *  kernel_data = kernel->data;

    const int input_w      = input->ne[0];
    const int input_h      = input->ne[1];
    const int output_w     = dst->ne[0];
    const int output_h     = dst->ne[1];
    const int channels_in  = input->ne[2];
    const int channels_out = kernel->ne[2];
    const int kernel_w     = kernel->ne[0];
    const int kernel_h     = kernel->ne[1];
    const int stride       = dst->op_params[0];
    const int batches      = input->ne[3];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    cudaStream_t st = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(bias_add_node == nullptr || ggml_is_contiguous(bias_add_node));

    const int total  = output_w * output_h * channels_out * batches;
    const int blocks = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

    if (kernel_w == 2 && kernel_h == 2 && stride == 2 && ggml_cuda_conv_transpose_k2s2_enabled()) {
        if (batches == 1 && kernel->type == GGML_TYPE_F16 && ggml_cuda_conv_transpose_k2s2_gemm_enabled()) {
            const int spatial = input_w * input_h;
            ggml_cuda_pool_alloc<float> tmp(ctx.pool(), (size_t) channels_out * spatial * batches);
            ggml_cuda_pool_alloc<half> input_f16(ctx.pool(), (size_t) channels_in * spatial * batches);
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            to_fp16_cuda(input_data, input_f16.get(), (int64_t) channels_in * spatial * batches, st);

            cublasHandle_t handle = ctx.cublas_handle();
            CUBLAS_CHECK(cublasSetStream(handle, st));
            const float alpha = 1.0f;
            const float beta  = 0.0f;

            const int scatter_total = spatial * channels_out * batches;
            const int scatter_blocks =
                (scatter_total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;
            const int pack_total = channels_in * channels_out;
            const int pack_blocks =
                (pack_total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

            for (int kh = 0; kh < 2; ++kh) {
                for (int kw = 0; kw < 2; ++kw) {
                    const void * kernel_q = nullptr;
                    ggml_cuda_pool_alloc<half> packed_f16(ctx.pool());
                    packed_f16.alloc(pack_total);
                    conv2d_transpose_k2_s2_pack_kernel<half><<<pack_blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                        (const half *) kernel_data, packed_f16.get(), channels_in, channels_out, kh, kw);
                    kernel_q = packed_f16.get();

                    CUBLAS_CHECK(cublasGemmEx(handle,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              channels_out,
                                              spatial * batches,
                                              channels_in,
                                              &alpha,
                                              kernel_q,
                                              CUDA_R_16F,
                                              channels_out,
                                              input_f16.get(),
                                              CUDA_R_16F,
                                              spatial,
                                              &beta,
                                              tmp.get(),
                                              CUDA_R_32F,
                                              channels_out,
                                              CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT));

                    if (bias_data != nullptr) {
                        conv2d_transpose_k2_s2_scatter_kernel<true><<<scatter_blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                            tmp.get(), bias_data, conv_output_data, add_output_data, input_w, input_h, output_w, output_h,
                            channels_out, batches, kh, kw);
                    } else {
                        conv2d_transpose_k2_s2_scatter_kernel<false><<<scatter_blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                            tmp.get(), nullptr, conv_output_data, nullptr, input_w, input_h, output_w, output_h,
                            channels_out, batches, kh, kw);
                    }
                }
            }
            return;
        }

        if (kernel->type == GGML_TYPE_F16) {
            if (bias_data != nullptr) {
                conv2d_transpose_k2_s2_kernel<half, true><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                    input_data, (const half *) kernel_data, bias_data, conv_output_data, add_output_data, input_w, input_h, output_w, output_h,
                    channels_in, channels_out, batches);
            } else {
                conv2d_transpose_k2_s2_kernel<half, false><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                    input_data, (const half *) kernel_data, nullptr, conv_output_data, nullptr, input_w, input_h, output_w, output_h,
                    channels_in, channels_out, batches);
            }
        } else {
            if (bias_data != nullptr) {
                conv2d_transpose_k2_s2_kernel<float, true><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                    input_data, (const float *) kernel_data, bias_data, conv_output_data, add_output_data, input_w, input_h, output_w, output_h,
                    channels_in, channels_out, batches);
            } else {
                conv2d_transpose_k2_s2_kernel<float, false><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                    input_data, (const float *) kernel_data, nullptr, conv_output_data, nullptr, input_w, input_h, output_w, output_h,
                    channels_in, channels_out, batches);
            }
        }
        return;
    }

    if (kernel->type == GGML_TYPE_F16) {
        if (bias_data != nullptr) {
            conv2d_transpose_kernel<half, true><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                input_data, (const half *) kernel_data, bias_data, conv_output_data, add_output_data, input_w, input_h, output_w, output_h,
                kernel_w, kernel_h, stride, channels_in, channels_out, batches);
        } else {
            conv2d_transpose_kernel<half, false><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                input_data, (const half *) kernel_data, nullptr, conv_output_data, nullptr, input_w, input_h, output_w, output_h,
                kernel_w, kernel_h, stride, channels_in, channels_out, batches);
        }

    } else {
        if (bias_data != nullptr) {
            conv2d_transpose_kernel<float, true><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                input_data, (const float *) kernel_data, bias_data, conv_output_data, add_output_data, input_w, input_h, output_w, output_h,
                kernel_w, kernel_h, stride, channels_in, channels_out, batches);
        } else {
            conv2d_transpose_kernel<float, false><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                input_data, (const float *) kernel_data, nullptr, conv_output_data, nullptr, input_w, input_h, output_w, output_h,
                kernel_w, kernel_h, stride, channels_in, channels_out, batches);
        }
    }
}
