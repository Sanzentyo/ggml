#include "binbcast.cuh"
#include "mmq.cuh"
#include "unary.cuh"
#include <cstdlib>
#include <cstdint>
#include <utility>

static __device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_add_gelu(const float a, const float b) {
    return ggml_cuda_op_gelu_single(a + b);
}

static __device__ __forceinline__ float op_add_gelu_erf(const float a, const float b) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    const float x = a + b;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_add_gelu_quick(const float a, const float b) {
    const float GELU_QUICK_COEF = -1.702f;
    const float x = a + b;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast(const src0_t *         src0,
                                   const src1_t *         src1,
                                   dst_t *                dst,
                                   const int              ne0,
                                   const int              ne1,
                                   const int              ne2,
                                   const uint3            ne3,
                                   const uint3            ne10,
                                   const uint3            ne11,
                                   const uint3            ne12,
                                   const uint3            ne13,
                                 /*const int              s0,*/
                                   const int              s1,
                                   const int              s2,
                                   const int              s3,
                                   const int              s00,
                                   const int              s01,
                                   const int              s02,
                                   const int              s03,
                                   const int              s10,
                                   const int              s11,
                                   const int              s12,
                                   const int              s13,
                                   src1_ptrs... src1s) {
    const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
    const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
    const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);

    if (i0s >= (uint32_t)ne0 || i1 >= (uint32_t)ne1 || i2 >= (uint32_t)ne2 || i3 >= ne3.z) {
        return;
    }

    const uint32_t i11 = fastmodulo(i1, ne11);
    const uint32_t i12 = fastmodulo(i2, ne12);
    const uint32_t i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x * gridDim.x) {
        const uint32_t i10 = fastmodulo(i0, ne10);

        float result = src0_row ? (float) src0_row[i0*s00] : 0.0f;
        if constexpr (sizeof...(src1_ptrs) > 0) {
            result = (..., (result = bin_op(result, (float)src1s[i_src1 + i10*s10])));
        } else {
            result = bin_op(result, (float)src1[i_src1 + i10*s10]);
        }

        dst_row[i0] = (dst_t) result;
    }
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast_unravel(const src0_t *         src0,
                                           const src1_t *         src1,
                                           dst_t *                dst,
                                           const uint3            ne0,
                                           const uint3            ne1,
                                           const uint3            ne2,
                                           const uint32_t         ne3,
                                           const uint3            prod_012,
                                           const uint3            prod_01,
                                           const uint3            ne10,
                                           const uint3            ne11,
                                           const uint3            ne12,
                                           const uint3            ne13,
                                         /*const int              s0,*/
                                           const int              s1,
                                           const int              s2,
                                           const int              s3,
                                           const int              s00,
                                           const int              s01,
                                           const int              s02,
                                           const int              s03,
                                           const int              s10,
                                           const int              s11,
                                           const int              s12,
                                           const int              s13,
                                           src1_ptrs... src1s) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = fastdiv(i, prod_012);
    const uint32_t i2 = fastdiv(i - i3 * prod_012.z, prod_01);
    const uint32_t i1 = fastdiv(i - i3 * prod_012.z - i2 * prod_01.z, ne0);
    const uint32_t i0 = i - i3 * prod_012.z - i2 * prod_01.z - i1 * ne0.z;

    if (i0 >= ne0.z || i1 >= ne1.z || i2 >= ne2.z || i3 >= ne3) {
        return;
    }

    const int i11 = fastmodulo(i1, ne11);
    const int i12 = fastmodulo(i2, ne12);
    const int i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    const int i10 = fastmodulo(i0, ne10);

    float result = src0_row ? (float) src0_row[i0*s00] : 0.0f;
    if constexpr (sizeof...(src1_ptrs) > 0) {
        result = (..., (result = bin_op(result, (float)src1s[i_src1 + i10*s10])));
    } else {
        result = bin_op(result, (float)src1[i_src1 + i10*s10]);
    }

    dst_row[i0] = (dst_t) result;
}

template <float (*bin_op)(const float, const float)>
static __global__ void k_bin_bcast_axis_f32(const float * __restrict__ src0,
                                            const float * __restrict__ src1,
                                            float * __restrict__ dst,
                                            const int64_t total,
                                            const int64_t ne0,
                                            const int64_t ne1,
                                            const int64_t ne2,
                                            const int axis) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= total) {
        return;
    }

    int64_t i_src1;
    switch (axis) {
        case 0:
            i_src1 = i % ne0;
            break;
        case 1:
            i_src1 = (i / ne0) % ne1;
            break;
        case 2:
            i_src1 = (i / (ne0*ne1)) % ne2;
            break;
        default:
            i_src1 = i / (ne0*ne1*ne2);
            break;
    }

    dst[i] = bin_op(src0[i], src1[i_src1]);
}

template <float (*bin_op)(const float, const float)>
static __global__ void k_bin_bcast_axis0_f32(const float * __restrict__ src0,
                                             const float * __restrict__ src1,
                                             float * __restrict__ dst,
                                             const int64_t ne0,
                                             const int64_t nrows) {
    const int64_t i0 = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }

    const int64_t row = int64_t(blockIdx.z)*gridDim.y + blockIdx.y;
    if (row >= nrows) {
        return;
    }

    const int64_t offset = row*ne0 + i0;
    dst[offset] = bin_op(src0[offset], src1[i0]);
}

template <float (*bin_op)(const float, const float)>
static __global__ void k_bin_bcast_axis2_f32(const float * __restrict__ src0,
                                             const float * __restrict__ src1,
                                             float * __restrict__ dst,
                                             const int64_t plane,
                                             const int64_t ne2) {
    const int64_t i01 = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i01 >= plane) {
        return;
    }

    const int64_t i2 = blockIdx.y % ne2;
    const int64_t offset = int64_t(blockIdx.y)*plane + i01;
    dst[offset] = bin_op(src0[offset], src1[i2]);
}

template <mmq_q8_1_ds_layout ds_layout, int cols_per_block>
static __global__ void k_add_and_quantize_mmq_q8_1_warp_cols(const float * __restrict__ src0,
                                                             const float * __restrict__ src1,
                                                             float * __restrict__ dst,
                                                             void * __restrict__ vy,
                                                             const int64_t ne00,
                                                             const int64_t ne0,
                                                             const int64_t ne1,
                                                             const int64_t ne2) {
    constexpr int vals_per_mmq_block = 4 * QK8_1;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int64_t i1 = (int64_t) blockIdx.x * cols_per_block + warp;
    if (i1 >= ne1) {
        return;
    }

    const int64_t qblock = blockIdx.y;
    const int64_t i0 = qblock * vals_per_mmq_block + lane * 4;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t row_base = ((i3 * ne2 + i2) * ne1 + i1) * ne00;

    float values[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int64_t col = i0 + k;
        if (col < ne00) {
            const int64_t idx = row_base + col;
            values[k] = src0[idx] + src1[idx];
            dst[idx] = values[k];
        }
    }

    const float4 xi = make_float4(values[0], values[1], values[2], values[3]);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    const char4 q = make_char4(roundf(xi.x * d_inv),
                               roundf(xi.y * d_inv),
                               roundf(xi.z * d_inv),
                               roundf(xi.w * d_inv));

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    const int64_t ib0 = blockIdx.z * ((int64_t) gridDim.y * ne1);
    const int64_t ib = ib0 + qblock * ne1 + i1;
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[lane] = q;

    const int iqs = lane * 4;
    if (iqs % 32 == 0) {
        const float d = 1.0f / d_inv;
        if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
            y[ib].d4[iqs / 32] = d;
        } else {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        }
    }

    GGML_UNUSED(ne0);
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_contiguous(const src0_t * __restrict__ src0,
                                        const src1_t * __restrict__ src1,
                                        dst_t * __restrict__ dst,
                                        const int64_t total) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= total) {
        return;
    }

    dst[i] = (dst_t) bin_op((float) src0[i], (float) src1[i]);
}

template <typename src_t, typename dst_t>
static __global__ void k_rope_pair_fused(const src_t * __restrict__ x_re,
                                         const src_t * __restrict__ x_im,
                                         const float * __restrict__ cos,
                                         const float * __restrict__ sin,
                                         dst_t * __restrict__ dst,
                                         const int64_t half,
                                         const int64_t n_tokens,
                                         const int64_t n_heads_b,
                                         const int64_t x_re_s1,
                                         const int64_t x_re_s2,
                                         const int64_t x_re_s3,
                                         const int64_t x_im_s1,
                                         const int64_t x_im_s2,
                                         const int64_t x_im_s3,
                                         const int64_t cos_s1,
                                         const int64_t cos_s2,
                                         const int64_t sin_s1,
                                         const int64_t sin_s2,
                                         const int64_t dst_s0,
                                         const int64_t dst_s1,
                                         const int64_t dst_s2,
                                         const int64_t dst_s3) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = half * n_tokens * n_heads_b;
    if (i >= total) {
        return;
    }

    const int64_t h = i % half;
    const int64_t t = (i / half) % n_tokens;
    const int64_t b = i / (half * n_tokens);

    const float re = (float) x_re[h * x_re_s1 + t * x_re_s2 + b * x_re_s3];
    const float im = (float) x_im[h * x_im_s1 + t * x_im_s2 + b * x_im_s3];
    const float c  = cos[h * cos_s1 + t * cos_s2];
    const float s  = sin[h * sin_s1 + t * sin_s2];

    const int64_t dst_base = h * dst_s1 + t * dst_s2 + b * dst_s3;
    const float re_c = __fmul_rn(re, c);
    const float im_s = __fmul_rn(im, s);
    const float re_s = __fmul_rn(re, s);
    const float im_c = __fmul_rn(im, c);

    dst[dst_base]          = (dst_t) __fsub_rn(re_c, im_s);
    dst[dst_base + dst_s0] = (dst_t) __fadd_rn(re_s, im_c);
}

static int ggml_cuda_get_bin_bcast_axis_f32(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }
    if (!ggml_are_same_shape(src0, dst) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return -1;
    }
    if (ggml_nelements(src1) == ggml_nelements(dst)) {
        return -1;
    }

    int axis = -1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src1->ne[i] == dst->ne[i]) {
            if (axis >= 0) {
                return -1;
            }
            axis = i;
        } else if (src1->ne[i] != 1) {
            return -1;
        }
    }

    return axis;
}

template <float (*bin_op)(const float, const float)>
static bool ggml_cuda_try_bin_bcast_axis_f32(const ggml_tensor * src0,
                                             const ggml_tensor * src1,
                                             ggml_tensor * dst,
                                             cudaStream_t stream) {
    static const bool disabled =
        std::getenv("GGML_CUDA_DISABLE_BIN_BCAST_AXIS_FAST") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_BIN_BCAST_AXIS_FAST"));
    if (disabled) {
        return false;
    }

    const int axis = ggml_cuda_get_bin_bcast_axis_f32(src0, src1, dst);
    if (axis < 0) {
        return false;
    }

    const int64_t total = ggml_nelements(dst);
    const int block_size = 256;
    const bool disable_axis0 =
        std::getenv("GGML_CUDA_DISABLE_BIN_BCAST_AXIS0_FAST") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_BIN_BCAST_AXIS0_FAST"));
    const int64_t axis0_grid_y = total / dst->ne[0];
    if (axis == 0 && !disable_axis0) {
        const int64_t grid_y = std::min<int64_t>(axis0_grid_y, 65535);
        const int64_t grid_z = (axis0_grid_y + grid_y - 1) / grid_y;
        if (grid_z <= 65535) {
            const dim3 blocks((dst->ne[0] + block_size - 1) / block_size, grid_y, grid_z);
            k_bin_bcast_axis0_f32<bin_op><<<blocks, block_size, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data, dst->ne[0], axis0_grid_y);
            return true;
        }
    }

    const bool disable_axis2 =
        std::getenv("GGML_CUDA_DISABLE_BIN_BCAST_AXIS2_FAST") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_BIN_BCAST_AXIS2_FAST"));
    const int64_t axis2_grid_y = dst->ne[2]*dst->ne[3];
    if (axis == 2 && !disable_axis2 && axis2_grid_y <= 65535) {
        const int64_t plane = dst->ne[0]*dst->ne[1];
        const dim3 blocks((plane + block_size - 1) / block_size, axis2_grid_y, 1);
        k_bin_bcast_axis2_f32<bin_op><<<blocks, block_size, 0, stream>>>(
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data, plane, dst->ne[2]);
        return true;
    }

    const int blocks = (total + block_size - 1) / block_size;
    k_bin_bcast_axis_f32<bin_op><<<blocks, block_size, 0, stream>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        total, dst->ne[0], dst->ne[1], dst->ne[2], axis);
    return true;
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static bool ggml_cuda_try_bin_contiguous(const ggml_tensor * src0,
                                         const ggml_tensor * src1,
                                         ggml_tensor * dst,
                                         const src0_t * src0_dd,
                                         const src1_t * src1_dd,
                                         dst_t * dst_dd,
                                         cudaStream_t stream) {
    static const bool disabled =
        std::getenv("GGML_CUDA_DISABLE_BIN_CONTIGUOUS_FAST") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_BIN_CONTIGUOUS_FAST"));
    if (disabled) {
        return false;
    }
    if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t total = ggml_nelements(dst);
    const int block_size = 256;
    const int blocks = (total + block_size - 1) / block_size;
    k_bin_contiguous<bin_op><<<blocks, block_size, 0, stream>>>(src0_dd, src1_dd, dst_dd, total);
    return true;
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t, size_t... I>
static void launch_bin_bcast_pack(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                  const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
                                  cudaStream_t stream, std::index_sequence<I...>) {
    GGML_TENSOR_BINARY_OP_LOCALS

    int nr0 = ne10 / ne0;
    int nr1 = ne11 / ne1;
    int nr2 = ne12 / ne2;
    int nr3 = ne13 / ne3;

    int nr[4] = { nr0, nr1, nr2, nr3 };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };

    size_t cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t cnb1[] = { nb10, nb11, nb12, nb13 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && !ggml_is_permuted(src0) && !ggml_is_permuted(src1)) {
        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
        //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
        //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
        //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

        size_t nb0 = cnb[0];
        size_t nb1 = cnb[1];
        size_t nb2 = cnb[2];
        size_t nb3 = cnb[3];

        size_t nb00 = cnb0[0];
        size_t nb01 = cnb0[1];
        size_t nb02 = cnb0[2];
        size_t nb03 = cnb0[3];

        size_t nb10 = cnb1[0];
        size_t nb11 = cnb1[1];
        size_t nb12 = cnb1[2];
        size_t nb13 = cnb1[3];

      //size_t s0 = nb0 / sizeof(dst_t);
        size_t s1 = nb1 / sizeof(dst_t);
        size_t s2 = nb2 / sizeof(dst_t);
        size_t s3 = nb3 / sizeof(dst_t);

        size_t s10 = nb10 / sizeof(src1_t);
        size_t s11 = nb11 / sizeof(src1_t);
        size_t s12 = nb12 / sizeof(src1_t);
        size_t s13 = nb13 / sizeof(src1_t);

        size_t s00 = nb00 / sizeof(src0_t);
        size_t s01 = nb01 / sizeof(src0_t);
        size_t s02 = nb02 / sizeof(src0_t);
        size_t s03 = nb03 / sizeof(src0_t);

        GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

        GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

        GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

        const int block_size = 128;

        int64_t hne0 = std::max(ne0 / 2LL, 1LL);

        dim3 block_dims;
        block_dims.x = std::min<unsigned int>(hne0, block_size);
        block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
        block_dims.z = std::min(std::min<unsigned int>(ne2 * ne3, block_size / block_dims.x / block_dims.y), 64U);

        dim3 block_nums((hne0 + block_dims.x - 1) / block_dims.x, (ne1 + block_dims.y - 1) / block_dims.y,
                        (ne2 * ne3 + block_dims.z - 1) / block_dims.z);

        const uint3 ne10 = init_fastdiv_values((uint32_t) cne1[0]);
        const uint3 ne11 = init_fastdiv_values((uint32_t) cne1[1]);
        const uint3 ne12 = init_fastdiv_values((uint32_t) cne1[2]);
        const uint3 ne13 = init_fastdiv_values((uint32_t) cne1[3]);

        if (block_nums.z > 65535 || block_nums.y > 65535) {
            int         block_num  = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            const uint3 prod_012    = init_fastdiv_values((uint32_t) (ne0 * ne1 * ne2));
            const uint3 prod_01     = init_fastdiv_values((uint32_t) (ne0 * ne1));
            const uint3 ne0_fastdiv = init_fastdiv_values((uint32_t) ne0);
            const uint3 ne1_fastdiv = init_fastdiv_values((uint32_t) ne1);
            const uint3 ne2_fastdiv = init_fastdiv_values((uint32_t) ne2);

            if constexpr (sizeof...(I) > 0) {
                k_bin_bcast_unravel<bin_op, src0_t, src1_t, dst_t><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv, ne2_fastdiv, ne3, prod_012, prod_01, ne10, ne11,
                    ne12, ne13,
                  /*s0,*/ s1,  s2,  s3,
                    s00, s01, s02, s03,
                    s10, s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            } else {
                k_bin_bcast_unravel<bin_op, src0_t, src1_t, dst_t>
                    <<<block_num, block_size, 0, stream>>>(src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv,
                                                           ne2_fastdiv, ne3, prod_012, prod_01, ne10, ne11, ne12, ne13,
                                                         /*s0,*/ s1,  s2,  s3,
                                                           s00, s01, s02, s03,
                                                           s10, s11, s12, s13);
            }
        } else {
            const uint3 ne3_fastdiv = init_fastdiv_values((uint32_t) ne3);
            if constexpr (sizeof...(I) > 0) {
                k_bin_bcast<bin_op, src0_t, src1_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv, ne10, ne11, ne12, ne13,
                  /*s0,*/ s1, s2,  s3,
                    s00 ,s01, s02, s03,
                    s10, s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            } else {
                k_bin_bcast<bin_op, src0_t, src1_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv, ne10, ne11, ne12, ne13,
                  /*s0,*/ s1,  s2,  s3,
                    s00, s01, s02, s03,
                    s10, s11, s12, s13);
            }
        }
    }
}

template <typename T>
static __global__ void k_repeat_back(
    const T * __restrict__ src, T * __restrict__ dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {

    const int64_t tid0  = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t tid1  = int64_t(blockIdx.y)*blockDim.y + threadIdx.y;
    const int64_t tid23 = int64_t(blockIdx.z)*blockDim.z + threadIdx.z;
    const int64_t tid2  = tid23 % ne2;
    const int64_t tid3  = tid23 / ne2;

    if (tid0 >= ne0) {
        return;
    }

    T sum = 0;
    for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
        for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
            for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
                for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                    sum += src[i3*s03 + i2*s02 + i1*s01 + i0*s00];
                }
            }
        }
    }
    dst[tid3*ne2*ne1*ne0 + tid2*ne1*ne0 + tid1*ne0 + tid0] = sum;
}

template <float (*bin_op)(const float, const float), int n_fuse = 1>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {
        if (src0 != nullptr && ggml_cuda_try_bin_contiguous<bin_op>(src0, src1, dst, src0_dd, src1_dd, dst_dd, stream)) {
            return;
        }
        launch_bin_bcast_pack<bin_op, src0_t, src1_t, dst_t>(
            src0, src1, dst, src0_dd, src1_dd, dst_dd, stream, std::make_index_sequence<n_fuse>{});
    }
};

template <typename T>
static void repeat_back_cuda(
    const T * src, T * dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums((ne0 + WARP_SIZE - 1) / WARP_SIZE, ne1, ne2*ne3);
    k_repeat_back<T><<<block_nums, block_dims, 0, stream>>>
        (src, dst, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3);
}

template<class op>
static void ggml_cuda_op_bin_bcast(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const void * src1_dd, void * dst_dd, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_BF16) {
            op()(src0, src1, dst, (const float *)src0_dd, (const nv_bfloat16 *)src1_dd, (float *)dst_dd, stream);
        } else if (src1->type == GGML_TYPE_F16) {
            op()(src0, src1, dst, (const float *)src0_dd, (const half *)src1_dd, (float *)dst_dd, stream);
        } else {
            op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const half *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()(src0, src1, dst, (const nv_bfloat16 *) src0_dd, (const nv_bfloat16 *)src1_dd, (nv_bfloat16 *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()(src0, src1, dst, (const nv_bfloat16 *) src0_dd, (const float *)src1_dd, (nv_bfloat16 *) dst_dd, stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat, 0>>(dst, dst->src[0], dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_op_add_with_mmq_q8_1_prequant(ctx, dst)) {
        return;
    }
    if (ggml_cuda_try_bin_bcast_axis_f32<op_add>(dst->src[0], dst->src[1], dst, ctx.stream())) {
        return;
    }
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

bool ggml_cuda_op_add_with_mmq_q8_1_prequant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ctx.mmq_prequant_target_tensor != dst || ctx.mmq_prequant_cache_key_tensor == nullptr) {
        return false;
    }

    const auto clear_target = [&ctx] {
        ctx.mmq_prequant_target_tensor = nullptr;
        ctx.mmq_prequant_cache_key_tensor = nullptr;
        ctx.mmq_prequant_target_consumer_type = GGML_TYPE_COUNT;
        ctx.mmq_prequant_target_ne0_padded = 0;
        ctx.mmq_prequant_target_ne1 = 0;
        ctx.mmq_prequant_target_ne2 = 0;
        ctx.mmq_prequant_target_ne3 = 0;
        ctx.mmq_prequant_target_q8_only = false;
    };

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0 == nullptr || src1 == nullptr ||
        dst->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
        (ctx.mmq_prequant_target_consumer_type != GGML_TYPE_Q4_1 &&
         ctx.mmq_prequant_target_consumer_type != GGML_TYPE_Q8_0) ||
        !ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, dst) ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
        ctx.mmq_prequant_target_ne0_padded <= 0 || ctx.mmq_prequant_target_ne1 <= 0 ||
        ctx.mmq_prequant_target_ne2 <= 0 || ctx.mmq_prequant_target_ne3 <= 0) {
        clear_target();
        return false;
    }

    const int64_t ne00 = ctx.mmq_prequant_cache_key_tensor->ne[0];
    const int64_t ne0_padded = ctx.mmq_prequant_target_ne0_padded;
    const int64_t ne1 = ctx.mmq_prequant_target_ne1;
    const int64_t ne2 = ctx.mmq_prequant_target_ne2;
    const int64_t ne3 = ctx.mmq_prequant_target_ne3;
    if (ne00 != dst->ne[0] || ne0_padded % (4 * QK8_1) != 0 ||
        ne1 * ne2 * ne3 * ne00 != ggml_nelements(dst)) {
        clear_target();
        return false;
    }

    ggml_backend_cuda_context::mmq_prequant_cache_entry entry;
    const size_t nbytes_q8_1 =
        ne3 * ne2 * ne1 * ne0_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc) * sizeof(block_q8_1_mmq);
    entry.consumer_type = ctx.mmq_prequant_target_consumer_type;
    entry.ne0_padded = ne0_padded;
    entry.ne1 = ne1;
    entry.ne2 = ne2;
    entry.ne3 = ne3;
    entry.storage = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes_q8_1);
    entry.data = entry.storage->get();

    const int cols_per_block = [] {
        const char * env = getenv("GGML_CUDA_MMQ_ADD_PREQUANT_WARP_COLS");
        return env == nullptr ? 4 : atoi(env);
    }();
    const int64_t qblocks = ne0_padded / (4 * QK8_1);
#define GGML_CUDA_LAUNCH_ADD_PREQUANT(DS_LAYOUT)                                                                                       \
    do {                                                                                                                               \
        if (cols_per_block == 16) {                                                                                                    \
            const dim3 num_blocks((ne1 + 15) / 16, qblocks, ne2 * ne3);                                                                \
            const dim3 block_size(WARP_SIZE, 16, 1);                                                                                   \
            k_add_and_quantize_mmq_q8_1_warp_cols<DS_LAYOUT, 16><<<num_blocks, block_size, 0, ctx.stream()>>>(                         \
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data, entry.data, ne00, ne0_padded, ne1, ne2);  \
        } else if (cols_per_block == 8) {                                                                                              \
            const dim3 num_blocks((ne1 + 7) / 8, qblocks, ne2 * ne3);                                                                  \
            const dim3 block_size(WARP_SIZE, 8, 1);                                                                                    \
            k_add_and_quantize_mmq_q8_1_warp_cols<DS_LAYOUT, 8><<<num_blocks, block_size, 0, ctx.stream()>>>(                          \
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data, entry.data, ne00, ne0_padded, ne1, ne2);  \
        } else {                                                                                                                       \
            const dim3 num_blocks((ne1 + 3) / 4, qblocks, ne2 * ne3);                                                                  \
            const dim3 block_size(WARP_SIZE, 4, 1);                                                                                    \
            k_add_and_quantize_mmq_q8_1_warp_cols<DS_LAYOUT, 4><<<num_blocks, block_size, 0, ctx.stream()>>>(                          \
                (const float *) src0->data, (const float *) src1->data, (float *) dst->data, entry.data, ne00, ne0_padded, ne1, ne2);  \
        }                                                                                                                             \
    } while (0)
    if (mmq_get_q8_1_ds_layout(ctx.mmq_prequant_target_consumer_type) == MMQ_Q8_1_DS_LAYOUT_D4) {
        GGML_CUDA_LAUNCH_ADD_PREQUANT(MMQ_Q8_1_DS_LAYOUT_D4);
    } else {
        GGML_CUDA_LAUNCH_ADD_PREQUANT(MMQ_Q8_1_DS_LAYOUT_DS4);
    }
#undef GGML_CUDA_LAUNCH_ADD_PREQUANT
    CUDA_CHECK(cudaGetLastError());

    ctx.mmq_prequant_cache.emplace_back(ctx.mmq_prequant_cache_key_tensor, std::move(entry));
    clear_target();
    return true;
}

template <float (*op)(const float, const float)>
static void ggml_cuda_op_add_unary_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * unary) {
    const ggml_tensor * src0 = add->src[0];
    const ggml_tensor * src1 = add->src[1];

    ggml_tensor fused = *add;
    fused.data = unary->data;

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
        add->type == GGML_TYPE_F32 && unary->type == GGML_TYPE_F32) {
        if (ggml_cuda_try_bin_bcast_axis_f32<op>(src0, src1, &fused, ctx.stream())) {
            return;
        }

        launch_bin_bcast_pack<op, float, float, float>(
            src0, src1, &fused, (const float *) src0->data, (const float *) src1->data, (float *) unary->data,
            ctx.stream(), std::make_index_sequence<1>{});
        return;
    }

    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 &&
        add->type == GGML_TYPE_BF16 && unary->type == GGML_TYPE_BF16) {
        launch_bin_bcast_pack<op, nv_bfloat16, float, nv_bfloat16>(
            src0, src1, &fused, (const nv_bfloat16 *) src0->data, (const float *) src1->data, (nv_bfloat16 *) unary->data,
            ctx.stream(), std::make_index_sequence<1>{});
        return;
    }

    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 &&
        add->type == GGML_TYPE_BF16 && unary->type == GGML_TYPE_BF16) {
        launch_bin_bcast_pack<op, nv_bfloat16, nv_bfloat16, nv_bfloat16>(
            src0, src1, &fused, (const nv_bfloat16 *) src0->data, (const nv_bfloat16 *) src1->data, (nv_bfloat16 *) unary->data,
            ctx.stream(), std::make_index_sequence<1>{});
        return;
    }

    fprintf(stderr,
            "%s: unsupported types: add=%s unary=%s src0=%s src1=%s\n",
            __func__,
            ggml_type_name(add->type),
            ggml_type_name(unary->type),
            ggml_type_name(src0->type),
            ggml_type_name(src1->type));
    GGML_ABORT("unsupported add+unary fusion types");
}

void ggml_cuda_op_add_unary(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * unary) {
    switch (ggml_get_unary_op(unary)) {
        case GGML_UNARY_OP_GELU:
            ggml_cuda_op_add_unary_impl<op_add_gelu>(ctx, add, unary);
            break;
        case GGML_UNARY_OP_GELU_ERF:
            ggml_cuda_op_add_unary_impl<op_add_gelu_erf>(ctx, add, unary);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            ggml_cuda_op_add_unary_impl<op_add_gelu_quick>(ctx, add, unary);
            break;
        default:
            GGML_ABORT("unsupported add+unary fusion");
    }
}

void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_sub>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

template <float (*op)(const float, const float), int n_fuse>
static void ggml_cuda_op_fused_binbcast_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, float, float, float>(src0, src1, dst,
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, half, half>(src0, src1, dst,
            (const half *) src0->data, (const half *) src1->data, (half *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, float, half>(src0, src1, dst,
            (const half *) src0->data, (const float *) src1->data, (half *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, half, float, float>(src0, src1, dst,
            (const half *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else {
        fprintf(stderr,
                "%s: unsupported types for fusion: dst: %s, src0: %s, src1: %s\n",
                __func__, ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}


void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse) {
    GGML_ASSERT(2 <= n_fuse && n_fuse <= 8);

    switch (n_fuse) {
        case 2:
            ggml_cuda_op_fused_binbcast_impl<op_add, 2>(ctx, dst);
            break;
        case 3:
            ggml_cuda_op_fused_binbcast_impl<op_add, 3>(ctx, dst);
            break;
        case 4:
            ggml_cuda_op_fused_binbcast_impl<op_add, 4>(ctx, dst);
            break;
        case 5:
            ggml_cuda_op_fused_binbcast_impl<op_add, 5>(ctx, dst);
            break;
        case 6:
            ggml_cuda_op_fused_binbcast_impl<op_add, 6>(ctx, dst);
            break;
        case 7:
            ggml_cuda_op_fused_binbcast_impl<op_add, 7>(ctx, dst);
            break;
        case 8:
            ggml_cuda_op_fused_binbcast_impl<op_add, 8>(ctx, dst);
            break;
        default:
            GGML_ASSERT(false && "Unsupported n_fuse value");
    }
}

void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse) {
    GGML_ASSERT(2 <= n_fuse && n_fuse <= 8);

    switch (n_fuse) {
        case 2:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 2>(ctx, dst);
            break;
        case 3:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 3>(ctx, dst);
            break;
        case 4:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 4>(ctx, dst);
            break;
        case 5:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 5>(ctx, dst);
            break;
        case 6:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 6>(ctx, dst);
            break;
        case 7:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 7>(ctx, dst);
            break;
        case 8:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 8>(ctx, dst);
            break;
        default:
            GGML_ASSERT(false && "Unsupported n_fuse value");
    }
}

void ggml_cuda_op_rope_pair_fused(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor * x_re,
                                  const ggml_tensor * x_im,
                                  const ggml_tensor * cos,
                                  const ggml_tensor * sin,
                                  ggml_tensor * dst) {
    GGML_ASSERT(x_re->type == GGML_TYPE_F32 || x_re->type == GGML_TYPE_BF16);
    GGML_ASSERT(x_im->type == x_re->type);
    GGML_ASSERT(cos->type == GGML_TYPE_F32);
    GGML_ASSERT(sin->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == x_re->type);

    const int64_t half = x_re->ne[1];
    const int64_t n_tokens = x_re->ne[2];
    const int64_t n_heads_b = x_re->ne[3];
    const int64_t total = half * n_tokens * n_heads_b;
    const int block_size = 256;
    const int blocks = (total + block_size - 1) / block_size;

    const auto stride_type = [](const ggml_tensor * t, int dim) {
        GGML_ASSERT(t->nb[dim] % ggml_type_size(t->type) == 0);
        return int64_t(t->nb[dim] / ggml_type_size(t->type));
    };

    if (x_re->type == GGML_TYPE_BF16) {
        k_rope_pair_fused<<<blocks, block_size, 0, ctx.stream()>>>(
            (const nv_bfloat16 *) x_re->data, (const nv_bfloat16 *) x_im->data,
            (const float *) cos->data, (const float *) sin->data, (nv_bfloat16 *) dst->data,
            half, n_tokens, n_heads_b,
            stride_type(x_re, 1), stride_type(x_re, 2), stride_type(x_re, 3),
            stride_type(x_im, 1), stride_type(x_im, 2), stride_type(x_im, 3),
            stride_type(cos, 1), stride_type(cos, 2),
            stride_type(sin, 1), stride_type(sin, 2),
            stride_type(dst, 0), stride_type(dst, 1), stride_type(dst, 2), stride_type(dst, 3));
    } else {
        k_rope_pair_fused<<<blocks, block_size, 0, ctx.stream()>>>(
            (const float *) x_re->data, (const float *) x_im->data,
            (const float *) cos->data, (const float *) sin->data, (float *) dst->data,
            half, n_tokens, n_heads_b,
            stride_type(x_re, 1), stride_type(x_re, 2), stride_type(x_re, 3),
            stride_type(x_im, 1), stride_type(x_im, 2), stride_type(x_im, 3),
            stride_type(cos, 1), stride_type(cos, 2),
            stride_type(sin, 1), stride_type(sin, 2),
            stride_type(dst, 0), stride_type(dst, 1), stride_type(dst, 2), stride_type(dst, 3));
    }
}

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(dst, src0));

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(ne2*ne3 <= (1 << 15));

    const size_t ts = ggml_type_size(src0->type);
    const size_t s00 = nb00 / ts;
    const size_t s01 = nb01 / ts;
    const size_t s02 = nb02 / ts;
    const size_t s03 = nb03 / ts;

    switch (dst->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            float       * dst_d  = (float       *) dst->data;
            repeat_back_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
        } break;
        default: {
            GGML_ASSERT(false);
        } break;
    }
}
