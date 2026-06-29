#include "cpy-utils.cuh"
#include "cpy.cuh"
#include "dequantize.cuh"

#include <cstdint>
#include <cstdio>
#include <type_traits>
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
#include "ggml-musa/mudnn.cuh"
#endif  // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY

typedef void (*cpy_kernel_t)(const char* cx, char* cdst);

const int CUDA_CPY_TILE_DIM_2D = 32;  // 2D tile dimension for transposed blocks
const int CUDA_CPY_BLOCK_NM = 8;      // block size of 3rd dimension if available
const int CUDA_CPY_BLOCK_ROWS = 8;    // block dimension for marching through rows

template <cpy_kernel_t cpy_1>
static __global__ void cpy_scalar(const char* cx,
                                  char* cdst,
                                  const int64_t ne,
                                  const int64_t ne00,
                                  const int64_t ne01,
                                  const int64_t ne02,
                                  const int64_t nb00,
                                  const int64_t nb01,
                                  const int64_t nb02,
                                  const int64_t nb03,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t nb10,
                                  const int64_t nb11,
                                  const int64_t nb12,
                                  const int64_t nb13) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened
    // tensor then combine those indices with the corresponding byte offsets to get the total
    // offsets
    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int64_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int64_t i13 = i / (ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int64_t dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_1, typename dst_t>
static __global__ void cpy_scalar_to_contiguous(const char* cx,
                                                char* cdst,
                                                const int64_t ne,
                                                const int64_t ne00,
                                                const int64_t ne01,
                                                const int64_t ne02,
                                                const int64_t nb00,
                                                const int64_t nb01,
                                                const int64_t nb02,
                                                const int64_t nb03) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int64_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    cpy_1(cx + x_offset, cdst + i * sizeof(dst_t));
}

template <cpy_kernel_t cpy_1, typename dst_t>
static __global__ void cpy_scalar_to_contiguous_rows(const char* cx,
                                                     char* cdst,
                                                     const int64_t ne00,
                                                     const int64_t ne01,
                                                     const int64_t ne02,
                                                     const int64_t nrows,
                                                     const int64_t nb00,
                                                     const int64_t nb01,
                                                     const int64_t nb02,
                                                     const int64_t nb03) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    const int64_t i03 = row / (ne01 * ne02);
    const int64_t i02 = (row - i03 * ne01 * ne02) / ne01;
    const int64_t i01 = row - i03 * ne01 * ne02 - i02 * ne01;
    const int64_t row_src_offset = i01 * nb01 + i02 * nb02 + i03 * nb03;
    const int64_t row_dst_offset = row * ne00 * sizeof(dst_t);

    for (int64_t i00 = threadIdx.x; i00 < ne00; i00 += blockDim.x) {
        cpy_1(cx + row_src_offset + i00 * nb00, cdst + row_dst_offset + i00 * sizeof(dst_t));
    }
}

template <typename T>
static __global__ void cpy_pair_to_contiguous_rows(const char* __restrict__ src0,
                                                   const char* __restrict__ src1,
                                                   char* __restrict__ dst0,
                                                   char* __restrict__ dst1,
                                                   const int64_t ne00,
                                                   const int64_t ne01,
                                                   const int64_t ne02,
                                                   const int64_t nrows,
                                                   const int64_t nb00,
                                                   const int64_t nb01,
                                                   const int64_t nb02,
                                                   const int64_t nb03) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    const int64_t i03 = row / (ne01 * ne02);
    const int64_t i02 = (row - i03 * ne01 * ne02) / ne01;
    const int64_t i01 = row - i03 * ne01 * ne02 - i02 * ne01;
    const int64_t row_src_offset = i01 * nb01 + i02 * nb02 + i03 * nb03;
    const int64_t row_dst_offset = row * ne00;

    T* dst0_t = (T*) dst0;
    T* dst1_t = (T*) dst1;
    for (int64_t i00 = threadIdx.x; i00 < ne00; i00 += blockDim.x) {
        dst0_t[row_dst_offset + i00] = *(const T*) (src0 + row_src_offset + i00 * nb00);
        dst1_t[row_dst_offset + i00] = *(const T*) (src1 + row_src_offset + i00 * nb00);
    }
}

static __global__ void cpy_f32_to_contiguous_rows_vec4(const char* cx,
                                                       char* cdst,
                                                       const int64_t ne00,
                                                       const int64_t ne01,
                                                       const int64_t ne02,
                                                       const int64_t nrows,
                                                       const int64_t nb01,
                                                       const int64_t nb02,
                                                       const int64_t nb03) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    const int64_t i03 = row / (ne01 * ne02);
    const int64_t i02 = (row - i03 * ne01 * ne02) / ne01;
    const int64_t i01 = row - i03 * ne01 * ne02 - i02 * ne01;
    const int64_t row_src_offset = i01 * nb01 + i02 * nb02 + i03 * nb03;

    const float4* src = (const float4*) (cx + row_src_offset);
    float4* dst = (float4*) (cdst + row * ne00 * sizeof(float));
    const int64_t ne00_vec4 = ne00 / 4;

    for (int64_t i = threadIdx.x; i < ne00_vec4; i += blockDim.x) {
        dst[i] = src[i];
    }
}

static __global__ void cpy_f32_to_contiguous_rows_vec4_group4(const char* cx,
                                                              char* cdst,
                                                              const int64_t ne00,
                                                              const int64_t ne01,
                                                              const int64_t ne02,
                                                              const int64_t nrows,
                                                              const int64_t nb01,
                                                              const int64_t nb02,
                                                              const int64_t nb03) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }

    const int64_t i03 = row / (ne01 * ne02);
    const int64_t i02 = (row - i03 * ne01 * ne02) / ne01;
    const int64_t i01 = row - i03 * ne01 * ne02 - i02 * ne01;
    const int64_t row_src_offset = i01 * nb01 + i02 * nb02 + i03 * nb03;

    const float4* src = (const float4*) (cx + row_src_offset);
    float4* dst = (float4*) (cdst + row * ne00 * sizeof(float));
    const int64_t row_vec4 = ne00 / 4;

    for (int64_t i = threadIdx.x; i < row_vec4; i += blockDim.x) {
        dst[i] = src[i];
    }
}

static __global__ void cpy_pair_f32_to_contiguous_rows_vec4_group4(const char* __restrict__ src0,
                                                                   const char* __restrict__ src1,
                                                                   char* __restrict__ dst0,
                                                                   char* __restrict__ dst1,
                                                                   const int64_t ne00,
                                                                   const int64_t ne01,
                                                                   const int64_t ne02,
                                                                   const int64_t nrows,
                                                                   const int64_t nb01,
                                                                   const int64_t nb02,
                                                                   const int64_t nb03) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }

    const int64_t i03 = row / (ne01 * ne02);
    const int64_t i02 = (row - i03 * ne01 * ne02) / ne01;
    const int64_t i01 = row - i03 * ne01 * ne02 - i02 * ne01;
    const int64_t row_src_offset = i01 * nb01 + i02 * nb02 + i03 * nb03;

    const float4* src0_row = (const float4*) (src0 + row_src_offset);
    const float4* src1_row = (const float4*) (src1 + row_src_offset);
    float4* dst0_row = (float4*) (dst0 + row * ne00 * sizeof(float));
    float4* dst1_row = (float4*) (dst1 + row * ne00 * sizeof(float));
    const int64_t row_vec4 = ne00 / 4;

    for (int64_t i = threadIdx.x; i < row_vec4; i += blockDim.x) {
        dst0_row[i] = src0_row[i];
        dst1_row[i] = src1_row[i];
    }
}

static __global__ void cpy_f32_to_contiguous_rows_bias_axis0(const char* __restrict__ cx,
                                                             const float* __restrict__ bias,
                                                             float* __restrict__ dst,
                                                             const int64_t ne00,
                                                             const int64_t ne01,
                                                             const int64_t ne02,
                                                             const int64_t nrows,
                                                             const int64_t nb00,
                                                             const int64_t nb01,
                                                             const int64_t nb02,
                                                             const int64_t nb03) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    const int64_t i03 = row / (ne01 * ne02);
    const int64_t i02 = (row - i03 * ne01 * ne02) / ne01;
    const int64_t i01 = row - i03 * ne01 * ne02 - i02 * ne01;
    const int64_t row_src_offset = i01 * nb01 + i02 * nb02 + i03 * nb03;
    const int64_t row_dst_offset = row * ne00;

    for (int64_t i00 = threadIdx.x; i00 < ne00; i00 += blockDim.x) {
        const float x = *(const float*) (cx + row_src_offset + i00 * nb00);
        dst[row_dst_offset + i00] = x + bias[i00];
    }
}

static __global__ void add_bias_axis2_permute_2013_cont(const float* __restrict__ src,
                                                        const float* __restrict__ bias,
                                                        float* __restrict__ dst,
                                                        const int64_t width,
                                                        const int64_t height,
                                                        const int64_t channels,
                                                        const int64_t batches) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t ne = width * height * channels * batches;
    if (i >= ne) {
        return;
    }

    const int64_t c = i % channels;
    const int64_t x = (i / channels) % width;
    const int64_t y = (i / (channels * width)) % height;
    const int64_t b = i / (channels * width * height);
    const int64_t src_idx = x + width * (y + height * (c + channels * b));
    dst[i] = src[src_idx] + bias[c];
}

template <typename T>
static __global__ void cpy_scalar_transpose(const char* cx,
                                            char* cdst,
                                            const int64_t ne,
                                            const int64_t ne00,
                                            const int64_t ne01,
                                            const int64_t ne02,
                                            const int64_t nb00,
                                            const int64_t nb01,
                                            const int64_t nb02,
                                            const int64_t nb03,
                                            const int64_t ne10,
                                            const int64_t ne11,
                                            const int64_t ne12,
                                            const int64_t nb10,
                                            const int64_t nb11,
                                            const int64_t nb12,
                                            const int64_t nb13) {
    const T* src = reinterpret_cast<const T*>(cx);
    T* dst = reinterpret_cast<T*>(cdst);

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    const int x = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int y = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    const int tx = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x;  // transpose block offset
    const int ty = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

    __shared__ float tile[2][CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D + 1];
    int cur_tile_buf = 0;

#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM; ++i) {
        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM + i;
        if (imat >= nmat)
            break;

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if (x < ne01 && y + j < ne00) {
                const int row = threadIdx.y + j;
                const int col = threadIdx.x * sizeof(float) / sizeof(T);
                T* tile2 = reinterpret_cast<T*>(tile[cur_tile_buf][row]);
                tile2[col] = src[imat * n + (y + j) * ne01 + x];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if (ty + j < ne01 && tx < ne00) {
                const int col = (threadIdx.y + j) * sizeof(float) / sizeof(T);
                const T* tile2 = reinterpret_cast<const T*>(tile[cur_tile_buf][threadIdx.x]);
                dst[imat * n + (ty + j) * ne00 + tx] = tile2[col];
            }
        }

        cur_tile_buf = (cur_tile_buf + 1) % 2;
    }

    GGML_UNUSED_VARS(ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static __device__ void cpy_blck_q8_0_f32(const char* cxi, char* cdsti) {
    float* cdstf = (float*) (cdsti);

#pragma unroll
    for (int j = 0; j < QK8_0; j += 2) {
        float2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + 1) = dq.y;
    }
}

template <dequantize_kernel_t dequant, int qk>
static __device__ void cpy_blck_q_f32(const char* cxi, char* cdsti) {
    float* cdstf = (float*) (cdsti);

#pragma unroll
    for (int j = 0; j < qk / 2; j++) {
        float2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + qk / 2) = dq.y;
    }
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char* cx,
                                 char* cdst,
                                 const int64_t ne,
                                 const int64_t ne00,
                                 const int64_t ne01,
                                 const int64_t ne02,
                                 const int64_t nb00,
                                 const int64_t nb01,
                                 const int64_t nb02,
                                 const int64_t nb03,
                                 const int64_t ne10,
                                 const int64_t ne11,
                                 const int64_t ne12,
                                 const int64_t nb10,
                                 const int64_t nb11,
                                 const int64_t nb12,
                                 const int64_t nb13) {
    const int64_t i = ((int64_t) blockDim.x * blockIdx.x + threadIdx.x) * qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int64_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int64_t i13 = i / (ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int64_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_q_f32(const char* cx,
                                 char* cdst,
                                 const int64_t ne,
                                 const int64_t ne00,
                                 const int64_t ne01,
                                 const int64_t ne02,
                                 const int64_t nb00,
                                 const int64_t nb01,
                                 const int64_t nb02,
                                 const int64_t nb03,
                                 const int64_t ne10,
                                 const int64_t ne11,
                                 const int64_t ne12,
                                 const int64_t nb10,
                                 const int64_t nb11,
                                 const int64_t nb12,
                                 const int64_t nb13) {
    const int64_t i = ((int64_t) blockDim.x * blockIdx.x + threadIdx.x) * qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int64_t x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int64_t i13 = i / (ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int64_t dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <typename src_t, typename dst_t>
static __global__ void cpy_scalar_contiguous(const char* cx, char* cdst, const int64_t ne) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const src_t* x = (const src_t*) cx;
    dst_t* dst = (dst_t*) cdst;

    dst[i] = ggml_cuda_cast<dst_t>(x[i]);
}

template <typename dst2_t>
static __global__ void cpy_f32_to_lowp_contiguous_vec4(const float4* cx,
                                                       dst2_t* cdst,
                                                       const int64_t ne4) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne4) {
        return;
    }

    const float4 x = cx[i];
    cdst[2 * i + 0] = ggml_cuda_cast<dst2_t>(make_float2(x.x, x.y));
    cdst[2 * i + 1] = ggml_cuda_cast<dst2_t>(make_float2(x.z, x.w));
}

template <typename dst2_t>
static bool cpy_f32_to_lowp_contiguous_vec4_enabled(const char* cx,
                                                    const char* cdst,
                                                    const int64_t ne) {
    return getenv("GGML_CUDA_DISABLE_CPY_F32_LOWP_VEC4") == nullptr && ne % 4 == 0 &&
           reinterpret_cast<uintptr_t>(cx) % alignof(float4) == 0 &&
           reinterpret_cast<uintptr_t>(cdst) % alignof(dst2_t) == 0;
}

template <typename dst2_t>
static void ggml_cpy_f32_to_lowp_contiguous_vec4_cuda(const char* cx,
                                                      char* cdst,
                                                      const int64_t ne,
                                                      cudaStream_t stream) {
    const int64_t ne4 = ne / 4;
    const int64_t num_blocks = (ne4 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_to_lowp_contiguous_vec4<dst2_t>
        <<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>((const float4*) cx, (dst2_t*) cdst, ne4);
}

template <typename src_t, typename dst_t>
static void ggml_cpy_scalar_contiguous_cuda(const char* cx,
                                            char* cdst,
                                            const int64_t ne,
                                            cudaStream_t stream) {
    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_scalar_contiguous<src_t, dst_t>
        <<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(cx, cdst, ne);
}

template <typename src_t, typename dst_t, bool transposed = false, bool dst_contiguous = false>
static void ggml_cpy_scalar_cuda(const char* cx,
                                 char* cdst,
                                 const int64_t ne,
                                 const int64_t ne00,
                                 const int64_t ne01,
                                 const int64_t ne02,
                                 const int64_t nb00,
                                 const int64_t nb01,
                                 const int64_t nb02,
                                 const int64_t nb03,
                                 const int64_t ne10,
                                 const int64_t ne11,
                                 const int64_t ne12,
                                 const int64_t nb10,
                                 const int64_t nb11,
                                 const int64_t nb12,
                                 const int64_t nb13,
                                 cudaStream_t stream) {
    if (transposed) {
        GGML_ASSERT(ne == ne00 * ne01 * ne02);  // ne[3] is 1 assumed
        int64_t ne00n, ne01n, ne02n;
        if (nb00 <= nb02) {  // most likely safe to handle nb00 = nb02 case here
            ne00n = ne00;
            ne01n = ne01;
            ne02n = ne02;
        } else {
            ne00n = ne00;
            ne01n = ne01 * ne02;
            ne02n = 1;
        }

        int64_t grid_x = (ne01n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_y = (ne00n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_z = (ne / (ne01n * ne00n) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM;
        GGML_ASSERT(grid_x < UINT_MAX);
        GGML_ASSERT(grid_y < USHRT_MAX);
        GGML_ASSERT(grid_z < USHRT_MAX);
        dim3 dimGrid(grid_x, grid_y, grid_z);
        dim3 dimBlock(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
        cpy_scalar_transpose<dst_t><<<dimGrid, dimBlock, 0, stream>>>(cx,
                                                                      cdst,
                                                                      ne,
                                                                      ne00n,
                                                                      ne01n,
                                                                      ne02n,
                                                                      nb00,
                                                                      nb01,
                                                                      nb02,
                                                                      nb03,
                                                                      ne10,
                                                                      ne11,
                                                                      ne12,
                                                                      nb10,
                                                                      nb11,
                                                                      nb12,
                                                                      nb13);
    } else if (dst_contiguous) {
        const int64_t nrows = ne / ne00;
        if (ne00 >= 32 && nrows < UINT_MAX &&
            getenv("GGML_CUDA_DISABLE_CPY_TO_CONTIGUOUS_ROW_FASTPATH") == nullptr) {
            bool use_vec4 = false;
            if constexpr (std::is_same_v<src_t, float> && std::is_same_v<dst_t, float>) {
                use_vec4 = getenv("GGML_CUDA_DISABLE_CPY_TO_CONTIGUOUS_ROW_VEC4") == nullptr &&
                           nb00 == sizeof(float) && ne00 % 4 == 0 &&
                           reinterpret_cast<uintptr_t>(cx) % alignof(float4) == 0 &&
                           reinterpret_cast<uintptr_t>(cdst) % alignof(float4) == 0 &&
                           nb01 % alignof(float4) == 0 && nb02 % alignof(float4) == 0 &&
                           nb03 % alignof(float4) == 0;
            }
            if (use_vec4) {
                const bool use_row_group =
                    getenv("GGML_CUDA_DISABLE_CPY_VEC4_ROW_GROUP") == nullptr &&
                    (ne00 == 256 || ne00 == 1024 ||
                     (ne00 == 64 && getenv("GGML_CUDA_DISABLE_CPY_ROWS64_VEC4_ROW4") == nullptr)) &&
                    nrows >= 4 && reinterpret_cast<uintptr_t>(cx) % alignof(float4) == 0 &&
                    reinterpret_cast<uintptr_t>(cdst) % alignof(float4) == 0;
                if (use_row_group) {
                    constexpr int rows_per_block = 4;
                    const int64_t num_blocks = (nrows + rows_per_block - 1) / rows_per_block;
                    const int64_t row_vec4 = ne00 / 4;
                    const int threads_per_row = row_vec4 < 64 ? int(row_vec4) : 64;
                    GGML_ASSERT(num_blocks < UINT_MAX);
                    cpy_f32_to_contiguous_rows_vec4_group4<<<num_blocks,
                                                             dim3(threads_per_row, rows_per_block),
                                                             0,
                                                             stream>>>(
                        cx, cdst, ne00, ne01, ne02, nrows, nb01, nb02, nb03);
                } else {
                    cpy_f32_to_contiguous_rows_vec4<<<nrows, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
                        cx, cdst, ne00, ne01, ne02, nrows, nb01, nb02, nb03);
                }
            } else {
                cpy_scalar_to_contiguous_rows<cpy_1_scalar<src_t, dst_t>, dst_t>
                    <<<nrows, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
                        cx, cdst, ne00, ne01, ne02, nrows, nb00, nb01, nb02, nb03);
            }
        } else {
            const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
            GGML_ASSERT(num_blocks < UINT_MAX);
            cpy_scalar_to_contiguous<cpy_1_scalar<src_t, dst_t>, dst_t>
                <<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
                    cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03);
        }
    } else {
        const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
        GGML_ASSERT(num_blocks < UINT_MAX);
        cpy_scalar<cpy_1_scalar<src_t, dst_t>>
            <<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(cx,
                                                             cdst,
                                                             ne,
                                                             ne00,
                                                             ne01,
                                                             ne02,
                                                             nb00,
                                                             nb01,
                                                             nb02,
                                                             nb03,
                                                             ne10,
                                                             ne11,
                                                             ne12,
                                                             nb10,
                                                             nb11,
                                                             nb12,
                                                             nb13);
    }
}

static void ggml_cpy_f32_q8_0_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int64_t num_blocks = ne / QK8_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, 1, 0, stream>>>(cx,
                                                                      cdst,
                                                                      ne,
                                                                      ne00,
                                                                      ne01,
                                                                      ne02,
                                                                      nb00,
                                                                      nb01,
                                                                      nb02,
                                                                      nb03,
                                                                      ne10,
                                                                      ne11,
                                                                      ne12,
                                                                      nb10,
                                                                      nb11,
                                                                      nb12,
                                                                      nb13);
}

static void ggml_cpy_q8_0_f32_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q8_0_f32, QK8_0><<<num_blocks, 1, 0, stream>>>(cx,
                                                                      cdst,
                                                                      ne,
                                                                      ne00,
                                                                      ne01,
                                                                      ne02,
                                                                      nb00,
                                                                      nb01,
                                                                      nb02,
                                                                      nb03,
                                                                      ne10,
                                                                      ne11,
                                                                      ne12,
                                                                      nb10,
                                                                      nb11,
                                                                      nb12,
                                                                      nb13);
}

static void ggml_cpy_f32_q4_0_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int64_t num_blocks = ne / QK4_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, 1, 0, stream>>>(cx,
                                                                      cdst,
                                                                      ne,
                                                                      ne00,
                                                                      ne01,
                                                                      ne02,
                                                                      nb00,
                                                                      nb01,
                                                                      nb02,
                                                                      nb03,
                                                                      ne10,
                                                                      ne11,
                                                                      ne12,
                                                                      nb10,
                                                                      nb11,
                                                                      nb12,
                                                                      nb13);
}

static void ggml_cpy_q4_0_f32_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0><<<num_blocks, 1, 0, stream>>>(cx,
                                                                                           cdst,
                                                                                           ne,
                                                                                           ne00,
                                                                                           ne01,
                                                                                           ne02,
                                                                                           nb00,
                                                                                           nb01,
                                                                                           nb02,
                                                                                           nb03,
                                                                                           ne10,
                                                                                           ne11,
                                                                                           ne12,
                                                                                           nb10,
                                                                                           nb11,
                                                                                           nb12,
                                                                                           nb13);
}

static void ggml_cpy_f32_q4_1_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int64_t num_blocks = ne / QK4_1;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, 1, 0, stream>>>(cx,
                                                                      cdst,
                                                                      ne,
                                                                      ne00,
                                                                      ne01,
                                                                      ne02,
                                                                      nb00,
                                                                      nb01,
                                                                      nb02,
                                                                      nb03,
                                                                      ne10,
                                                                      ne11,
                                                                      ne12,
                                                                      nb10,
                                                                      nb11,
                                                                      nb12,
                                                                      nb13);
}

static void ggml_cpy_q4_1_f32_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1><<<num_blocks, 1, 0, stream>>>(cx,
                                                                                           cdst,
                                                                                           ne,
                                                                                           ne00,
                                                                                           ne01,
                                                                                           ne02,
                                                                                           nb00,
                                                                                           nb01,
                                                                                           nb02,
                                                                                           nb03,
                                                                                           ne10,
                                                                                           ne11,
                                                                                           ne12,
                                                                                           nb10,
                                                                                           nb11,
                                                                                           nb12,
                                                                                           nb13);
}

static void ggml_cpy_f32_q5_0_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int64_t num_blocks = ne / QK5_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, 1, 0, stream>>>(cx,
                                                                      cdst,
                                                                      ne,
                                                                      ne00,
                                                                      ne01,
                                                                      ne02,
                                                                      nb00,
                                                                      nb01,
                                                                      nb02,
                                                                      nb03,
                                                                      ne10,
                                                                      ne11,
                                                                      ne12,
                                                                      nb10,
                                                                      nb11,
                                                                      nb12,
                                                                      nb13);
}

static void ggml_cpy_q5_0_f32_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0><<<num_blocks, 1, 0, stream>>>(cx,
                                                                                           cdst,
                                                                                           ne,
                                                                                           ne00,
                                                                                           ne01,
                                                                                           ne02,
                                                                                           nb00,
                                                                                           nb01,
                                                                                           nb02,
                                                                                           nb03,
                                                                                           ne10,
                                                                                           ne11,
                                                                                           ne12,
                                                                                           nb10,
                                                                                           nb11,
                                                                                           nb12,
                                                                                           nb13);
}

static void ggml_cpy_f32_q5_1_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int64_t num_blocks = ne / QK5_1;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, 1, 0, stream>>>(cx,
                                                                      cdst,
                                                                      ne,
                                                                      ne00,
                                                                      ne01,
                                                                      ne02,
                                                                      nb00,
                                                                      nb01,
                                                                      nb02,
                                                                      nb03,
                                                                      ne10,
                                                                      ne11,
                                                                      ne12,
                                                                      nb10,
                                                                      nb11,
                                                                      nb12,
                                                                      nb13);
}

static void ggml_cpy_q5_1_f32_cuda(const char* cx,
                                   char* cdst,
                                   const int64_t ne,
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t nb00,
                                   const int64_t nb01,
                                   const int64_t nb02,
                                   const int64_t nb03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t nb10,
                                   const int64_t nb11,
                                   const int64_t nb12,
                                   const int64_t nb13,
                                   cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1><<<num_blocks, 1, 0, stream>>>(cx,
                                                                                           cdst,
                                                                                           ne,
                                                                                           ne00,
                                                                                           ne01,
                                                                                           ne02,
                                                                                           nb00,
                                                                                           nb01,
                                                                                           nb02,
                                                                                           nb03,
                                                                                           ne10,
                                                                                           ne11,
                                                                                           ne12,
                                                                                           nb10,
                                                                                           nb11,
                                                                                           nb12,
                                                                                           nb13);
}

static void ggml_cpy_f32_iq4_nl_cuda(const char* cx,
                                     char* cdst,
                                     const int64_t ne,
                                     const int64_t ne00,
                                     const int64_t ne01,
                                     const int64_t ne02,
                                     const int64_t nb00,
                                     const int64_t nb01,
                                     const int64_t nb02,
                                     const int64_t nb03,
                                     const int64_t ne10,
                                     const int64_t ne11,
                                     const int64_t ne12,
                                     const int64_t nb10,
                                     const int64_t nb11,
                                     const int64_t nb12,
                                     const int64_t nb13,
                                     cudaStream_t stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    const int64_t num_blocks = ne / QK4_NL;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, 1, 0, stream>>>(cx,
                                                                         cdst,
                                                                         ne,
                                                                         ne00,
                                                                         ne01,
                                                                         ne02,
                                                                         nb00,
                                                                         nb01,
                                                                         nb02,
                                                                         nb03,
                                                                         ne10,
                                                                         ne11,
                                                                         ne12,
                                                                         nb10,
                                                                         nb11,
                                                                         nb12,
                                                                         nb13);
}

void ggml_cuda_cpy(ggml_backend_cuda_context& ctx, const ggml_tensor* src0, ggml_tensor* src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    // GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    // GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char* src0_ddc = (char*) src0->data;
    char* src1_ddc = (char*) src1->data;

    const bool contiguous_srcs = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
    const bool use_to_contiguous_fastpath =
        ggml_is_contiguous(src1) &&
        getenv("GGML_CUDA_DISABLE_CPY_TO_CONTIGUOUS_FASTPATH") == nullptr;
    const bool can_be_transposed = nb01 == (int64_t) ggml_element_size(src0) && src0->ne[3] == 1 &&
                                   nb02 == ne00 * ne01 * (int64_t) ggml_element_size(src0);

    const bool profile_cpy = getenv("GGML_CUDA_PROFILE_CPY") != nullptr;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    if (profile_cpy) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_stop));
        CUDA_CHECK(cudaEventRecord(profile_start, main_stream));
    }

    if (src0->type == src1->type && contiguous_srcs) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
        } else
#endif  // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
        {
            CUDA_CHECK(cudaMemcpyAsync(
                src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<float, float, true>(src0_ddc,
                                                     src1_ddc,
                                                     ne,
                                                     ne00,
                                                     ne01,
                                                     ne02,
                                                     nb00,
                                                     nb01,
                                                     nb02,
                                                     nb03,
                                                     ne10,
                                                     ne11,
                                                     ne12,
                                                     nb10,
                                                     nb11,
                                                     nb12,
                                                     nb13,
                                                     main_stream);
        } else if (use_to_contiguous_fastpath) {
            ggml_cpy_scalar_cuda<float, float, false, true>(src0_ddc,
                                                            src1_ddc,
                                                            ne,
                                                            ne00,
                                                            ne01,
                                                            ne02,
                                                            nb00,
                                                            nb01,
                                                            nb02,
                                                            nb03,
                                                            ne10,
                                                            ne11,
                                                            ne12,
                                                            nb10,
                                                            nb11,
                                                            nb12,
                                                            nb13,
                                                            main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, float>(src0_ddc,
                                               src1_ddc,
                                               ne,
                                               ne00,
                                               ne01,
                                               ne02,
                                               nb00,
                                               nb01,
                                               nb02,
                                               nb03,
                                               ne10,
                                               ne11,
                                               ne12,
                                               nb10,
                                               nb11,
                                               nb12,
                                               nb13,
                                               main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            if (cpy_f32_to_lowp_contiguous_vec4_enabled<nv_bfloat162>(src0_ddc, src1_ddc, ne)) {
                ggml_cpy_f32_to_lowp_contiguous_vec4_cuda<nv_bfloat162>(
                    src0_ddc, src1_ddc, ne, main_stream);
            } else {
                ggml_cpy_scalar_contiguous_cuda<float, nv_bfloat16>(
                    src0_ddc, src1_ddc, ne, main_stream);
            }
        } else {
            ggml_cpy_scalar_cuda<float, nv_bfloat16>(src0_ddc,
                                                     src1_ddc,
                                                     ne,
                                                     ne00,
                                                     ne01,
                                                     ne02,
                                                     nb00,
                                                     nb01,
                                                     nb02,
                                                     nb03,
                                                     ne10,
                                                     ne11,
                                                     ne12,
                                                     nb10,
                                                     nb11,
                                                     nb12,
                                                     nb13,
                                                     main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            if (cpy_f32_to_lowp_contiguous_vec4_enabled<half2>(src0_ddc, src1_ddc, ne)) {
                ggml_cpy_f32_to_lowp_contiguous_vec4_cuda<half2>(
                    src0_ddc, src1_ddc, ne, main_stream);
            } else {
                ggml_cpy_scalar_contiguous_cuda<float, half>(src0_ddc, src1_ddc, ne, main_stream);
            }
        } else {
            ggml_cpy_scalar_cuda<float, half>(src0_ddc,
                                              src1_ddc,
                                              ne,
                                              ne00,
                                              ne01,
                                              ne02,
                                              nb00,
                                              nb01,
                                              nb02,
                                              nb03,
                                              ne10,
                                              ne11,
                                              ne12,
                                              nb10,
                                              nb11,
                                              nb12,
                                              nb13,
                                              main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda(src0_ddc,
                                 src1_ddc,
                                 ne,
                                 ne00,
                                 ne01,
                                 ne02,
                                 nb00,
                                 nb01,
                                 nb02,
                                 nb03,
                                 ne10,
                                 ne11,
                                 ne12,
                                 nb10,
                                 nb11,
                                 nb12,
                                 nb13,
                                 main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_cuda(src0_ddc,
                               src1_ddc,
                               ne,
                               ne00,
                               ne01,
                               ne02,
                               nb00,
                               nb01,
                               nb02,
                               nb03,
                               ne10,
                               ne11,
                               ne12,
                               nb10,
                               nb11,
                               nb12,
                               nb13,
                               main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<half, half, true>(src0_ddc,
                                                   src1_ddc,
                                                   ne,
                                                   ne00,
                                                   ne01,
                                                   ne02,
                                                   nb00,
                                                   nb01,
                                                   nb02,
                                                   nb03,
                                                   ne10,
                                                   ne11,
                                                   ne12,
                                                   nb10,
                                                   nb11,
                                                   nb12,
                                                   nb13,
                                                   main_stream);
        } else if (use_to_contiguous_fastpath) {
            ggml_cpy_scalar_cuda<half, half, false, true>(src0_ddc,
                                                          src1_ddc,
                                                          ne,
                                                          ne00,
                                                          ne01,
                                                          ne02,
                                                          nb00,
                                                          nb01,
                                                          nb02,
                                                          nb03,
                                                          ne10,
                                                          ne11,
                                                          ne12,
                                                          nb10,
                                                          nb11,
                                                          nb12,
                                                          nb13,
                                                          main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, half>(src0_ddc,
                                             src1_ddc,
                                             ne,
                                             ne00,
                                             ne01,
                                             ne02,
                                             nb00,
                                             nb01,
                                             nb02,
                                             nb03,
                                             ne10,
                                             ne11,
                                             ne12,
                                             nb10,
                                             nb11,
                                             nb12,
                                             nb13,
                                             main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, nv_bfloat16>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, nv_bfloat16>(src0_ddc,
                                                    src1_ddc,
                                                    ne,
                                                    ne00,
                                                    ne01,
                                                    ne02,
                                                    nb00,
                                                    nb01,
                                                    nb02,
                                                    nb03,
                                                    ne10,
                                                    ne11,
                                                    ne12,
                                                    nb10,
                                                    nb11,
                                                    nb12,
                                                    nb13,
                                                    main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, float>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, float>(src0_ddc,
                                              src1_ddc,
                                              ne,
                                              ne00,
                                              ne01,
                                              ne02,
                                              nb00,
                                              nb01,
                                              nb02,
                                              nb03,
                                              ne10,
                                              ne11,
                                              ne12,
                                              nb10,
                                              nb11,
                                              nb12,
                                              nb13,
                                              main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16, true>(src0_ddc,
                                                                 src1_ddc,
                                                                 ne,
                                                                 ne00,
                                                                 ne01,
                                                                 ne02,
                                                                 nb00,
                                                                 nb01,
                                                                 nb02,
                                                                 nb03,
                                                                 ne10,
                                                                 ne11,
                                                                 ne12,
                                                                 nb10,
                                                                 nb11,
                                                                 nb12,
                                                                 nb13,
                                                                 main_stream);
        } else if (use_to_contiguous_fastpath) {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16, false, true>(src0_ddc,
                                                                        src1_ddc,
                                                                        ne,
                                                                        ne00,
                                                                        ne01,
                                                                        ne02,
                                                                        nb00,
                                                                        nb01,
                                                                        nb02,
                                                                        nb03,
                                                                        ne10,
                                                                        ne11,
                                                                        ne12,
                                                                        nb10,
                                                                        nb11,
                                                                        nb12,
                                                                        nb13,
                                                                        main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16>(src0_ddc,
                                                           src1_ddc,
                                                           ne,
                                                           ne00,
                                                           ne01,
                                                           ne02,
                                                           nb00,
                                                           nb01,
                                                           nb02,
                                                           nb03,
                                                           ne10,
                                                           ne11,
                                                           ne12,
                                                           nb10,
                                                           nb11,
                                                           nb12,
                                                           nb13,
                                                           main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, half>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, half>(src0_ddc,
                                                    src1_ddc,
                                                    ne,
                                                    ne00,
                                                    ne01,
                                                    ne02,
                                                    nb00,
                                                    nb01,
                                                    nb02,
                                                    nb03,
                                                    ne10,
                                                    ne11,
                                                    ne12,
                                                    nb10,
                                                    nb11,
                                                    nb12,
                                                    nb13,
                                                    main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, float>(
                src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, float>(src0_ddc,
                                                     src1_ddc,
                                                     ne,
                                                     ne00,
                                                     ne01,
                                                     ne02,
                                                     nb00,
                                                     nb01,
                                                     nb02,
                                                     nb03,
                                                     ne10,
                                                     ne11,
                                                     ne12,
                                                     nb10,
                                                     nb11,
                                                     nb12,
                                                     nb13,
                                                     main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<int32_t, int32_t, true>(src0_ddc,
                                                         src1_ddc,
                                                         ne,
                                                         ne00,
                                                         ne01,
                                                         ne02,
                                                         nb00,
                                                         nb01,
                                                         nb02,
                                                         nb03,
                                                         ne10,
                                                         ne11,
                                                         ne12,
                                                         nb10,
                                                         nb11,
                                                         nb12,
                                                         nb13,
                                                         main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, int32_t>(src0_ddc,
                                                   src1_ddc,
                                                   ne,
                                                   ne00,
                                                   ne01,
                                                   ne02,
                                                   nb00,
                                                   nb01,
                                                   nb02,
                                                   nb03,
                                                   ne10,
                                                   ne11,
                                                   ne12,
                                                   nb10,
                                                   nb11,
                                                   nb12,
                                                   nb13,
                                                   main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, int32_t>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, int32_t>(src0_ddc,
                                                 src1_ddc,
                                                 ne,
                                                 ne00,
                                                 ne01,
                                                 ne02,
                                                 nb00,
                                                 nb01,
                                                 nb02,
                                                 nb03,
                                                 ne10,
                                                 ne11,
                                                 ne12,
                                                 nb10,
                                                 nb11,
                                                 nb12,
                                                 nb13,
                                                 main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<int32_t, float>(src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, float>(src0_ddc,
                                                 src1_ddc,
                                                 ne,
                                                 ne00,
                                                 ne01,
                                                 ne02,
                                                 nb00,
                                                 nb01,
                                                 nb02,
                                                 nb03,
                                                 ne10,
                                                 ne11,
                                                 ne12,
                                                 nb10,
                                                 nb11,
                                                 nb12,
                                                 nb13,
                                                 main_stream);
        }
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n",
                   __func__,
                   ggml_type_name(src0->type),
                   ggml_type_name(src1->type));
    }

    if (profile_cpy) {
        CUDA_CHECK(cudaEventRecord(profile_stop, main_stream));
        CUDA_CHECK(cudaEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        fprintf(stderr,
                "GGML_CUDA_PROFILE_CPY ms=%.6f src=%s dst=%s types=%s->%s "
                "src_ne=[%lld,%lld,%lld,%lld] dst_ne=[%lld,%lld,%lld,%lld] "
                "src_nb=[%lld,%lld,%lld,%lld] dst_nb=[%lld,%lld,%lld,%lld] "
                "contiguous_srcs=%d dst_contiguous=%d to_contiguous_fastpath=%d "
                "can_be_transposed=%d ne=%lld\n",
                elapsed_ms,
                src0->name,
                src1->name,
                ggml_type_name(src0->type),
                ggml_type_name(src1->type),
                (long long) src0->ne[0],
                (long long) src0->ne[1],
                (long long) src0->ne[2],
                (long long) src0->ne[3],
                (long long) src1->ne[0],
                (long long) src1->ne[1],
                (long long) src1->ne[2],
                (long long) src1->ne[3],
                (long long) src0->nb[0],
                (long long) src0->nb[1],
                (long long) src0->nb[2],
                (long long) src0->nb[3],
                (long long) src1->nb[0],
                (long long) src1->nb[1],
                (long long) src1->nb[2],
                (long long) src1->nb[3],
                contiguous_srcs ? 1 : 0,
                ggml_is_contiguous(src1) ? 1 : 0,
                use_to_contiguous_fastpath ? 1 : 0,
                can_be_transposed ? 1 : 0,
                (long long) ne);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_stop));
    }
}

bool ggml_cuda_cpy_pair_to_contiguous(ggml_backend_cuda_context& ctx,
                                      const ggml_tensor* src0,
                                      ggml_tensor* dst0,
                                      const ggml_tensor* src1,
                                      ggml_tensor* dst1) {
    if (std::getenv("GGML_CUDA_DISABLE_CPY_PAIR_CONT_FUSION") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_CPY_PAIR_CONT_FUSION"))) {
        return false;
    }

    if (src0 == nullptr || src1 == nullptr || dst0 == nullptr || dst1 == nullptr) {
        return false;
    }
    if (src0->type != src1->type || src0->type != dst0->type || src1->type != dst1->type) {
        return false;
    }
    if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, dst0) ||
        !ggml_are_same_shape(src1, dst1) || !ggml_is_contiguous(dst0) ||
        !ggml_is_contiguous(dst1)) {
        return false;
    }
    const auto tensor_end = [](const ggml_tensor* tensor) {
        return (uintptr_t) tensor->data + ggml_nbytes(tensor);
    };
    const uintptr_t dst0_start = (uintptr_t) dst0->data;
    const uintptr_t dst1_start = (uintptr_t) dst1->data;
    if (dst0_start < tensor_end(dst1) && dst1_start < tensor_end(dst0)) {
        return false;
    }
    if (src0->nb[0] != src1->nb[0] || src0->nb[1] != src1->nb[1] || src0->nb[2] != src1->nb[2] ||
        src0->nb[3] != src1->nb[3]) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nelements(src0) / ne00;
    if (ne00 < 32 || nrows >= UINT_MAX) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_F32:
            if (src0->nb[0] == sizeof(float) && ne00 % 4 == 0 &&
                reinterpret_cast<uintptr_t>(src0->data) % alignof(float4) == 0 &&
                reinterpret_cast<uintptr_t>(src1->data) % alignof(float4) == 0 &&
                reinterpret_cast<uintptr_t>(dst0->data) % alignof(float4) == 0 &&
                reinterpret_cast<uintptr_t>(dst1->data) % alignof(float4) == 0 &&
                src0->nb[1] % alignof(float4) == 0 && src0->nb[2] % alignof(float4) == 0 &&
                src0->nb[3] % alignof(float4) == 0) {
                constexpr int rows_per_block = 4;
                const int64_t num_blocks = (nrows + rows_per_block - 1) / rows_per_block;
                const int64_t row_vec4 = ne00 / 4;
                const int threads_per_row = row_vec4 < 64 ? int(row_vec4) : 64;
                GGML_ASSERT(num_blocks < UINT_MAX);
                cpy_pair_f32_to_contiguous_rows_vec4_group4<<<num_blocks,
                                                              dim3(threads_per_row, rows_per_block),
                                                              0,
                                                              ctx.stream()>>>(
                    (const char*) src0->data,
                    (const char*) src1->data,
                    (char*) dst0->data,
                    (char*) dst1->data,
                    ne00,
                    src0->ne[1],
                    src0->ne[2],
                    nrows,
                    src0->nb[1],
                    src0->nb[2],
                    src0->nb[3]);
            } else {
                cpy_pair_to_contiguous_rows<float>
                    <<<nrows, CUDA_CPY_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                      (const char*) src1->data,
                                                                      (char*) dst0->data,
                                                                      (char*) dst1->data,
                                                                      ne00,
                                                                      src0->ne[1],
                                                                      src0->ne[2],
                                                                      nrows,
                                                                      src0->nb[0],
                                                                      src0->nb[1],
                                                                      src0->nb[2],
                                                                      src0->nb[3]);
            }
            return true;
        case GGML_TYPE_F16:
            cpy_pair_to_contiguous_rows<half>
                <<<nrows, CUDA_CPY_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                  (const char*) src1->data,
                                                                  (char*) dst0->data,
                                                                  (char*) dst1->data,
                                                                  ne00,
                                                                  src0->ne[1],
                                                                  src0->ne[2],
                                                                  nrows,
                                                                  src0->nb[0],
                                                                  src0->nb[1],
                                                                  src0->nb[2],
                                                                  src0->nb[3]);
            return true;
        case GGML_TYPE_BF16:
            cpy_pair_to_contiguous_rows<nv_bfloat16>
                <<<nrows, CUDA_CPY_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                  (const char*) src1->data,
                                                                  (char*) dst0->data,
                                                                  (char*) dst1->data,
                                                                  ne00,
                                                                  src0->ne[1],
                                                                  src0->ne[2],
                                                                  nrows,
                                                                  src0->nb[0],
                                                                  src0->nb[1],
                                                                  src0->nb[2],
                                                                  src0->nb[3]);
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_cpy_bias_axis0(ggml_backend_cuda_context& ctx,
                              const ggml_tensor* src,
                              const ggml_tensor* bias,
                              ggml_tensor* dst) {
    if (std::getenv("GGML_CUDA_DISABLE_CPY_BIAS_AXIS0_FUSION") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_CPY_BIAS_AXIS0_FUSION"))) {
        return false;
    }

    if (src->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst) || !ggml_is_contiguous(dst) || !ggml_is_contiguous(bias)) {
        return false;
    }
    if (bias->ne[0] != dst->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
        return false;
    }

    const int64_t ne00 = src->ne[0];
    const int64_t nrows = ggml_nelements(src) / ne00;
    if (ne00 < 32 || nrows >= UINT_MAX) {
        return false;
    }

    cpy_f32_to_contiguous_rows_bias_axis0<<<nrows, CUDA_CPY_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const char*) src->data,
        (const float*) bias->data,
        (float*) dst->data,
        ne00,
        src->ne[1],
        src->ne[2],
        nrows,
        src->nb[0],
        src->nb[1],
        src->nb[2],
        src->nb[3]);
    return true;
}

bool ggml_cuda_add_bias_axis2_permute_cont(ggml_backend_cuda_context& ctx,
                                           const ggml_tensor* add,
                                           ggml_tensor* cont) {
    if (std::getenv("GGML_CUDA_DISABLE_ADD_BIAS_PERMUTE_CONT_FUSION") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_ADD_BIAS_PERMUTE_CONT_FUSION"))) {
        return false;
    }

    if (add == nullptr || cont == nullptr || add->op != GGML_OP_ADD || cont->op != GGML_OP_CONT ||
        cont->src[0] == nullptr || cont->src[0]->op != GGML_OP_PERMUTE ||
        cont->src[0]->src[0] != add) {
        return false;
    }

    const ggml_tensor* src = nullptr;
    const ggml_tensor* bias = nullptr;
    if (add->src[0] != nullptr && add->src[1] != nullptr && add->src[0]->type == GGML_TYPE_F32 &&
        add->src[1]->type == GGML_TYPE_F32) {
        if (add->src[1]->ne[0] == 1 && add->src[1]->ne[1] == 1 &&
            add->src[1]->ne[2] == add->ne[2] && add->src[1]->ne[3] == 1) {
            src = add->src[0];
            bias = add->src[1];
        } else if (add->src[0]->ne[0] == 1 && add->src[0]->ne[1] == 1 &&
                   add->src[0]->ne[2] == add->ne[2] && add->src[0]->ne[3] == 1) {
            src = add->src[1];
            bias = add->src[0];
        }
    }

    if (src == nullptr || bias == nullptr || add->type != GGML_TYPE_F32 ||
        cont->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, add) || !ggml_is_contiguous(src) || !ggml_is_contiguous(bias) ||
        !ggml_is_contiguous(cont)) {
        return false;
    }
    if (cont->ne[0] != add->ne[2] || cont->ne[1] != add->ne[0] || cont->ne[2] != add->ne[1] ||
        cont->ne[3] != add->ne[3]) {
        return false;
    }

    const int64_t ne = ggml_nelements(cont);
    const int64_t blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(blocks < UINT_MAX);
    add_bias_axis2_permute_2013_cont<<<blocks, CUDA_CPY_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float*) src->data,
        (const float*) bias->data,
        (float*) cont->data,
        add->ne[0],
        add->ne[1],
        add->ne[2],
        add->ne[3]);
    return true;
}

void ggml_cuda_dup(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];
    ggml_cuda_cpy(ctx, src0, dst);
}
