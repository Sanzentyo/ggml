#include "win-part.cuh"

static __global__ void win_part_f32(const char* __restrict__ src,
                                    float* __restrict__ dst,
                                    const int64_t ne00,
                                    const int64_t ne01,
                                    const int64_t ne02,
                                    const size_t nb01,
                                    const size_t nb02,
                                    const int64_t ne0,
                                    const int64_t ne1,
                                    const int64_t ne2,
                                    const int32_t npx,
                                    const int32_t npy,
                                    const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2 * int64_t(npx) * npy;

    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    const int64_t rem1 = idx / ne0;
    const int64_t i1 = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2 = rem2 % ne2;
    const int64_t i3 = rem2 / ne2;

    const int64_t py = i3 / npx;
    const int64_t px = i3 % npx;

    const int64_t src_col = px * w + i1;
    const int64_t src_row = py * w + i2;

    if (src_col >= ne01 || src_row >= ne02) {
        dst[idx] = 0.0f;
        return;
    }

    const char* src_ptr = src + src_row * nb02 + src_col * nb01;
    dst[idx] = reinterpret_cast<const float*>(src_ptr)[i0];
}

static __global__ void win_unpart_f32(const char* __restrict__ src,
                                      float* __restrict__ dst,
                                      const size_t nb01,
                                      const size_t nb02,
                                      const size_t nb03,
                                      const int64_t ne0,
                                      const int64_t ne1,
                                      const int64_t ne2,
                                      const int32_t npx,
                                      const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2;

    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    const int64_t rem = idx / ne0;
    const int64_t i1 = rem % ne1;
    const int64_t i2 = rem / ne1;

    const int64_t px = i1 / w;
    const int64_t py = i2 / w;
    const int64_t wx = i1 % w;
    const int64_t wy = i2 % w;
    const int64_t wi = py * npx + px;

    const char* src_ptr = src + wi * nb03 + wy * nb02 + wx * nb01;
    dst[idx] = reinterpret_cast<const float*>(src_ptr)[i0];
}

void ggml_cuda_op_win_part(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t npx = ((const int32_t*) dst->op_params)[0];
    const int32_t npy = ((const int32_t*) dst->op_params)[1];
    const int32_t w = ((const int32_t*) dst->op_params)[2];

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t blocks = (total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;

    win_part_f32<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                        (float*) dst->data,
                                                                        src0->ne[0],
                                                                        src0->ne[1],
                                                                        src0->ne[2],
                                                                        src0->nb[1],
                                                                        src0->nb[2],
                                                                        dst->ne[0],
                                                                        dst->ne[1],
                                                                        dst->ne[2],
                                                                        npx,
                                                                        npy,
                                                                        w);
}

void ggml_cuda_op_win_unpart(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t w = ((const int32_t*) dst->op_params)[0];
    const int px = (w - int(dst->ne[1] % w)) % w;
    const int npx = int(px + dst->ne[1]) / w;

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2];
    const int64_t blocks = (total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;

    win_unpart_f32<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                          (float*) dst->data,
                                                                          src0->nb[1],
                                                                          src0->nb[2],
                                                                          src0->nb[3],
                                                                          dst->ne[0],
                                                                          dst->ne[1],
                                                                          dst->ne[2],
                                                                          npx,
                                                                          w);
}
