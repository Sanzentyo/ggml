#include "convert.cuh"
#include "win-part.cuh"

static bool ggml_cuda_win_part_vec4_enabled() {
    const char* env = getenv("GGML_CUDA_DISABLE_WIN_PART_VEC4");
    return env == nullptr || std::atoi(env) == 0;
}

static bool ggml_cuda_win_part_f32_vec4_compatible(const ggml_tensor* src0,
                                                   const ggml_tensor* dst) {
    return src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           ggml_cuda_win_part_vec4_enabled() && src0->ne[0] % 4 == 0 && dst->ne[0] % 4 == 0 &&
           src0->data != nullptr && dst->data != nullptr &&
           reinterpret_cast<uintptr_t>(src0->data) % alignof(float4) == 0 &&
           reinterpret_cast<uintptr_t>(dst->data) % alignof(float4) == 0 &&
           src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
           src0->nb[1] % sizeof(float4) == 0 && src0->nb[2] % sizeof(float4) == 0 &&
           src0->nb[3] % sizeof(float4) == 0 && ggml_is_contiguous(dst);
}

template <typename T>
static __global__ void win_part(const char* __restrict__ src,
                                T* __restrict__ dst,
                                const int64_t ne00,
                                const int64_t ne01,
                                const int64_t ne02,
                                const size_t nb01,
                                const size_t nb02,
                                const size_t nb03,
                                const int64_t ne0,
                                const int64_t ne1,
                                const int64_t ne2,
                                const int64_t ne3,
                                const int32_t npx,
                                const int32_t npy,
                                const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2 * ne3;

    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    const int64_t rem1 = idx / ne0;
    const int64_t i1 = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2 = rem2 % ne2;
    const int64_t i3 = rem2 / ne2;

    const int64_t np = int64_t(npx) * npy;
    const int64_t batch = i3 / np;
    const int64_t window = i3 % np;
    const int64_t py = window / npx;
    const int64_t px = window % npx;

    const int64_t src_col = px * w + i1;
    const int64_t src_row = py * w + i2;

    if (src_col >= ne01 || src_row >= ne02) {
        dst[idx] = T(0.0f);
        return;
    }

    const char* src_ptr = src + batch * nb03 + src_row * nb02 + src_col * nb01;
    dst[idx] = reinterpret_cast<const T*>(src_ptr)[i0];
}

static __global__ void win_part_f32_vec4(const char* __restrict__ src,
                                         float4* __restrict__ dst,
                                         const int64_t ne00_vec4,
                                         const int64_t ne01,
                                         const int64_t ne02,
                                         const size_t nb01,
                                         const size_t nb02,
                                         const size_t nb03,
                                         const int64_t ne1,
                                         const int64_t ne2,
                                         const int64_t ne3,
                                         const int32_t npx,
                                         const int32_t npy,
                                         const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne00_vec4 * ne1 * ne2 * ne3;

    if (idx >= total) {
        return;
    }

    const int64_t i0v = idx % ne00_vec4;
    const int64_t rem1 = idx / ne00_vec4;
    const int64_t i1 = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2 = rem2 % ne2;
    const int64_t i3 = rem2 / ne2;

    const int64_t np = int64_t(npx) * npy;
    const int64_t batch = i3 / np;
    const int64_t window = i3 % np;
    const int64_t py = window / npx;
    const int64_t px = window % npx;

    const int64_t src_col = px * w + i1;
    const int64_t src_row = py * w + i2;

    if (src_col >= ne01 || src_row >= ne02) {
        dst[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const char* src_ptr = src + batch * nb03 + src_row * nb02 + src_col * nb01;
    dst[idx] = reinterpret_cast<const float4*>(src_ptr)[i0v];
}

template <typename T>
static __global__ void win_part_f32_cpy(const char* __restrict__ src,
                                        float* __restrict__ win_part_out,
                                        T* __restrict__ dst,
                                        const int64_t ne00,
                                        const int64_t ne01,
                                        const int64_t ne02,
                                        const size_t nb01,
                                        const size_t nb02,
                                        const size_t nb03,
                                        const int64_t ne0,
                                        const int64_t ne1,
                                        const int64_t ne2,
                                        const int64_t ne3,
                                        const int32_t npx,
                                        const int32_t npy,
                                        const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2 * ne3;

    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    const int64_t rem1 = idx / ne0;
    const int64_t i1 = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2 = rem2 % ne2;
    const int64_t i3 = rem2 / ne2;

    const int64_t np = int64_t(npx) * npy;
    const int64_t batch = i3 / np;
    const int64_t window = i3 % np;
    const int64_t py = window / npx;
    const int64_t px = window % npx;

    const int64_t src_col = px * w + i1;
    const int64_t src_row = py * w + i2;

    if (src_col >= ne01 || src_row >= ne02) {
        if (win_part_out != nullptr) {
            win_part_out[idx] = 0.0f;
        }
        dst[idx] = T(0.0f);
        return;
    }

    const char* src_ptr = src + batch * nb03 + src_row * nb02 + src_col * nb01;
    const float value = reinterpret_cast<const float*>(src_ptr)[i0];
    if (win_part_out != nullptr) {
        win_part_out[idx] = value;
    }
    dst[idx] = ggml_cuda_cast<T>(value);
}

template <typename T>
static __global__ void win_unpart(const char* __restrict__ src,
                                  T* __restrict__ dst,
                                  const size_t nb01,
                                  const size_t nb02,
                                  const size_t nb03,
                                  const int64_t ne0,
                                  const int64_t ne1,
                                  const int64_t ne2,
                                  const int64_t ne3,
                                  const int32_t npx,
                                  const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2 * ne3;

    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    const int64_t rem1 = idx / ne0;
    const int64_t i1 = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2 = rem2 % ne2;
    const int64_t i3 = rem2 / ne2;

    const int64_t px = i1 / w;
    const int64_t py = i2 / w;
    const int64_t wx = i1 % w;
    const int64_t wy = i2 % w;
    const int64_t wi = i3 * (npx * ((ne2 + w - 1) / w)) + py * npx + px;

    const char* src_ptr = src + wi * nb03 + wy * nb02 + wx * nb01;
    dst[idx] = reinterpret_cast<const T*>(src_ptr)[i0];
}

static __global__ void win_unpart_f32_vec4(const char* __restrict__ src,
                                           float4* __restrict__ dst,
                                           const size_t nb01,
                                           const size_t nb02,
                                           const size_t nb03,
                                           const int64_t ne0_vec4,
                                           const int64_t ne1,
                                           const int64_t ne2,
                                           const int64_t ne3,
                                           const int32_t npx,
                                           const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne0_vec4 * ne1 * ne2 * ne3;

    if (idx >= total) {
        return;
    }

    const int64_t i0v = idx % ne0_vec4;
    const int64_t rem1 = idx / ne0_vec4;
    const int64_t i1 = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2 = rem2 % ne2;
    const int64_t i3 = rem2 / ne2;

    const int64_t px = i1 / w;
    const int64_t py = i2 / w;
    const int64_t wx = i1 % w;
    const int64_t wy = i2 % w;
    const int64_t wi = i3 * (npx * ((ne2 + w - 1) / w)) + py * npx + px;

    const char* src_ptr = src + wi * nb03 + wy * nb02 + wx * nb01;
    dst[idx] = reinterpret_cast<const float4*>(src_ptr)[i0v];
}

static __global__ void win_unpart_add_f32_vec4(const char* __restrict__ src,
                                               const float4* __restrict__ residual,
                                               float4* __restrict__ dst,
                                               const size_t nb01,
                                               const size_t nb02,
                                               const size_t nb03,
                                               const int64_t ne0_vec4,
                                               const int64_t ne1,
                                               const int64_t ne2,
                                               const int64_t ne3,
                                               const int32_t npx,
                                               const int32_t w) {
    const int64_t idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t total = ne0_vec4 * ne1 * ne2 * ne3;

    if (idx >= total) {
        return;
    }

    const int64_t i0v = idx % ne0_vec4;
    const int64_t rem1 = idx / ne0_vec4;
    const int64_t i1 = rem1 % ne1;
    const int64_t rem2 = rem1 / ne1;
    const int64_t i2 = rem2 % ne2;
    const int64_t i3 = rem2 / ne2;

    const int64_t px = i1 / w;
    const int64_t py = i2 / w;
    const int64_t wx = i1 % w;
    const int64_t wy = i2 % w;
    const int64_t wi = i3 * (npx * ((ne2 + w - 1) / w)) + py * npx + px;

    const char* src_ptr = src + wi * nb03 + wy * nb02 + wx * nb01;
    const float4 unpart = reinterpret_cast<const float4*>(src_ptr)[i0v];
    const float4 res = residual[idx];
    dst[idx] = make_float4(res.x + unpart.x, res.y + unpart.y, res.z + unpart.z, res.w + unpart.w);
}

void ggml_cuda_op_win_part(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    const int32_t npx = ((const int32_t*) dst->op_params)[0];
    const int32_t npy = ((const int32_t*) dst->op_params)[1];
    const int32_t w = ((const int32_t*) dst->op_params)[2];

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t blocks = (total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;

    if (ggml_cuda_win_part_f32_vec4_compatible(src0, dst)) {
        const int64_t vec_total = total / 4;
        const int64_t vec_blocks =
            (vec_total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;
        win_part_f32_vec4<<<vec_blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char*) src0->data,
            (float4*) dst->data,
            dst->ne[0] / 4,
            src0->ne[1],
            src0->ne[2],
            src0->nb[1],
            src0->nb[2],
            src0->nb[3],
            dst->ne[1],
            dst->ne[2],
            dst->ne[3],
            npx,
            npy,
            w);
    } else if (src0->type == GGML_TYPE_F16) {
        win_part<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                        (half*) dst->data,
                                                                        src0->ne[0],
                                                                        src0->ne[1],
                                                                        src0->ne[2],
                                                                        src0->nb[1],
                                                                        src0->nb[2],
                                                                        src0->nb[3],
                                                                        dst->ne[0],
                                                                        dst->ne[1],
                                                                        dst->ne[2],
                                                                        dst->ne[3],
                                                                        npx,
                                                                        npy,
                                                                        w);
    } else if (src0->type == GGML_TYPE_BF16) {
        win_part<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                        (nv_bfloat16*) dst->data,
                                                                        src0->ne[0],
                                                                        src0->ne[1],
                                                                        src0->ne[2],
                                                                        src0->nb[1],
                                                                        src0->nb[2],
                                                                        src0->nb[3],
                                                                        dst->ne[0],
                                                                        dst->ne[1],
                                                                        dst->ne[2],
                                                                        dst->ne[3],
                                                                        npx,
                                                                        npy,
                                                                        w);
    } else {
        win_part<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                        (float*) dst->data,
                                                                        src0->ne[0],
                                                                        src0->ne[1],
                                                                        src0->ne[2],
                                                                        src0->nb[1],
                                                                        src0->nb[2],
                                                                        src0->nb[3],
                                                                        dst->ne[0],
                                                                        dst->ne[1],
                                                                        dst->ne[2],
                                                                        dst->ne[3],
                                                                        npx,
                                                                        npy,
                                                                        w);
    }
}

bool ggml_cuda_op_win_part_cpy(ggml_backend_cuda_context& ctx,
                               ggml_tensor* win_part_node,
                               ggml_tensor* cpy_node) {
    if (std::getenv("GGML_CUDA_DISABLE_WIN_PART_CPY_FUSION") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_WIN_PART_CPY_FUSION"))) {
        return false;
    }

    const ggml_tensor* src0 = win_part_node->src[0];
    ggml_tensor* dst = cpy_node->src[1];
    const bool explicit_enable =
        std::getenv("GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_ENABLE_WIN_PART_CPY_FUSION")) != 0;

    if (win_part_node->op != GGML_OP_WIN_PART || cpy_node->op != GGML_OP_CPY ||
        cpy_node->src[0] != win_part_node || dst != cpy_node || src0->type != GGML_TYPE_F32 ||
        win_part_node->type != GGML_TYPE_F32 ||
        (dst->type != GGML_TYPE_BF16 && dst->type != GGML_TYPE_F16) ||
        !ggml_are_same_shape(win_part_node, dst) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type == GGML_TYPE_BF16 && !explicit_enable) {
        return false;
    }

    const int32_t npx = ((const int32_t*) win_part_node->op_params)[0];
    const int32_t npy = ((const int32_t*) win_part_node->op_params)[1];
    const int32_t w = ((const int32_t*) win_part_node->op_params)[2];

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t blocks = (total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;

    if (dst->type == GGML_TYPE_BF16) {
        win_part_f32_cpy<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char*) src0->data,
            (float*) win_part_node->data,
            (nv_bfloat16*) dst->data,
            src0->ne[0],
            src0->ne[1],
            src0->ne[2],
            src0->nb[1],
            src0->nb[2],
            src0->nb[3],
            dst->ne[0],
            dst->ne[1],
            dst->ne[2],
            dst->ne[3],
            npx,
            npy,
            w);
        return true;
    }

    win_part_f32_cpy<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const char*) src0->data,
        (float*) win_part_node->data,
        (half*) dst->data,
        src0->ne[0],
        src0->ne[1],
        src0->ne[2],
        src0->nb[1],
        src0->nb[2],
        src0->nb[3],
        dst->ne[0],
        dst->ne[1],
        dst->ne[2],
        dst->ne[3],
        npx,
        npy,
        w);
    return true;
}

void ggml_cuda_op_win_unpart(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    const int32_t w = ((const int32_t*) dst->op_params)[0];
    const int px = (w - int(dst->ne[1] % w)) % w;
    const int npx = int(px + dst->ne[1]) / w;

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t blocks = (total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;

    if (ggml_cuda_win_part_f32_vec4_compatible(src0, dst)) {
        const int64_t vec_total = total / 4;
        const int64_t vec_blocks =
            (vec_total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;
        win_unpart_f32_vec4<<<vec_blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char*) src0->data,
            (float4*) dst->data,
            src0->nb[1],
            src0->nb[2],
            src0->nb[3],
            dst->ne[0] / 4,
            dst->ne[1],
            dst->ne[2],
            dst->ne[3],
            npx,
            w);
    } else if (src0->type == GGML_TYPE_F16) {
        win_unpart<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                          (half*) dst->data,
                                                                          src0->nb[1],
                                                                          src0->nb[2],
                                                                          src0->nb[3],
                                                                          dst->ne[0],
                                                                          dst->ne[1],
                                                                          dst->ne[2],
                                                                          dst->ne[3],
                                                                          npx,
                                                                          w);
    } else if (src0->type == GGML_TYPE_BF16) {
        win_unpart<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                          (nv_bfloat16*) dst->data,
                                                                          src0->nb[1],
                                                                          src0->nb[2],
                                                                          src0->nb[3],
                                                                          dst->ne[0],
                                                                          dst->ne[1],
                                                                          dst->ne[2],
                                                                          dst->ne[3],
                                                                          npx,
                                                                          w);
    } else {
        win_unpart<<<blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>((const char*) src0->data,
                                                                          (float*) dst->data,
                                                                          src0->nb[1],
                                                                          src0->nb[2],
                                                                          src0->nb[3],
                                                                          dst->ne[0],
                                                                          dst->ne[1],
                                                                          dst->ne[2],
                                                                          dst->ne[3],
                                                                          npx,
                                                                          w);
    }
}

bool ggml_cuda_op_win_unpart_add(ggml_backend_cuda_context& ctx,
                                 ggml_tensor* win_unpart_node,
                                 ggml_tensor* add_node) {
    if (std::getenv("GGML_CUDA_DISABLE_WIN_UNPART_ADD_FUSION") != nullptr &&
        std::atoi(std::getenv("GGML_CUDA_DISABLE_WIN_UNPART_ADD_FUSION"))) {
        return false;
    }

    if (win_unpart_node->op != GGML_OP_WIN_UNPART || add_node->op != GGML_OP_ADD ||
        win_unpart_node->type != GGML_TYPE_F32 || add_node->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(win_unpart_node, add_node) || !ggml_is_contiguous(add_node)) {
        return false;
    }

    const ggml_tensor* src0 = win_unpart_node->src[0];
    const ggml_tensor* residual = nullptr;
    if (add_node->src[0] == win_unpart_node) {
        residual = add_node->src[1];
    } else if (add_node->src[1] == win_unpart_node) {
        residual = add_node->src[0];
    } else {
        return false;
    }

    if (src0 == nullptr || src0->type != GGML_TYPE_F32 || residual == nullptr ||
        residual->type != GGML_TYPE_F32 || !ggml_are_same_shape(residual, add_node) ||
        !ggml_is_contiguous(residual)) {
        return false;
    }

    const int32_t w = ((const int32_t*) win_unpart_node->op_params)[0];
    const int px = (w - int(win_unpart_node->ne[1] % w)) % w;
    const int npx = int(px + win_unpart_node->ne[1]) / w;
    const int64_t total = win_unpart_node->ne[0] * win_unpart_node->ne[1] * win_unpart_node->ne[2] *
                          win_unpart_node->ne[3];

    if (!ggml_cuda_win_part_f32_vec4_compatible(src0, add_node) || total % 4 != 0 ||
        reinterpret_cast<uintptr_t>(residual->data) % alignof(float4) != 0) {
        return false;
    }

    const int64_t vec_total = total / 4;
    const int64_t vec_blocks =
        (vec_total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;
    win_unpart_add_f32_vec4<<<vec_blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const char*) src0->data,
        (const float4*) residual->data,
        (float4*) add_node->data,
        src0->nb[1],
        src0->nb[2],
        src0->nb[3],
        win_unpart_node->ne[0] / 4,
        win_unpart_node->ne[1],
        win_unpart_node->ne[2],
        win_unpart_node->ne[3],
        npx,
        w);
    return true;
}
