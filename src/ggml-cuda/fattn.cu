#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

static __global__ void fattn_pad_head56_to64_f32(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = 64*ne1*ne2*ne3;
    if (i >= n) {
        return;
    }

    const int64_t d = i % 64;
    const int64_t t = i / 64;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    dst[i] = d < 56 ? src[d + i1*s1 + i2*s2 + i3*s3] : 0.0f;
}

static __global__ void fattn_pad_head56_to64_f16(
        const float * __restrict__ src,
        half        * __restrict__ dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = 64*ne1*ne2*ne3;
    if (i >= n) {
        return;
    }

    const int64_t d = i % 64;
    const int64_t t = i / 64;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    dst[i] = d < 56 ? __float2half(src[d + i1*s1 + i2*s2 + i3*s3]) : __float2half(0.0f);
}

template <int D, int DP>
static __global__ void fattn_pad_head_f32_q_kv_f16_same_shape(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t q_s1,
        const int64_t q_s2,
        const int64_t q_s3,
        const int64_t k_s1,
        const int64_t k_s2,
        const int64_t k_s3,
        const int64_t v_s1,
        const int64_t v_s2,
        const int64_t v_s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = int64_t(DP)*ne1*ne2*ne3;
    if (i >= n) {
        return;
    }

    const int64_t d = i % DP;
    const int64_t t = i / DP;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    if (d < D) {
        Q_dst[i] = Q[d + i1*q_s1 + i2*q_s2 + i3*q_s3];
        K_dst[i] = __float2half(K[d + i1*k_s1 + i2*k_s2 + i3*k_s3]);
        V_dst[i] = __float2half(V[d + i1*v_s1 + i2*v_s2 + i3*v_s3]);
    } else {
        Q_dst[i] = 0.0f;
        K_dst[i] = __float2half(0.0f);
        V_dst[i] = __float2half(0.0f);
    }
}

static __global__ void fattn_pad_head56_to64_q_f32_kv_f16(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t q_s1,
        const int64_t q_s2,
        const int64_t q_s3,
        const int64_t k_s1,
        const int64_t k_s2,
        const int64_t k_s3,
        const int64_t v_s1,
        const int64_t v_s2,
        const int64_t v_s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = 64*ne1*ne2*ne3;
    if (i >= n) {
        return;
    }

    const int64_t d = i % 64;
    const int64_t t = i / 64;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    if (d < 56) {
        Q_dst[i] = Q[d + i1*q_s1 + i2*q_s2 + i3*q_s3];
        K_dst[i] = __float2half(K[d + i1*k_s1 + i2*k_s2 + i3*k_s3]);
        V_dst[i] = __float2half(V[d + i1*v_s1 + i2*v_s2 + i3*v_s3]);
    } else {
        Q_dst[i] = 0.0f;
        K_dst[i] = __float2half(0.0f);
        V_dst[i] = __float2half(0.0f);
    }
}

static __global__ void fattn_pad_head56_to64_q_f32_kv_f16_row4(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t rows,
        const int64_t q_s1,
        const int64_t q_s2,
        const int64_t q_s3,
        const int64_t k_s1,
        const int64_t k_s2,
        const int64_t k_s3,
        const int64_t v_s1,
        const int64_t v_s2,
        const int64_t v_s3) {
    constexpr int rows_per_block = 8;
    const int lane = threadIdx.x & 15;
    const int64_t row = int64_t(blockIdx.x)*rows_per_block + (threadIdx.x >> 4);
    if (row >= rows || lane >= 15) {
        return;
    }

    const int64_t i1 = row % ne1;
    const int64_t t2 = row / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    const int64_t dst_base = row*64;
    if (lane == 14) {
        const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4 *>(Q_dst + dst_base + 56) = zero4;
        *reinterpret_cast<float4 *>(Q_dst + dst_base + 60) = zero4;
        *reinterpret_cast<half2 *>(K_dst + dst_base + 56) = make_half2(0.0f, 0.0f);
        *reinterpret_cast<half2 *>(K_dst + dst_base + 58) = make_half2(0.0f, 0.0f);
        *reinterpret_cast<half2 *>(K_dst + dst_base + 60) = make_half2(0.0f, 0.0f);
        *reinterpret_cast<half2 *>(K_dst + dst_base + 62) = make_half2(0.0f, 0.0f);
        *reinterpret_cast<half2 *>(V_dst + dst_base + 56) = make_half2(0.0f, 0.0f);
        *reinterpret_cast<half2 *>(V_dst + dst_base + 58) = make_half2(0.0f, 0.0f);
        *reinterpret_cast<half2 *>(V_dst + dst_base + 60) = make_half2(0.0f, 0.0f);
        *reinterpret_cast<half2 *>(V_dst + dst_base + 62) = make_half2(0.0f, 0.0f);
        return;
    }

    const int d = 4*lane;
    const int64_t q_base = i1*q_s1 + i2*q_s2 + i3*q_s3 + d;
    const int64_t k_base = i1*k_s1 + i2*k_s2 + i3*k_s3 + d;
    const int64_t v_base = i1*v_s1 + i2*v_s2 + i3*v_s3 + d;
    const float4 q4 = make_float4(Q[q_base + 0], Q[q_base + 1], Q[q_base + 2], Q[q_base + 3]);
    const float4 k4 = make_float4(K[k_base + 0], K[k_base + 1], K[k_base + 2], K[k_base + 3]);
    const float4 v4 = make_float4(V[v_base + 0], V[v_base + 1], V[v_base + 2], V[v_base + 3]);

    *reinterpret_cast<float4 *>(Q_dst + dst_base + d) = q4;
    *reinterpret_cast<half2 *>(K_dst + dst_base + d + 0) = __floats2half2_rn(k4.x, k4.y);
    *reinterpret_cast<half2 *>(K_dst + dst_base + d + 2) = __floats2half2_rn(k4.z, k4.w);
    *reinterpret_cast<half2 *>(V_dst + dst_base + d + 0) = __floats2half2_rn(v4.x, v4.y);
    *reinterpret_cast<half2 *>(V_dst + dst_base + d + 2) = __floats2half2_rn(v4.z, v4.w);
}

static __global__ void fattn_pad_head56_to64_qk_f32_f16_v56_f16(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t q_s1,
        const int64_t q_s2,
        const int64_t q_s3,
        const int64_t k_s1,
        const int64_t k_s2,
        const int64_t k_s3,
        const int64_t v_s1,
        const int64_t v_s2,
        const int64_t v_s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t qk_n = 64*ne1*ne2*ne3;
    if (i >= qk_n) {
        return;
    }

    const int64_t d = i % 64;
    const int64_t t = i / 64;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    if (d < 56) {
        Q_dst[i] = Q[d + i1*q_s1 + i2*q_s2 + i3*q_s3];
        K_dst[i] = __float2half(K[d + i1*k_s1 + i2*k_s2 + i3*k_s3]);
        V_dst[d + 56*t] = __float2half(V[d + i1*v_s1 + i2*v_s2 + i3*v_s3]);
    } else {
        Q_dst[i] = 0.0f;
        K_dst[i] = __float2half(0.0f);
    }
}

static __global__ void fattn_pad_head56_to64_q_f32_kv_f16_mixed(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t q_ne1,
        const int64_t q_ne2,
        const int64_t q_ne3,
        const int64_t k_ne1,
        const int64_t k_ne2,
        const int64_t k_ne3,
        const int64_t q_s1,
        const int64_t q_s2,
        const int64_t q_s3,
        const int64_t k_s1,
        const int64_t k_s2,
        const int64_t k_s3,
        const int64_t v_s1,
        const int64_t v_s2,
        const int64_t v_s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t q_n = 64*q_ne1*q_ne2*q_ne3;
    const int64_t k_n = 64*k_ne1*k_ne2*k_ne3;
    if (i >= q_n && i >= k_n) {
        return;
    }

    const int64_t d = i % 64;
    if (i < q_n) {
        const int64_t t = i / 64;
        const int64_t i1 = t % q_ne1;
        const int64_t t2 = t / q_ne1;
        const int64_t i2 = t2 % q_ne2;
        const int64_t i3 = t2 / q_ne2;
        Q_dst[i] = d < 56 ? Q[d + i1*q_s1 + i2*q_s2 + i3*q_s3] : 0.0f;
    }
    if (i < k_n) {
        const int64_t t = i / 64;
        const int64_t i1 = t % k_ne1;
        const int64_t t2 = t / k_ne1;
        const int64_t i2 = t2 % k_ne2;
        const int64_t i3 = t2 / k_ne2;
        if (d < 56) {
            K_dst[i] = __float2half(K[d + i1*k_s1 + i2*k_s2 + i3*k_s3]);
            V_dst[i] = __float2half(V[d + i1*v_s1 + i2*v_s2 + i3*v_s3]);
        } else {
            K_dst[i] = __float2half(0.0f);
            V_dst[i] = __float2half(0.0f);
        }
    }
}

static __global__ void fattn_pad_head56_to64_q_f32_kv_f16_mixed_row4(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t q_ne1,
        const int64_t q_ne2,
        const int64_t q_rows,
        const int64_t k_ne1,
        const int64_t k_ne2,
        const int64_t k_rows,
        const int64_t q_s1,
        const int64_t q_s2,
        const int64_t q_s3,
        const int64_t k_s1,
        const int64_t k_s2,
        const int64_t k_s3,
        const int64_t v_s1,
        const int64_t v_s2,
        const int64_t v_s3) {
    constexpr int rows_per_block = 8;
    const int lane = threadIdx.x & 15;
    const int64_t row = int64_t(blockIdx.x)*rows_per_block + (threadIdx.x >> 4);
    const int d = 4*lane;
    if (lane >= 15) {
        return;
    }

    if (row < q_rows) {
        const int64_t i1 = row % q_ne1;
        const int64_t t2 = row / q_ne1;
        const int64_t i2 = t2 % q_ne2;
        const int64_t i3 = t2 / q_ne2;
        const int64_t dst_base = row*64;
        if (lane == 14) {
            const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            *reinterpret_cast<float4 *>(Q_dst + dst_base + 56) = zero4;
            *reinterpret_cast<float4 *>(Q_dst + dst_base + 60) = zero4;
        } else {
            const int64_t q_base = i1*q_s1 + i2*q_s2 + i3*q_s3 + d;
            const float4 q4 = make_float4(Q[q_base + 0], Q[q_base + 1], Q[q_base + 2], Q[q_base + 3]);
            *reinterpret_cast<float4 *>(Q_dst + dst_base + d) = q4;
        }
    }

    if (row < k_rows) {
        const int64_t i1 = row % k_ne1;
        const int64_t t2 = row / k_ne1;
        const int64_t i2 = t2 % k_ne2;
        const int64_t i3 = t2 / k_ne2;
        const int64_t dst_base = row*64;
        if (lane == 14) {
            *reinterpret_cast<half2 *>(K_dst + dst_base + 56) = make_half2(0.0f, 0.0f);
            *reinterpret_cast<half2 *>(K_dst + dst_base + 58) = make_half2(0.0f, 0.0f);
            *reinterpret_cast<half2 *>(K_dst + dst_base + 60) = make_half2(0.0f, 0.0f);
            *reinterpret_cast<half2 *>(K_dst + dst_base + 62) = make_half2(0.0f, 0.0f);
            *reinterpret_cast<half2 *>(V_dst + dst_base + 56) = make_half2(0.0f, 0.0f);
            *reinterpret_cast<half2 *>(V_dst + dst_base + 58) = make_half2(0.0f, 0.0f);
            *reinterpret_cast<half2 *>(V_dst + dst_base + 60) = make_half2(0.0f, 0.0f);
            *reinterpret_cast<half2 *>(V_dst + dst_base + 62) = make_half2(0.0f, 0.0f);
        } else {
            const int64_t k_base = i1*k_s1 + i2*k_s2 + i3*k_s3 + d;
            const int64_t v_base = i1*v_s1 + i2*v_s2 + i3*v_s3 + d;
            const float4 k4 = make_float4(K[k_base + 0], K[k_base + 1], K[k_base + 2], K[k_base + 3]);
            const float4 v4 = make_float4(V[v_base + 0], V[v_base + 1], V[v_base + 2], V[v_base + 3]);
            *reinterpret_cast<half2 *>(K_dst + dst_base + d + 0) = __floats2half2_rn(k4.x, k4.y);
            *reinterpret_cast<half2 *>(K_dst + dst_base + d + 2) = __floats2half2_rn(k4.z, k4.w);
            *reinterpret_cast<half2 *>(V_dst + dst_base + d + 0) = __floats2half2_rn(v4.x, v4.y);
            *reinterpret_cast<half2 *>(V_dst + dst_base + d + 2) = __floats2half2_rn(v4.z, v4.w);
        }
    }
}

static __global__ void fattn_pad_head56_to64_qk_f32_f16_v56_f16_mixed(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t q_ne1,
        const int64_t q_ne2,
        const int64_t q_ne3,
        const int64_t k_ne1,
        const int64_t k_ne2,
        const int64_t k_ne3,
        const int64_t q_s1,
        const int64_t q_s2,
        const int64_t q_s3,
        const int64_t k_s1,
        const int64_t k_s2,
        const int64_t k_s3,
        const int64_t v_s1,
        const int64_t v_s2,
        const int64_t v_s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t q_n = 64*q_ne1*q_ne2*q_ne3;
    const int64_t k_n = 64*k_ne1*k_ne2*k_ne3;
    if (i >= q_n && i >= k_n) {
        return;
    }

    const int64_t d = i % 64;
    if (i < q_n) {
        const int64_t t = i / 64;
        const int64_t i1 = t % q_ne1;
        const int64_t t2 = t / q_ne1;
        const int64_t i2 = t2 % q_ne2;
        const int64_t i3 = t2 / q_ne2;
        Q_dst[i] = d < 56 ? Q[d + i1*q_s1 + i2*q_s2 + i3*q_s3] : 0.0f;
    }
    if (i < k_n) {
        const int64_t t = i / 64;
        const int64_t i1 = t % k_ne1;
        const int64_t t2 = t / k_ne1;
        const int64_t i2 = t2 % k_ne2;
        const int64_t i3 = t2 / k_ne2;
        if (d < 56) {
            K_dst[i] = __float2half(K[d + i1*k_s1 + i2*k_s2 + i3*k_s3]);
            V_dst[d + 56*t] = __float2half(V[d + i1*v_s1 + i2*v_s2 + i3*v_s3]);
        } else {
            K_dst[i] = __float2half(0.0f);
        }
    }
}

static __global__ void fattn_pad_head56_to64_q_f32_kv_f16_contiguous(
        const float * __restrict__ Q,
        const float * __restrict__ K,
        const float * __restrict__ V,
        float       * __restrict__ Q_dst,
        half        * __restrict__ K_dst,
        half        * __restrict__ V_dst,
        const int64_t n) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int64_t d = i & 63;
    const int64_t t = i >> 6;
    if (d < 56) {
        const int64_t src_i = d + 56*t;
        Q_dst[i] = Q[src_i];
        K_dst[i] = __float2half(K[src_i]);
        V_dst[i] = __float2half(V[src_i]);
    } else {
        Q_dst[i] = 0.0f;
        K_dst[i] = __float2half(0.0f);
        V_dst[i] = __float2half(0.0f);
    }
}

static __global__ void fattn_copy_head56_to_f16(
        const float * __restrict__ src,
        half        * __restrict__ dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = 56*ne1*ne2*ne3;
    if (i >= n) {
        return;
    }

    const int64_t d = i % 56;
    const int64_t t = i / 56;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    dst[i] = __float2half(src[d + i1*s1 + i2*s2 + i3*s3]);
}

static __global__ void fattn_slice_head64_to56_f32(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = 56*ne1*ne2*ne3;
    if (i >= n) {
        return;
    }

    const int64_t d = i % 56;
    const int64_t t = i / 56;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    dst[d + i1*s1 + i2*s2 + i3*s3] = src[d + 64*(i1 + ne1*(i2 + ne2*i3))];
}

static __global__ void fattn_slice_head64_to56_f32_contiguous(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int64_t n) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int64_t d = i % 56;
    const int64_t t = i / 56;
    dst[i] = src[d + 64*t];
}

template <int D, int DP>
static __global__ void fattn_slice_head_padded_to_f32(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = int64_t(D)*ne1*ne2*ne3;
    if (i >= n) {
        return;
    }

    const int64_t d = i % D;
    const int64_t t = i / D;
    const int64_t i1 = t % ne1;
    const int64_t t2 = t / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    dst[d + i1*s1 + i2*s2 + i3*s3] = src[d + DP*(i1 + ne1*(i2 + ne2*i3))];
}

static void ggml_cuda_flash_attn_ext_head56_pad_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
static void ggml_cuda_flash_attn_ext_head72_pad_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int DKQ, int DV, int ncols2, int DV_DST = DV, int D_SRC = DKQ, bool no_mask_no_sinks = false>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
}

template <int DKQ, int DV, int DV_DST = DV, int D_SRC = DKQ, bool no_mask_no_sinks = false>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1, DV_DST, D_SRC, no_mask_no_sinks>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            if (mask == nullptr && KQV->src[4] == nullptr && getenv("GGML_CUDA_DISABLE_FATTN256_NOMASK_FAST") == nullptr) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256, 256, 256, true>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            }
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void fattn_make_contiguous_f32_head64_tensor(ggml_tensor & tensor, float * data) {
    tensor.type = GGML_TYPE_F32;
    tensor.data = data;
    tensor.ne[0] = 64;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = 64*sizeof(float);
    tensor.nb[2] = tensor.ne[1]*tensor.nb[1];
    tensor.nb[3] = tensor.ne[2]*tensor.nb[2];
    tensor.view_src = nullptr;
    tensor.view_offs = 0;
}

static void fattn_make_contiguous_f16_head64_tensor(ggml_tensor & tensor, half * data) {
    tensor.type = GGML_TYPE_F16;
    tensor.data = data;
    tensor.ne[0] = 64;
    tensor.nb[0] = sizeof(half);
    tensor.nb[1] = 64*sizeof(half);
    tensor.nb[2] = tensor.ne[1]*tensor.nb[1];
    tensor.nb[3] = tensor.ne[2]*tensor.nb[2];
    tensor.view_src = nullptr;
    tensor.view_offs = 0;
}

static void fattn_make_contiguous_f16_head56_tensor(ggml_tensor & tensor, half * data) {
    tensor.type = GGML_TYPE_F16;
    tensor.data = data;
    tensor.ne[0] = 56;
    tensor.nb[0] = sizeof(half);
    tensor.nb[1] = 56*sizeof(half);
    tensor.nb[2] = tensor.ne[1]*tensor.nb[1];
    tensor.nb[3] = tensor.ne[2]*tensor.nb[2];
    tensor.view_src = nullptr;
    tensor.view_offs = 0;
}

static void fattn_make_contiguous_f32_head80_tensor(ggml_tensor & tensor, float * data) {
    tensor.type = GGML_TYPE_F32;
    tensor.data = data;
    tensor.ne[0] = 80;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = 80*sizeof(float);
    tensor.nb[2] = tensor.ne[1]*tensor.nb[1];
    tensor.nb[3] = tensor.ne[2]*tensor.nb[2];
    tensor.view_src = nullptr;
    tensor.view_offs = 0;
}

static void fattn_make_contiguous_f16_head80_tensor(ggml_tensor & tensor, half * data) {
    tensor.type = GGML_TYPE_F16;
    tensor.data = data;
    tensor.ne[0] = 80;
    tensor.nb[0] = sizeof(half);
    tensor.nb[1] = 80*sizeof(half);
    tensor.nb[2] = tensor.ne[1]*tensor.nb[1];
    tensor.nb[3] = tensor.ne[2]*tensor.nb[2];
    tensor.view_src = nullptr;
    tensor.view_offs = 0;
}

static void ggml_cuda_flash_attn_ext_head72_pad_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K->type == GGML_TYPE_F32);
    GGML_ASSERT(V->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == 72);
    GGML_ASSERT(K->ne[0] == 72);
    GGML_ASSERT(V->ne[0] == 72);
    GGML_ASSERT(dst->ne[0] == 72);
    GGML_ASSERT(dst->src[3] == nullptr);
    GGML_ASSERT(dst->src[4] == nullptr);
    GGML_ASSERT(Q->ne[1] == K->ne[1] && Q->ne[1] == V->ne[1]);
    GGML_ASSERT(Q->ne[2] == K->ne[2] && Q->ne[2] == V->ne[2]);
    GGML_ASSERT(Q->ne[3] == K->ne[3] && Q->ne[3] == V->ne[3]);

    const bool direct_head72_output = getenv("GGML_CUDA_DISABLE_FATTN72_DIRECT_OUT") == nullptr && ggml_is_contiguous(dst);
    cudaStream_t stream = ctx.stream();
    const bool profile_fattn72 = getenv("GGML_CUDA_PROFILE_FATTN72") != nullptr;
    if (direct_head72_output && getenv("GGML_CUDA_DISABLE_FATTN72_INLINE_PACK") == nullptr) {
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_after_mma = nullptr;
        if (profile_fattn72) {
            CUDA_CHECK(cudaEventCreate(&profile_start));
            CUDA_CHECK(cudaEventCreate(&profile_after_mma));
            CUDA_CHECK(cudaEventRecord(profile_start, stream));
        }

        ggml_tensor dst80 = *dst;
        dst80.data = dst->data;
        dst80.ne[0] = 72;
        dst80.nb[0] = sizeof(float);
        dst80.nb[1] = 72*sizeof(float);
        dst80.nb[2] = dst80.ne[1]*dst80.nb[1];
        dst80.nb[3] = dst80.ne[2]*dst80.nb[2];
        dst80.view_src = nullptr;
        dst80.view_offs = 0;
        dst80.src[0] = Q;
        dst80.src[1] = K;
        dst80.src[2] = V;
        dst80.src[3] = nullptr;
        dst80.src[4] = nullptr;
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<80, 80, 72, 72>(ctx, &dst80);

        if (profile_fattn72) {
            CUDA_CHECK(cudaEventRecord(profile_after_mma, stream));
            CUDA_CHECK(cudaEventSynchronize(profile_after_mma));

            float mma_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&mma_ms, profile_start, profile_after_mma));
            fprintf(stderr,
                    "GGML_CUDA_PROFILE_FATTN72 pack_ms=0.000000 mma_ms=%.6f slice_ms=0.000000 total_ms=%.6f "
                    "Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld] "
                    "dst=[%lld,%lld,%lld,%lld] inline_pack=1\n",
                    mma_ms,
                    mma_ms,
                    (long long) Q->ne[0],
                    (long long) Q->ne[1],
                    (long long) Q->ne[2],
                    (long long) Q->ne[3],
                    (long long) K->ne[0],
                    (long long) K->ne[1],
                    (long long) K->ne[2],
                    (long long) K->ne[3],
                    (long long) V->ne[0],
                    (long long) V->ne[1],
                    (long long) V->ne[2],
                    (long long) V->ne[3],
                    (long long) dst->ne[0],
                    (long long) dst->ne[1],
                    (long long) dst->ne[2],
                    (long long) dst->ne[3]);
            CUDA_CHECK(cudaEventDestroy(profile_start));
            CUDA_CHECK(cudaEventDestroy(profile_after_mma));
        }
        return;
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> Q_pad(pool, 80*Q->ne[1]*Q->ne[2]*Q->ne[3]);
    ggml_cuda_pool_alloc<half>  K_pad(pool, 80*K->ne[1]*K->ne[2]*K->ne[3]);
    ggml_cuda_pool_alloc<half>  V_pad(pool, 80*V->ne[1]*V->ne[2]*V->ne[3]);
    ggml_cuda_pool_alloc<float> dst_pad(pool);
    if (!direct_head72_output) {
        dst_pad.alloc(80*dst->ne[1]*dst->ne[2]*dst->ne[3]);
    }

    constexpr int block_size = 256;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_after_pack = nullptr;
    cudaEvent_t profile_after_mma = nullptr;
    cudaEvent_t profile_after_slice = nullptr;
    if (profile_fattn72) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_after_pack));
        CUDA_CHECK(cudaEventCreate(&profile_after_mma));
        CUDA_CHECK(cudaEventCreate(&profile_after_slice));
        CUDA_CHECK(cudaEventRecord(profile_start, stream));
    }

    const int64_t n = 80*Q->ne[1]*Q->ne[2]*Q->ne[3];
    const int grid_size = (n + block_size - 1) / block_size;
    fattn_pad_head_f32_q_kv_f16_same_shape<72, 80><<<grid_size, block_size, 0, stream>>>(
            (const float *) Q->data,
            (const float *) K->data,
            (const float *) V->data,
            Q_pad.ptr,
            K_pad.ptr,
            V_pad.ptr,
            Q->ne[1],
            Q->ne[2],
            Q->ne[3],
            Q->nb[1] / sizeof(float),
            Q->nb[2] / sizeof(float),
            Q->nb[3] / sizeof(float),
            K->nb[1] / sizeof(float),
            K->nb[2] / sizeof(float),
            K->nb[3] / sizeof(float),
            V->nb[1] / sizeof(float),
            V->nb[2] / sizeof(float),
            V->nb[3] / sizeof(float));
    if (profile_fattn72) {
        CUDA_CHECK(cudaEventRecord(profile_after_pack, stream));
    }

    ggml_tensor Q80 = *Q;
    ggml_tensor K80 = *K;
    ggml_tensor V80 = *V;
    ggml_tensor dst80 = *dst;
    fattn_make_contiguous_f32_head80_tensor(Q80, Q_pad.ptr);
    fattn_make_contiguous_f16_head80_tensor(K80, K_pad.ptr);
    fattn_make_contiguous_f16_head80_tensor(V80, V_pad.ptr);
    if (direct_head72_output) {
        dst80.data = dst->data;
        dst80.ne[0] = 72;
        dst80.nb[0] = sizeof(float);
        dst80.nb[1] = 72*sizeof(float);
        dst80.nb[2] = dst80.ne[1]*dst80.nb[1];
        dst80.nb[3] = dst80.ne[2]*dst80.nb[2];
        dst80.view_src = nullptr;
        dst80.view_offs = 0;
    } else {
        fattn_make_contiguous_f32_head80_tensor(dst80, dst_pad.ptr);
    }
    dst80.src[0] = &Q80;
    dst80.src[1] = &K80;
    dst80.src[2] = &V80;
    dst80.src[3] = nullptr;
    dst80.src[4] = nullptr;

    if (direct_head72_output) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<80, 80, 72>(ctx, &dst80);
    } else {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<80, 80>(ctx, &dst80);
    }
    if (profile_fattn72) {
        CUDA_CHECK(cudaEventRecord(profile_after_mma, stream));
    }

    if (!direct_head72_output) {
        const int64_t dst_n = 72*dst->ne[1]*dst->ne[2]*dst->ne[3];
        const int slice_grid_size = (dst_n + block_size - 1) / block_size;
        fattn_slice_head_padded_to_f32<72, 80><<<slice_grid_size, block_size, 0, stream>>>(
                dst_pad.ptr,
                (float *) dst->data,
                dst->ne[1],
                dst->ne[2],
                dst->ne[3],
                dst->nb[1] / sizeof(float),
                dst->nb[2] / sizeof(float),
                dst->nb[3] / sizeof(float));
    }
    if (profile_fattn72) {
        CUDA_CHECK(cudaEventRecord(profile_after_slice, stream));
        CUDA_CHECK(cudaEventSynchronize(profile_after_slice));

        float pack_ms = 0.0f;
        float mma_ms = 0.0f;
        float slice_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pack_ms, profile_start, profile_after_pack));
        CUDA_CHECK(cudaEventElapsedTime(&mma_ms, profile_after_pack, profile_after_mma));
        CUDA_CHECK(cudaEventElapsedTime(&slice_ms, profile_after_mma, profile_after_slice));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, profile_start, profile_after_slice));
        fprintf(stderr,
                "GGML_CUDA_PROFILE_FATTN72 pack_ms=%.6f mma_ms=%.6f slice_ms=%.6f total_ms=%.6f "
                "Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld] "
                "dst=[%lld,%lld,%lld,%lld]\n",
                pack_ms,
                mma_ms,
                slice_ms,
                total_ms,
                (long long) Q->ne[0],
                (long long) Q->ne[1],
                (long long) Q->ne[2],
                (long long) Q->ne[3],
                (long long) K->ne[0],
                (long long) K->ne[1],
                (long long) K->ne[2],
                (long long) K->ne[3],
                (long long) V->ne[0],
                (long long) V->ne[1],
                (long long) V->ne[2],
                (long long) V->ne[3],
                (long long) dst->ne[0],
                (long long) dst->ne[1],
                (long long) dst->ne[2],
                (long long) dst->ne[3]);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_after_pack));
        CUDA_CHECK(cudaEventDestroy(profile_after_mma));
        CUDA_CHECK(cudaEventDestroy(profile_after_slice));
    }
}

static void ggml_cuda_flash_attn_ext_head56_pad_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K->type == GGML_TYPE_F32);
    GGML_ASSERT(V->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == 56);
    GGML_ASSERT(K->ne[0] == 56);
    GGML_ASSERT(V->ne[0] == 56);
    GGML_ASSERT(dst->ne[0] == 56);
    GGML_ASSERT(dst->src[3] == nullptr);
    GGML_ASSERT(dst->src[4] == nullptr);

    const bool direct_head56_output = getenv("GGML_CUDA_DISABLE_FATTN56_DIRECT_OUT") == nullptr && ggml_is_contiguous(dst);
    const bool native_v56_mma = direct_head56_output && getenv("GGML_CUDA_ENABLE_FATTN56_NATIVE_V") != nullptr;
    const bool same_qkv_shape = Q->ne[1] == K->ne[1] && Q->ne[1] == V->ne[1] &&
                                Q->ne[2] == K->ne[2] && Q->ne[2] == V->ne[2] &&
                                Q->ne[3] == K->ne[3] && Q->ne[3] == V->ne[3];
    const bool large_single_sequence_kv = K->ne[1] >= 1024 && K->ne[3] == 1;
    const bool inline_pack_large = getenv("GGML_CUDA_ENABLE_FATTN56_INLINE_PACK_LARGE") != nullptr;
    const bool no_mask_no_sinks56 = getenv("GGML_CUDA_DISABLE_FATTN56_NOMASK_FAST") == nullptr;
    cudaStream_t stream = ctx.stream();
    const bool profile_fattn56 = getenv("GGML_CUDA_PROFILE_FATTN56") != nullptr;
    if (direct_head56_output && !native_v56_mma && (!large_single_sequence_kv || inline_pack_large) &&
        getenv("GGML_CUDA_DISABLE_FATTN56_INLINE_PACK") == nullptr) {
        cudaEvent_t profile_start = nullptr;
        cudaEvent_t profile_after_mma = nullptr;
        if (profile_fattn56) {
            CUDA_CHECK(cudaEventCreate(&profile_start));
            CUDA_CHECK(cudaEventCreate(&profile_after_mma));
            CUDA_CHECK(cudaEventRecord(profile_start, stream));
        }

        ggml_tensor dst64 = *dst;
        fattn_make_contiguous_f32_head64_tensor(dst64, (float *) dst->data);
        dst64.src[0] = Q;
        dst64.src[1] = K;
        dst64.src[2] = V;
        dst64.src[3] = nullptr;
        dst64.src[4] = nullptr;
        if (no_mask_no_sinks56) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 64, 56, 56, true>(ctx, &dst64);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 64, 56, 56>(ctx, &dst64);
        }

        if (profile_fattn56) {
            CUDA_CHECK(cudaEventRecord(profile_after_mma, stream));
            CUDA_CHECK(cudaEventSynchronize(profile_after_mma));

            float mma_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&mma_ms, profile_start, profile_after_mma));
            fprintf(stderr,
                    "GGML_CUDA_PROFILE_FATTN56 pack_ms=0.000000 mma_ms=%.6f slice_ms=0.000000 total_ms=%.6f "
                    "Q=[%lld,%lld,%lld,%lld] q_nb=[%lld,%lld,%lld,%lld] "
                    "K=[%lld,%lld,%lld,%lld] k_nb=[%lld,%lld,%lld,%lld] "
                    "V=[%lld,%lld,%lld,%lld] v_nb=[%lld,%lld,%lld,%lld] "
                    "qkv_contiguous=%d dst_contiguous=%d combined=%d\n",
                    mma_ms,
                    mma_ms,
                    (long long) Q->ne[0],
                    (long long) Q->ne[1],
                    (long long) Q->ne[2],
                    (long long) Q->ne[3],
                    (long long) Q->nb[0],
                    (long long) Q->nb[1],
                    (long long) Q->nb[2],
                    (long long) Q->nb[3],
                    (long long) K->ne[0],
                    (long long) K->ne[1],
                    (long long) K->ne[2],
                    (long long) K->ne[3],
                    (long long) K->nb[0],
                    (long long) K->nb[1],
                    (long long) K->nb[2],
                    (long long) K->nb[3],
                    (long long) V->ne[0],
                    (long long) V->ne[1],
                    (long long) V->ne[2],
                    (long long) V->ne[3],
                    (long long) V->nb[0],
                    (long long) V->nb[1],
                    (long long) V->nb[2],
                    (long long) V->nb[3],
                    ggml_is_contiguous(Q) && ggml_is_contiguous(K) && ggml_is_contiguous(V),
                    ggml_is_contiguous(dst),
                    same_qkv_shape);
            CUDA_CHECK(cudaEventDestroy(profile_start));
            CUDA_CHECK(cudaEventDestroy(profile_after_mma));
        }
        return;
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> Q_pad(pool, 64*Q->ne[1]*Q->ne[2]*Q->ne[3]);
    ggml_cuda_pool_alloc<half>  K_pad(pool, 64*K->ne[1]*K->ne[2]*K->ne[3]);
    ggml_cuda_pool_alloc<half>  V_pad(pool, (native_v56_mma ? 56 : 64)*V->ne[1]*V->ne[2]*V->ne[3]);
    ggml_cuda_pool_alloc<float> dst_pad(pool);
    if (!direct_head56_output) {
        dst_pad.alloc(64*dst->ne[1]*dst->ne[2]*dst->ne[3]);
    }

    constexpr int block_size = 256;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_after_pack = nullptr;
    cudaEvent_t profile_after_mma = nullptr;
    cudaEvent_t profile_after_slice = nullptr;
    if (profile_fattn56) {
        CUDA_CHECK(cudaEventCreate(&profile_start));
        CUDA_CHECK(cudaEventCreate(&profile_after_pack));
        CUDA_CHECK(cudaEventCreate(&profile_after_mma));
        CUDA_CHECK(cudaEventCreate(&profile_after_slice));
        CUDA_CHECK(cudaEventRecord(profile_start, stream));
    }

    auto launch_pad_f32 = [&](const ggml_tensor * src, float * tmp) {
        const int64_t n = 64*src->ne[1]*src->ne[2]*src->ne[3];
        const int grid_size = (n + block_size - 1) / block_size;
        fattn_pad_head56_to64_f32<<<grid_size, block_size, 0, stream>>>(
                (const float *) src->data, tmp,
                src->ne[1], src->ne[2], src->ne[3],
                src->nb[1] / sizeof(float),
                src->nb[2] / sizeof(float),
                src->nb[3] / sizeof(float));
    };
    auto launch_pad_f16 = [&](const ggml_tensor * src, half * tmp) {
        const int64_t n = 64*src->ne[1]*src->ne[2]*src->ne[3];
        const int grid_size = (n + block_size - 1) / block_size;
        fattn_pad_head56_to64_f16<<<grid_size, block_size, 0, stream>>>(
                (const float *) src->data, tmp,
                src->ne[1], src->ne[2], src->ne[3],
                src->nb[1] / sizeof(float),
                src->nb[2] / sizeof(float),
                src->nb[3] / sizeof(float));
    };
    auto launch_copy_v56_f16 = [&](const ggml_tensor * src, half * tmp) {
        const int64_t n = 56*src->ne[1]*src->ne[2]*src->ne[3];
        const int grid_size = (n + block_size - 1) / block_size;
        fattn_copy_head56_to_f16<<<grid_size, block_size, 0, stream>>>(
                (const float *) src->data, tmp,
                src->ne[1], src->ne[2], src->ne[3],
                src->nb[1] / sizeof(float),
                src->nb[2] / sizeof(float),
                src->nb[3] / sizeof(float));
    };
    if (same_qkv_shape && getenv("GGML_CUDA_DISABLE_FATTN56_COMBINED_PACK") == nullptr) {
        const int64_t n = 64*Q->ne[1]*Q->ne[2]*Q->ne[3];
        const int grid_size = (n + block_size - 1) / block_size;
        const bool use_contiguous_pack =
            getenv("GGML_CUDA_DISABLE_FATTN56_CONTIGUOUS_PACK") == nullptr &&
            ggml_is_contiguous(Q) && ggml_is_contiguous(K) && ggml_is_contiguous(V);
        if (native_v56_mma) {
            fattn_pad_head56_to64_qk_f32_f16_v56_f16<<<grid_size, block_size, 0, stream>>>(
                    (const float *) Q->data,
                    (const float *) K->data,
                    (const float *) V->data,
                    Q_pad.ptr, K_pad.ptr, V_pad.ptr,
                    Q->ne[1], Q->ne[2], Q->ne[3],
                    Q->nb[1] / sizeof(float),
                    Q->nb[2] / sizeof(float),
                    Q->nb[3] / sizeof(float),
                    K->nb[1] / sizeof(float),
                    K->nb[2] / sizeof(float),
                    K->nb[3] / sizeof(float),
                    V->nb[1] / sizeof(float),
                    V->nb[2] / sizeof(float),
                    V->nb[3] / sizeof(float));
        } else if (use_contiguous_pack) {
            fattn_pad_head56_to64_q_f32_kv_f16_contiguous<<<grid_size, block_size, 0, stream>>>(
                    (const float *) Q->data,
                    (const float *) K->data,
                    (const float *) V->data,
                    Q_pad.ptr, K_pad.ptr, V_pad.ptr,
                    n);
        } else if (getenv("GGML_CUDA_DISABLE_FATTN56_ROW4_PACK") == nullptr) {
            constexpr int row4_block_size = 128;
            constexpr int rows_per_block = 8;
            const int64_t rows = Q->ne[1]*Q->ne[2]*Q->ne[3];
            const int row4_grid_size = (rows + rows_per_block - 1) / rows_per_block;
            fattn_pad_head56_to64_q_f32_kv_f16_row4<<<row4_grid_size, row4_block_size, 0, stream>>>(
                    (const float *) Q->data,
                    (const float *) K->data,
                    (const float *) V->data,
                    Q_pad.ptr, K_pad.ptr, V_pad.ptr,
                    Q->ne[1], Q->ne[2], rows,
                    Q->nb[1] / sizeof(float),
                    Q->nb[2] / sizeof(float),
                    Q->nb[3] / sizeof(float),
                    K->nb[1] / sizeof(float),
                    K->nb[2] / sizeof(float),
                    K->nb[3] / sizeof(float),
                    V->nb[1] / sizeof(float),
                    V->nb[2] / sizeof(float),
                    V->nb[3] / sizeof(float));
        } else {
            fattn_pad_head56_to64_q_f32_kv_f16<<<grid_size, block_size, 0, stream>>>(
                    (const float *) Q->data,
                    (const float *) K->data,
                    (const float *) V->data,
                    Q_pad.ptr, K_pad.ptr, V_pad.ptr,
                    Q->ne[1], Q->ne[2], Q->ne[3],
                    Q->nb[1] / sizeof(float),
                    Q->nb[2] / sizeof(float),
                    Q->nb[3] / sizeof(float),
                    K->nb[1] / sizeof(float),
                    K->nb[2] / sizeof(float),
                    K->nb[3] / sizeof(float),
                    V->nb[1] / sizeof(float),
                    V->nb[2] / sizeof(float),
                    V->nb[3] / sizeof(float));
        }
    } else {
        const bool can_use_mixed_pack =
            getenv("GGML_CUDA_DISABLE_FATTN56_MIXED_PACK") == nullptr &&
            K->ne[1] == V->ne[1] && K->ne[2] == V->ne[2] && K->ne[3] == V->ne[3] &&
            Q->ne[2] == K->ne[2] && Q->ne[3] == K->ne[3];
        if (can_use_mixed_pack) {
            const int64_t q_n = 64*Q->ne[1]*Q->ne[2]*Q->ne[3];
            const int64_t k_n = 64*K->ne[1]*K->ne[2]*K->ne[3];
            const int64_t n = q_n > k_n ? q_n : k_n;
            const int grid_size = (n + block_size - 1) / block_size;
            if (native_v56_mma) {
                fattn_pad_head56_to64_qk_f32_f16_v56_f16_mixed<<<grid_size, block_size, 0, stream>>>(
                        (const float *) Q->data,
                        (const float *) K->data,
                        (const float *) V->data,
                        Q_pad.ptr, K_pad.ptr, V_pad.ptr,
                        Q->ne[1], Q->ne[2], Q->ne[3],
                        K->ne[1], K->ne[2], K->ne[3],
                        Q->nb[1] / sizeof(float),
                        Q->nb[2] / sizeof(float),
                        Q->nb[3] / sizeof(float),
                        K->nb[1] / sizeof(float),
                        K->nb[2] / sizeof(float),
                        K->nb[3] / sizeof(float),
                        V->nb[1] / sizeof(float),
                        V->nb[2] / sizeof(float),
                        V->nb[3] / sizeof(float));
            } else {
                if (getenv("GGML_CUDA_DISABLE_FATTN56_MIXED_ROW4_PACK") == nullptr) {
                    constexpr int row4_block_size = 128;
                    constexpr int rows_per_block = 8;
                    const int64_t q_rows = Q->ne[1]*Q->ne[2]*Q->ne[3];
                    const int64_t k_rows = K->ne[1]*K->ne[2]*K->ne[3];
                    const int64_t rows = q_rows > k_rows ? q_rows : k_rows;
                    const int row4_grid_size = (rows + rows_per_block - 1) / rows_per_block;
                    fattn_pad_head56_to64_q_f32_kv_f16_mixed_row4<<<row4_grid_size, row4_block_size, 0, stream>>>(
                            (const float *) Q->data,
                            (const float *) K->data,
                            (const float *) V->data,
                            Q_pad.ptr, K_pad.ptr, V_pad.ptr,
                            Q->ne[1], Q->ne[2], q_rows,
                            K->ne[1], K->ne[2], k_rows,
                            Q->nb[1] / sizeof(float),
                            Q->nb[2] / sizeof(float),
                            Q->nb[3] / sizeof(float),
                            K->nb[1] / sizeof(float),
                            K->nb[2] / sizeof(float),
                            K->nb[3] / sizeof(float),
                            V->nb[1] / sizeof(float),
                            V->nb[2] / sizeof(float),
                            V->nb[3] / sizeof(float));
                } else {
                    fattn_pad_head56_to64_q_f32_kv_f16_mixed<<<grid_size, block_size, 0, stream>>>(
                            (const float *) Q->data,
                            (const float *) K->data,
                            (const float *) V->data,
                            Q_pad.ptr, K_pad.ptr, V_pad.ptr,
                            Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[1], K->ne[2], K->ne[3],
                            Q->nb[1] / sizeof(float),
                            Q->nb[2] / sizeof(float),
                            Q->nb[3] / sizeof(float),
                            K->nb[1] / sizeof(float),
                            K->nb[2] / sizeof(float),
                            K->nb[3] / sizeof(float),
                            V->nb[1] / sizeof(float),
                            V->nb[2] / sizeof(float),
                            V->nb[3] / sizeof(float));
                }
            }
        } else {
            launch_pad_f32(Q, Q_pad.ptr);
            launch_pad_f16(K, K_pad.ptr);
            if (native_v56_mma) {
                launch_copy_v56_f16(V, V_pad.ptr);
            } else {
                launch_pad_f16(V, V_pad.ptr);
            }
        }
    }
    if (profile_fattn56) {
        CUDA_CHECK(cudaEventRecord(profile_after_pack, stream));
    }

    ggml_tensor Q64 = *Q;
    ggml_tensor K64 = *K;
    ggml_tensor V64 = *V;
    ggml_tensor dst64 = *dst;
    fattn_make_contiguous_f32_head64_tensor(Q64, Q_pad.ptr);
    fattn_make_contiguous_f16_head64_tensor(K64, K_pad.ptr);
    if (native_v56_mma) {
        fattn_make_contiguous_f16_head56_tensor(V64, V_pad.ptr);
    } else {
        fattn_make_contiguous_f16_head64_tensor(V64, V_pad.ptr);
    }
    fattn_make_contiguous_f32_head64_tensor(dst64, direct_head56_output ? (float *) dst->data : dst_pad.ptr);
    dst64.src[0] = &Q64;
    dst64.src[1] = &K64;
    dst64.src[2] = &V64;
    dst64.src[3] = nullptr;
    dst64.src[4] = nullptr;

    if (native_v56_mma) {
        if (no_mask_no_sinks56) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 56, 56, 64, true>(ctx, &dst64);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 56>(ctx, &dst64);
        }
    } else if (direct_head56_output) {
        if (no_mask_no_sinks56) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 64, 56, 64, true>(ctx, &dst64);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 64, 56>(ctx, &dst64);
        }
    } else {
        if (no_mask_no_sinks56) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 64, 64, 64, true>(ctx, &dst64);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 64>(ctx, &dst64);
        }
    }
    if (profile_fattn56) {
        CUDA_CHECK(cudaEventRecord(profile_after_mma, stream));
    }

    if (!direct_head56_output) {
        const int64_t n = 56*dst->ne[1]*dst->ne[2]*dst->ne[3];
        const int grid_size = (n + block_size - 1) / block_size;
        if (getenv("GGML_CUDA_DISABLE_FATTN56_CONTIGUOUS_PACK") == nullptr && ggml_is_contiguous(dst)) {
            fattn_slice_head64_to56_f32_contiguous<<<grid_size, block_size, 0, stream>>>(
                    dst_pad.ptr, (float *) dst->data, n);
        } else {
            fattn_slice_head64_to56_f32<<<grid_size, block_size, 0, stream>>>(
                    dst_pad.ptr, (float *) dst->data,
                    dst->ne[1], dst->ne[2], dst->ne[3],
                    dst->nb[1] / sizeof(float),
                    dst->nb[2] / sizeof(float),
                    dst->nb[3] / sizeof(float));
        }
    }
    if (profile_fattn56) {
        CUDA_CHECK(cudaEventRecord(profile_after_slice, stream));
        CUDA_CHECK(cudaEventSynchronize(profile_after_slice));

        float pack_ms = 0.0f;
        float mma_ms = 0.0f;
        float slice_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pack_ms, profile_start, profile_after_pack));
        CUDA_CHECK(cudaEventElapsedTime(&mma_ms, profile_after_pack, profile_after_mma));
        CUDA_CHECK(cudaEventElapsedTime(&slice_ms, profile_after_mma, profile_after_slice));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, profile_start, profile_after_slice));
        fprintf(stderr,
                "GGML_CUDA_PROFILE_FATTN56 pack_ms=%.6f mma_ms=%.6f slice_ms=%.6f total_ms=%.6f "
                "Q=[%lld,%lld,%lld,%lld] q_nb=[%lld,%lld,%lld,%lld] "
                "K=[%lld,%lld,%lld,%lld] k_nb=[%lld,%lld,%lld,%lld] "
                "V=[%lld,%lld,%lld,%lld] v_nb=[%lld,%lld,%lld,%lld] "
                "qkv_contiguous=%d dst_contiguous=%d combined=%d\n",
                pack_ms,
                mma_ms,
                slice_ms,
                total_ms,
                (long long) Q->ne[0],
                (long long) Q->ne[1],
                (long long) Q->ne[2],
                (long long) Q->ne[3],
                (long long) Q->nb[0],
                (long long) Q->nb[1],
                (long long) Q->nb[2],
                (long long) Q->nb[3],
                (long long) K->ne[0],
                (long long) K->ne[1],
                (long long) K->ne[2],
                (long long) K->ne[3],
                (long long) K->nb[0],
                (long long) K->nb[1],
                (long long) K->nb[2],
                (long long) K->nb[3],
                (long long) V->ne[0],
                (long long) V->ne[1],
                (long long) V->ne[2],
                (long long) V->ne[3],
                (long long) V->nb[0],
                (long long) V->nb[1],
                (long long) V->nb[2],
                (long long) V->nb[3],
                ggml_is_contiguous(Q) && ggml_is_contiguous(K) && ggml_is_contiguous(V),
                ggml_is_contiguous(dst),
                same_qkv_shape && getenv("GGML_CUDA_DISABLE_FATTN56_COMBINED_PACK") == nullptr);
        CUDA_CHECK(cudaEventDestroy(profile_start));
        CUDA_CHECK(cudaEventDestroy(profile_after_pack));
        CUDA_CHECK(cudaEventDestroy(profile_after_mma));
        CUDA_CHECK(cudaEventDestroy(profile_after_slice));
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  16:
        case  32:
        case  40:
        case  56:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    if (Q->ne[0] == 16 || Q->ne[0] == 32) {
        return BEST_FATTN_KERNEL_TILE;
    }

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 56 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 56 && Q->ne[0] != 72) {
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 56 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 56 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 8) {
            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use MFMA flash attention for CDNA (MI100+):
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 56 && Q->ne[0] != 72 && Q->ne[0] != 256 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        const int64_t eff_nq = Q->ne[1] * (gqa_opt_applies ? gqa_ratio : 1);
        // MMA vs tile crossover benchmarked on MI300X @ d32768:
        //   hsk=64  (gqa=4): MMA wins at eff >= 128 (+11%)
        //   hsk=128 (gqa=4): MMA wins at eff >= 128 (+4%)
        if (eff_nq >= (GGML_CUDA_CC_IS_CDNA1(cc) && Q->ne[0] == 64 ? 64 : 128)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        // Fall through to tile kernel for small effective batch sizes.
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (Q->ne[0] == 56 && K->ne[0] == 56 && V->ne[0] == 56 &&
            mask == nullptr && sinks == nullptr &&
            Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && V->type == GGML_TYPE_F32 &&
            turing_mma_available(ggml_cuda_info().devices[ctx.device].cc) &&
            getenv("GGML_CUDA_DISABLE_FATTN56_PAD_MMA") == nullptr) {
        ggml_cuda_flash_attn_ext_head56_pad_mma_f16(ctx, dst);
        return;
    }
    if (Q->ne[0] == 72 && K->ne[0] == 72 && V->ne[0] == 72 &&
            Q->ne[1] == K->ne[1] && Q->ne[1] == V->ne[1] &&
            Q->ne[2] == K->ne[2] && Q->ne[2] == V->ne[2] &&
            Q->ne[3] == K->ne[3] && Q->ne[3] == V->ne[3] &&
            mask == nullptr && sinks == nullptr &&
            Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && V->type == GGML_TYPE_F32 &&
            turing_mma_available(ggml_cuda_info().devices[ctx.device].cc) &&
            getenv("GGML_CUDA_ENABLE_FATTN72_PAD_MMA") != nullptr) {
        ggml_cuda_flash_attn_ext_head72_pad_mma_f16(ctx, dst);
        return;
    }
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
#ifdef GGML_CUDA_FATTN_DIAG
            fprintf(stderr,
                    "ggml_cuda_flash_attn_ext: no CUDA kernel for Q[%lld,%lld,%lld,%lld] %s "
                    "K[%lld,%lld,%lld,%lld] %s V[%lld,%lld,%lld,%lld] %s mask=%s cc=%d highest=%d turing=%d\n",
                    (long long) dst->src[0]->ne[0], (long long) dst->src[0]->ne[1],
                    (long long) dst->src[0]->ne[2], (long long) dst->src[0]->ne[3],
                    ggml_type_name(dst->src[0]->type),
                    (long long) dst->src[1]->ne[0], (long long) dst->src[1]->ne[1],
                    (long long) dst->src[1]->ne[2], (long long) dst->src[1]->ne[3],
                    ggml_type_name(dst->src[1]->type),
                    (long long) dst->src[2]->ne[0], (long long) dst->src[2]->ne[1],
                    (long long) dst->src[2]->ne[2], (long long) dst->src[2]->ne[3],
                    ggml_type_name(dst->src[2]->type),
                    dst->src[3] ? "yes" : "no",
                    ggml_cuda_info().devices[ctx.device].cc,
                    ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx.device].cc),
                    turing_mma_available(ggml_cuda_info().devices[ctx.device].cc));
#endif
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
