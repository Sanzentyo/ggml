#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

static void ggml_cuda_debug_compare_mmq_q8_only(
        ggml_backend_cuda_context & ctx,
        const float *               dst_d,
        const char *                q8_only_d,
        const ggml_type             consumer_type,
        const ggml_tensor *         dst,
        const int64_t               ne0_padded,
        const size_t                nbytes_payload,
        const size_t                nbytes_storage,
        cudaStream_t                stream) {
    if (getenv("GGML_CUDA_DEBUG_MMQ_Q8_ONLY_COMPARE") == nullptr) {
        return;
    }

    ggml_cuda_pool_alloc<char> ref_q8(ctx.pool());
    ref_q8.alloc(nbytes_storage);
    const size_t ts_dst = ggml_type_size(dst->type);
    quantize_mmq_q8_1_cuda(dst_d, nullptr, ref_q8.get(), consumer_type, dst->ne[0],
                           dst->nb[1] / ts_dst, dst->nb[2] / ts_dst, dst->nb[3] / ts_dst,
                           ne0_padded, dst->ne[1], dst->ne[2], dst->ne[3], stream);
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint8_t> ref(nbytes_payload);
    std::vector<uint8_t> got(nbytes_payload);
    CUDA_CHECK(cudaMemcpyAsync(ref.data(), ref_q8.get(), nbytes_payload, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(got.data(), q8_only_d, nbytes_payload, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t mismatch = 0;
    size_t first = 0;
    size_t ds4_mismatch = 0;
    size_t ds4_d_mismatch = 0;
    size_t ds4_sum_mismatch = 0;
    size_t qs_mismatch = 0;
    size_t first_ds4 = 0;
    size_t first_qs = 0;
    int max_qs_abs_diff = 0;

    for (size_t i = 0; i < nbytes_payload; ++i) {
        if (ref[i] != got[i]) {
            if (mismatch == 0) {
                first = i;
            }
            ++mismatch;
        }
    }

    const auto * ref_blocks = reinterpret_cast<const block_q8_1_mmq *>(ref.data());
    const auto * got_blocks = reinterpret_cast<const block_q8_1_mmq *>(got.data());
    const size_t nblocks = nbytes_payload / sizeof(block_q8_1_mmq);
    for (size_t block = 0; block < nblocks; ++block) {
        const auto * ref_ds4 = reinterpret_cast<const uint8_t *>(ref_blocks[block].ds4);
        const auto * got_ds4 = reinterpret_cast<const uint8_t *>(got_blocks[block].ds4);
        for (size_t i = 0; i < sizeof(ref_blocks[block].ds4); ++i) {
            if (ref_ds4[i] != got_ds4[i]) {
                if (ds4_mismatch == 0) {
                    first_ds4 = block * sizeof(block_q8_1_mmq) + i;
                }
                ++ds4_mismatch;
                if ((i % sizeof(half2)) < sizeof(half)) {
                    ++ds4_d_mismatch;
                } else {
                    ++ds4_sum_mismatch;
                }
            }
        }
        for (size_t i = 0; i < sizeof(ref_blocks[block].qs); ++i) {
            if (ref_blocks[block].qs[i] != got_blocks[block].qs[i]) {
                if (qs_mismatch == 0) {
                    first_qs = block * sizeof(block_q8_1_mmq) + sizeof(ref_blocks[block].ds4) + i;
                }
                ++qs_mismatch;
                max_qs_abs_diff = std::max(max_qs_abs_diff,
                                           std::abs(static_cast<int>(ref_blocks[block].qs[i]) -
                                                    static_cast<int>(got_blocks[block].qs[i])));
            }
        }
    }

    fprintf(stderr,
            "GGML_CUDA_DEBUG_MMQ_Q8_ONLY_COMPARE name=%s consumer_type=%s dst_ne=%lld,%lld,%lld,%lld "
            "bytes=%zu mismatch=%zu first=%zu ds4_mismatch=%zu first_ds4=%zu "
            "ds4_d_mismatch=%zu ds4_sum_mismatch=%zu "
            "qs_mismatch=%zu first_qs=%zu max_qs_abs_diff=%d\n",
            dst->name,
            ggml_type_name(consumer_type),
            (long long) dst->ne[0],
            (long long) dst->ne[1],
            (long long) dst->ne[2],
            (long long) dst->ne[3],
            nbytes_payload,
            mismatch,
            first,
            ds4_mismatch,
            first_ds4,
            ds4_d_mismatch,
            ds4_sum_mismatch,
            qs_mismatch,
            first_qs,
            max_qs_abs_diff);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_mul_mat_q_run_prequantized(
        ggml_backend_cuda_context & ctx,
        const char *                src0_d,
        const ggml_type             src0_type,
        const int *                 src1_q8_1_ptr,
        float *                     dst_d,
        const float *               bias_d,
        const mmq_activation        activation,
        const int64_t               ne00,
        const int64_t               ne01,
        const int64_t               ne1,
        const int64_t               s01,
        const int64_t               ne11,
        const int64_t               s1,
        const int64_t               ne02,
        const int64_t               ne12,
        const int64_t               s02,
        const int64_t               s2,
        const int64_t               ne03,
        const int64_t               ne13,
        const int64_t               s03,
        const int64_t               s3,
        const int64_t               ne10_padded,
        const bool                  use_native_fp4,
        const bool                  use_stream_k,
        const bool                  q4_1_full_tile_fastpath,
        block_q8_1_mmq *            q8_only_dst,
        cudaStream_t                stream) {
    const int64_t s12 = use_native_fp4 ?
                            ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                            ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12 * s12;

    const mmq_args args = {
        src0_d, src0_type, src1_q8_1_ptr, nullptr, nullptr, dst_d, q8_only_dst, bias_d, activation,
        ne00, ne01, ne1, s01, ne11, s1,
        ne02, ne12, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, q4_1_full_tile_fastpath, ne1};
    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_tensor * bias, const mmq_activation activation) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    GGML_ASSERT(!bias || (!ids && bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias) && bias->ne[0] == dst->ne[0]));

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = bias ? (const float *) bias->data : nullptr;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k =
        getenv("GGML_CUDA_DISABLE_MMQ_STREAM_K") == nullptr &&
        ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc));
    const bool profile_mmq = getenv("GGML_CUDA_PROFILE_MMQ") != nullptr;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_after_quant = nullptr;
    cudaEvent_t profile_after_mmq = nullptr;

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const bool q4_1_full_tile_fastpath =
        src0->type == GGML_TYPE_Q4_1 &&
        (getenv("GGML_CUDA_DISABLE_MMQ_Q4_1_FULL_TILE_FASTPATH") == nullptr ||
         std::atoi(getenv("GGML_CUDA_DISABLE_MMQ_Q4_1_FULL_TILE_FASTPATH")) == 0);
    if (!ids) {
        const auto cached_src1 = std::find_if(ctx.mmq_prequant_cache.begin(), ctx.mmq_prequant_cache.end(),
                                              [src1](const auto & entry) { return entry.first == src1; });
        const bool use_cached_src1 =
            !use_native_fp4 &&
            cached_src1 != ctx.mmq_prequant_cache.end() &&
            cached_src1->second.consumer_type == src0->type &&
            cached_src1->second.ne0_padded == ne10_padded &&
            cached_src1->second.ne1 == ne11 &&
            cached_src1->second.ne2 == ne12 &&
            cached_src1->second.ne3 == ne13 &&
            cached_src1->second.data != nullptr;
        if (use_cached_src1) {
            if (profile_mmq) {
                CUDA_CHECK(cudaEventCreate(&profile_start));
                CUDA_CHECK(cudaEventCreate(&profile_after_quant));
                CUDA_CHECK(cudaEventCreate(&profile_after_mmq));
                CUDA_CHECK(cudaEventRecord(profile_start, stream));
                CUDA_CHECK(cudaEventRecord(profile_after_quant, stream));
            }

            ggml_cuda_mul_mat_q_run_prequantized(ctx, src0_d, src0->type, (const int *) cached_src1->second.data,
                                                 dst_d, bias_d, activation,
                                                 ne00, ne01, ne1, s01, ne11, s1,
                                                 ne02, ne12, s02, s2,
                                                 ne03, ne13, s03, s3,
                                                 ne10_padded, false, use_stream_k,
                                                 q4_1_full_tile_fastpath, nullptr, stream);

            if (profile_mmq) {
                CUDA_CHECK(cudaEventRecord(profile_after_mmq, stream));
                CUDA_CHECK(cudaEventSynchronize(profile_after_mmq));
                float quant_ms = 0.0f;
                float mmq_ms = 0.0f;
                float total_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&quant_ms, profile_start, profile_after_quant));
                CUDA_CHECK(cudaEventElapsedTime(&mmq_ms, profile_after_quant, profile_after_mmq));
                CUDA_CHECK(cudaEventElapsedTime(&total_ms, profile_start, profile_after_mmq));
                fprintf(stderr,
                        "GGML_CUDA_PROFILE_MMQ cached_src1=1 quant_ms=%.6f mmq_ms=%.6f total_ms=%.6f "
                        "stream_k=%d native_fp4=0 src0_type=%s src0_ne=%lld,%lld,%lld,%lld "
                        "src1_ne=%lld,%lld,%lld,%lld dst_ne=%lld,%lld,%lld,%lld "
                        "src0_name=%s src1_name=%s src1_ptr=%p name=%s\n",
                        quant_ms,
                        mmq_ms,
                        total_ms,
                        use_stream_k ? 1 : 0,
                        ggml_type_name(src0->type),
                        (long long) src0->ne[0],
                        (long long) src0->ne[1],
                        (long long) src0->ne[2],
                        (long long) src0->ne[3],
                        (long long) src1->ne[0],
                        (long long) src1->ne[1],
                        (long long) src1->ne[2],
                        (long long) src1->ne[3],
                        (long long) dst->ne[0],
                        (long long) dst->ne[1],
                        (long long) dst->ne[2],
                        (long long) dst->ne[3],
                        src0->name,
                        src1->name,
                        (const void *) cached_src1->second.data,
                        dst->name);
                CUDA_CHECK(cudaEventDestroy(profile_start));
                CUDA_CHECK(cudaEventDestroy(profile_after_quant));
                CUDA_CHECK(cudaEventDestroy(profile_after_mmq));
            }
            return;
        }

        const bool produce_prequant_cache =
            !use_native_fp4 &&
            ctx.mmq_prequant_target_tensor == dst &&
            ctx.mmq_prequant_target_consumer_type != GGML_TYPE_COUNT &&
            ggml_is_contiguous(dst) &&
            dst->ne[2] == 1 && dst->ne[3] == 1;
        const bool fuse_activation_prequant =
            produce_prequant_cache &&
            activation != MMQ_ACT_NONE &&
            src0->type == GGML_TYPE_Q4_1 &&
            ctx.mmq_prequant_target_consumer_type == GGML_TYPE_Q4_1 &&
            getenv("GGML_CUDA_ENABLE_MMQ_ACT_PREQUANT_FUSION") != nullptr;
        const int64_t q8_only_max_cols = [] {
            const char * env = getenv("GGML_CUDA_MMQ_Q8_ONLY_MAX_COLS");
            return env == nullptr ? INT64_MAX : int64_t{std::atoll(env)};
        }();
        const int64_t q8_only_min_cols = [] {
            const char * env = getenv("GGML_CUDA_MMQ_Q8_ONLY_MIN_COLS");
            return env == nullptr ? int64_t{0} : int64_t{std::atoll(env)};
        }();
        const bool q8_only_producer_fusion =
            produce_prequant_cache &&
            ctx.mmq_prequant_target_q8_only &&
            activation != MMQ_ACT_NONE &&
            (src0->type == GGML_TYPE_Q4_1 || src0->type == GGML_TYPE_Q8_0) &&
            ctx.mmq_prequant_target_consumer_type == src0->type &&
            dst->ne[1] >= q8_only_min_cols &&
            dst->ne[1] <= q8_only_max_cols;
        if (q8_only_producer_fusion && getenv("GGML_CUDA_PROFILE_MMQ_Q8_ONLY_CANDIDATES") != nullptr) {
            fprintf(stderr,
                    "GGML_CUDA_PROFILE_MMQ_Q8_ONLY_TARGET src0_type=%s dst_ne=%lld,%lld,%lld,%lld name=%s\n",
                    ggml_type_name(src0->type),
                    (long long) dst->ne[0],
                    (long long) dst->ne[1],
                    (long long) dst->ne[2],
                    (long long) dst->ne[3],
                    dst->name);
        }
        ggml_backend_cuda_context::mmq_prequant_cache_entry prequant_cache_entry;
        if (produce_prequant_cache) {
            const int64_t dst_ne0_padded = GGML_PAD(dst->ne[0], MATRIX_ROW_PADDING);
            const size_t nbytes_dst_q8_1_payload =
                dst->ne[3] * dst->ne[2] * dst->ne[1] * dst_ne0_padded * sizeof(block_q8_1) / QK8_1;
            const size_t nbytes_dst_q8_1 =
                nbytes_dst_q8_1_payload +
                get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
            prequant_cache_entry.consumer_type = ctx.mmq_prequant_target_consumer_type;
            prequant_cache_entry.ne0_padded = dst_ne0_padded;
            prequant_cache_entry.ne1 = dst->ne[1];
            prequant_cache_entry.ne2 = dst->ne[2];
            prequant_cache_entry.ne3 = dst->ne[3];
            prequant_cache_entry.storage = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes_dst_q8_1);
            prequant_cache_entry.data = prequant_cache_entry.storage->get();
        }
        if (profile_mmq) {
            CUDA_CHECK(cudaEventCreate(&profile_start));
            CUDA_CHECK(cudaEventCreate(&profile_after_quant));
            CUDA_CHECK(cudaEventCreate(&profile_after_mmq));
            CUDA_CHECK(cudaEventRecord(profile_start, stream));
        }
        const int64_t qs11 = src1->nb[1] / ts_src1;
        const int64_t qs12 = src1->nb[2] / ts_src1;
        const int64_t qs13 = src1->nb[3] / ts_src1;
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);

        const bool use_stream_k_for_run = q8_only_producer_fusion ? false : use_stream_k;
        const mmq_activation activation_for_run =
            q8_only_producer_fusion ? activation : (fuse_activation_prequant ? MMQ_ACT_NONE : activation);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        src1_q8_1.alloc(nbytes_src1_q8_1);
        if (use_native_fp4) {
            static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
            quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, qs11, qs12, qs13, ne10_padded,
                                  ne11, ne12, ne13, stream);

        } else {
            quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, qs11, qs12, qs13, ne10_padded,
                                   ne11, ne12, ne13, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        if (profile_mmq) {
            CUDA_CHECK(cudaEventRecord(profile_after_quant, stream));
        }

        ggml_cuda_mul_mat_q_run_prequantized(ctx, src0_d, src0->type, (const int *) src1_q8_1.get(), dst_d,
                                             bias_d, activation_for_run,
                                             ne00, ne01, ne1, s01, ne11, s1,
                                             ne02, ne12, s02, s2,
                                             ne03, ne13, s03, s3,
                                             ne10_padded, use_native_fp4, use_stream_k_for_run,
                                             q4_1_full_tile_fastpath,
                                             q8_only_producer_fusion ? (block_q8_1_mmq *) prequant_cache_entry.data : nullptr,
                                             stream);
        if (profile_mmq) {
            CUDA_CHECK(cudaEventRecord(profile_after_mmq, stream));
            CUDA_CHECK(cudaEventSynchronize(profile_after_mmq));
            float quant_ms = 0.0f;
            float mmq_ms = 0.0f;
            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&quant_ms, profile_start, profile_after_quant));
            CUDA_CHECK(cudaEventElapsedTime(&mmq_ms, profile_after_quant, profile_after_mmq));
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, profile_start, profile_after_mmq));
            fprintf(stderr,
                    "GGML_CUDA_PROFILE_MMQ quant_ms=%.6f mmq_ms=%.6f total_ms=%.6f "
                    "stream_k=%d native_fp4=%d src0_type=%s src0_ne=%lld,%lld,%lld,%lld "
                    "src1_ne=%lld,%lld,%lld,%lld dst_ne=%lld,%lld,%lld,%lld "
                    "src0_name=%s src1_name=%s src1_ptr=%p name=%s\n",
                    quant_ms,
                    mmq_ms,
                    total_ms,
                    use_stream_k ? 1 : 0,
                    use_native_fp4 ? 1 : 0,
                    ggml_type_name(src0->type),
                    (long long) src0->ne[0],
                    (long long) src0->ne[1],
                    (long long) src0->ne[2],
                    (long long) src0->ne[3],
                    (long long) src1->ne[0],
                    (long long) src1->ne[1],
                    (long long) src1->ne[2],
                    (long long) src1->ne[3],
                    (long long) dst->ne[0],
                    (long long) dst->ne[1],
                    (long long) dst->ne[2],
                    (long long) dst->ne[3],
                    src0->name,
                    src1->name,
                    src1->data,
                    dst->name);
            CUDA_CHECK(cudaEventDestroy(profile_start));
            CUDA_CHECK(cudaEventDestroy(profile_after_quant));
            CUDA_CHECK(cudaEventDestroy(profile_after_mmq));
        }

        if (produce_prequant_cache) {
            const ggml_type consumer_type = prequant_cache_entry.consumer_type;
            ctx.mmq_prequant_target_tensor = nullptr;
            ctx.mmq_prequant_cache_key_tensor = nullptr;
            ctx.mmq_prequant_target_consumer_type = GGML_TYPE_COUNT;
            ctx.mmq_prequant_target_ne0_padded = 0;
            ctx.mmq_prequant_target_ne1 = 0;
            ctx.mmq_prequant_target_ne2 = 0;
            ctx.mmq_prequant_target_ne3 = 0;
            ctx.mmq_prequant_target_q8_only = false;

            if (q8_only_producer_fusion) {
                // The producer MMQ kernel wrote the activation output directly to the q8_1 MMQ cache.
                const size_t nbytes_q8_only_payload =
                    dst->ne[3] * dst->ne[2] * dst->ne[1] * prequant_cache_entry.ne0_padded * sizeof(block_q8_1) / QK8_1;
                const size_t nbytes_q8_only_storage =
                    nbytes_q8_only_payload + get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
                const int mmq_y = get_mmq_y_host(cc);
                GGML_ASSERT(mmq_y == 4 * QK8_1);
                const int64_t written_qblocks = (dst->ne[0] + mmq_y - 1) / mmq_y;
                const int64_t padded_qblocks = prequant_cache_entry.ne0_padded / (4 * QK8_1);
                if (written_qblocks < padded_qblocks) {
                    const size_t written_bytes =
                        (size_t) written_qblocks * dst->ne[1] * sizeof(block_q8_1_mmq);
                    const size_t tail_bytes =
                        (size_t) (padded_qblocks - written_qblocks) * dst->ne[1] * sizeof(block_q8_1_mmq);
                    CUDA_CHECK(cudaMemsetAsync(prequant_cache_entry.data + written_bytes, 0, tail_bytes, stream));
                }
                if (getenv("GGML_CUDA_DEBUG_MMQ_Q8_ONLY_COMPARE") != nullptr) {
                    ggml_cuda_mul_mat_q_run_prequantized(ctx, src0_d, src0->type, (const int *) src1_q8_1.get(), dst_d,
                                                         bias_d, activation,
                                                         ne00, ne01, ne1, s01, ne11, s1,
                                                         ne02, ne12, s02, s2,
                                                         ne03, ne13, s03, s3,
                                                         ne10_padded, false, use_stream_k_for_run,
                                                         q4_1_full_tile_fastpath, nullptr, stream);
                    CUDA_CHECK(cudaGetLastError());
                }
                ggml_cuda_debug_compare_mmq_q8_only(ctx, dst_d, prequant_cache_entry.data, consumer_type, dst,
                                                    prequant_cache_entry.ne0_padded, nbytes_q8_only_payload,
                                                    nbytes_q8_only_storage, stream);
            } else if (fuse_activation_prequant) {
                quantize_mmq_q8_1_inplace_activation_cuda(dst_d, prequant_cache_entry.data, consumer_type, activation,
                                                          dst->ne[0],
                                                          dst->nb[1] / ts_dst, dst->nb[2] / ts_dst, dst->nb[3] / ts_dst,
                                                          prequant_cache_entry.ne0_padded, dst->ne[1], dst->ne[2], dst->ne[3],
                                                          stream);
            } else {
                quantize_mmq_q8_1_cuda(dst_d, nullptr, prequant_cache_entry.data, consumer_type, dst->ne[0],
                                       dst->nb[1] / ts_dst, dst->nb[2] / ts_dst, dst->nb[3] / ts_dst,
                                       prequant_cache_entry.ne0_padded, dst->ne[1], dst->ne[2], dst->ne[3], stream);
            }
            CUDA_CHECK(cudaGetLastError());
            ctx.mmq_prequant_cache.emplace_back(dst, std::move(prequant_cache_entry));
        }
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d, nullptr, nullptr, MMQ_ACT_NONE,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, q4_1_full_tile_fastpath, ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k =
        getenv("GGML_CUDA_DISABLE_MMQ_STREAM_K") == nullptr &&
        ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc)) &&
        src1_ncols == ne11;
    const bool q4_1_full_tile_fastpath =
        src0->type == GGML_TYPE_Q4_1 &&
        (getenv("GGML_CUDA_DISABLE_MMQ_Q4_1_FULL_TILE_FASTPATH") == nullptr ||
         std::atoi(getenv("GGML_CUDA_DISABLE_MMQ_Q4_1_FULL_TILE_FASTPATH")) == 0);
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i, nullptr, nullptr, MMQ_ACT_NONE,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, q4_1_full_tile_fastpath, src1_ncols};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
