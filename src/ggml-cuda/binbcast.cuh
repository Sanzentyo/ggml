#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_add_with_mmq_q8_1_prequant(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add_unary(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * unary);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
bool ggml_cuda_op_fused_add_cpy(ggml_backend_cuda_context & ctx,
                                ggml_tensor * dst,
                                int n_fuse,
                                ggml_tensor * cpy);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_rope_pair_fused(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor * x_re,
                                  const ggml_tensor * x_im,
                                  const ggml_tensor * cos,
                                  const ggml_tensor * sin,
                                  ggml_tensor * dst,
                                  bool exact_inplace);
void ggml_cuda_op_cont_rope_pair_fused(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * x,
                                       const ggml_tensor * cos,
                                       const ggml_tensor * sin,
                                       ggml_tensor * dst);
