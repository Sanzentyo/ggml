#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_norm_fused_add(ggml_backend_cuda_context & ctx,
                                 ggml_tensor *               dst,
                                 ggml_tensor *               mul_tensor,
                                 ggml_tensor *               add_tensor,
                                 ggml_tensor *               cpy_tensor = nullptr);

void ggml_cuda_op_add_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               add_input,
                                     ggml_tensor *               norm_tensor,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     bool                        preserve_add_output = false,
                                     ggml_tensor *               cpy_tensor = nullptr);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
