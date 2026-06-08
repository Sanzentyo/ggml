#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);
bool ggml_cuda_cpy_bias_axis0(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * src,
                              const ggml_tensor * bias,
                              ggml_tensor * dst);
bool ggml_cuda_add_bias_axis2_permute_cont(ggml_backend_cuda_context & ctx,
                                           const ggml_tensor * add,
                                           ggml_tensor * cont);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
