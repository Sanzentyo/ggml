#include "common.cuh"

#define CUDA_WIN_PART_BLOCK_SIZE 256

void ggml_cuda_op_win_part(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
bool ggml_cuda_op_win_part_cpy(ggml_backend_cuda_context& ctx,
                               ggml_tensor* win_part_node,
                               ggml_tensor* cpy_node);
void ggml_cuda_op_win_unpart(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
bool ggml_cuda_op_win_unpart_add(ggml_backend_cuda_context& ctx,
                                 ggml_tensor* win_unpart_node,
                                 ggml_tensor* add_node);
