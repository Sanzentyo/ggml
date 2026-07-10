#pragma once

#include "common.cuh"

bool ggml_cuda_cudnn_mlp_fc1_gelu_bf16(ggml_backend_cuda_context& ctx,
                                       const ggml_tensor* mm_node,
                                       const ggml_tensor* bias,
                                       ggml_tensor* dst);

bool ggml_cuda_cudnn_mlp_fc1_gelu_f32(ggml_backend_cuda_context& ctx,
                                      const ggml_tensor* mm_node,
                                      const ggml_tensor* bias,
                                      ggml_tensor* dst);

bool ggml_cuda_cudnn_mlp_fc1_gelu_f16(ggml_backend_cuda_context& ctx,
                                      const ggml_tensor* mm_node,
                                      const ggml_tensor* bias,
                                      ggml_tensor* dst);
