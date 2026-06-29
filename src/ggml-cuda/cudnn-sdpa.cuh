#pragma once

#include "common.cuh"

bool ggml_cuda_cudnn_sdpa_head32(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
bool ggml_cuda_cudnn_sdpa_head64(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
