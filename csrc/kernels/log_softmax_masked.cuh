// ================================================================
// FlashRT — Masked Log-Softmax kernel (BF16)
//
// Fused log_softmax + mask for MaskGIT post-processing.
// Replaces: F.log_softmax(logits, dim=-1) + logits[..., mask_id] = -inf
//
// 1 warp per row, BF16 in → FP32 compute → BF16 out.
// Mask token at `mask_col` is set to -inf after log_softmax.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

void log_softmax_masked_bf16(
    __nv_bfloat16* data,     // [rows, cols] in-place
    int rows,
    int cols,
    int mask_col,            // column index to mask (set to -inf)
    cudaStream_t stream = 0);
