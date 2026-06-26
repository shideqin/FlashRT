// ================================================================
// FlashRT — MaskGIT per-row sample kernel (BF16)
//
// Fuses, for each [B,C,t_len] row over V columns:
//   1. filter_top_k: keep top `num_filt` logits, rest -inf
//   2. gumbel_sample(filtered, ct) → argmax  == predicted token
//   3. confidence = max(log_probs)
// One block per row, 256 threads, shared-mem bitonic sort of (val,idx).
// Deterministic Philox gumbel (graph-capture safe).
//
// Replaces _filter_top_k + _gumbel_sample + argmax + max (PyTorch).
// ================================================================

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

void maskgit_sample_row_bf16(
    const __nv_bfloat16* log_probs,   // [rows, V]
    int* pred_tokens,                 // [rows]   predicted token per row
    __nv_bfloat16* confidence,        // [rows]   max log_prob per row (bf16)
    int rows, int V, int num_filt,    // num_filt = top-k count for filter (e.g. ceil(0.1*V))
    int mask_id, float class_temp, unsigned long long seed,
    cudaStream_t stream);
