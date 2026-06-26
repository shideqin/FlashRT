// ================================================================
// FlashRT — MaskGIT cross-position select kernel
//
// Fuses, per batch item, the final selection step of MaskGIT sampling:
//   raw   = confidence - codebook_idx * layer_penalty_factor
//   score = (pt>0) ? raw/pt + gumbel : raw
//   mask already-filled positions (-inf)
//   top-k (k from device pointer) over [C*T] scores
//   scatter predicted tokens into the selected positions
// One block per batch; shared-mem bitonic sort; deterministic Philox.
// Replaces: score arith + _gumbel_sample + masked_fill + topk + scatter.
// ================================================================

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

void maskgit_select_topk_bf16(
    const __nv_bfloat16* confidence,  // [B, C*T]
    const int* pred_tokens,           // [B, C*T]
    int* sample_tokens,               // [B, C*T]  (in-place updated)
    const int* k_dev,                 // device pointer: #positions to fill this step
    int B, int C, int T, float layer_penalty_factor,
    float position_temp, int mask_id, unsigned long long seed,
    cudaStream_t stream);
