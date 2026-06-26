// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — Fused Q/K RMSNorm + RoPE kernel (Agent B).
//
// Combines fused_qk_norm_bf16 + qwen36_partial_rope_qk_bf16 into a single
// kernel launch. For OmniVoice MaskGIT (BS=356, NH=16, NKV=8, HD=128),
// this eliminates one kernel launch and the global memory round-trip
// for the intermediate q_flat/k_flat buffers (~24% of per-step time).
//
// Pipeline:
//   1. Read Q and K from strided Dq buffer
//   2. Apply per-head RMSNorm (Q and K separately)
//   3. Apply RoPE (first rope_dim elements of each head)
//   4. Write to flat output buffers (same layout as before)
//
// ================================================================

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

void fused_qk_norm_rope_bf16(
    const __nv_bfloat16* dq,         // [BS, QKVD] = [BS, NH*HD + 2*NKV*HD]
    const __nv_bfloat16* q_weight,   // [HD]  Q norm weight
    const __nv_bfloat16* k_weight,   // [HD]  K norm weight
    const __nv_bfloat16* cos,        // [BS, rope_dim]  RoPE cos
    const __nv_bfloat16* sin,        // [BS, rope_dim]  RoPE sin
    __nv_bfloat16* q_out,           // [BS * NH, HD]  flat Q output
    __nv_bfloat16* k_out,           // [BS * NKV, HD]  flat K output
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream);

// V15: Optimized variant with separate temp buffers for RoPE output.
// Eliminates 48 __syncthreads() (73→25) by writing RoPE to q_temp/k_temp
// instead of in-place with shared-memory buffering. Python swaps the
// q_out↔q_temp and k_out↔k_temp pointers after the kernel.
void fused_qk_norm_rope_v2_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,        // norm output (intermediate)
    __nv_bfloat16* k_out,        // norm output (intermediate)
    __nv_bfloat16* q_temp,       // RoPE output (new!)
    __nv_bfloat16* k_temp,       // RoPE output (new!)
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream);

// V21v3: Further optimized variant — norm output stays in shared memory,
// never written to global memory. Eliminates q_out/k_out global memory
// traffic entirely. RoPE reads from shared memory, writes to q_temp/k_temp.
// Signature identical to v2 (q_out/k_out pointers are unused by kernel).
void fused_qk_norm_rope_v3_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,        // UNUSED (kept for API compat)
    __nv_bfloat16* k_out,        // UNUSED (kept for API compat)
    __nv_bfloat16* q_temp,       // RoPE output
    __nv_bfloat16* k_temp,       // RoPE output
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream);

// V26v4: Warp-per-head parallelism — 8 warps process 8 heads in parallel,
// warp shuffle for RMS reduction. 6 __syncthreads() vs 48 in v3.
// Same API signature as v2/v3 for drop-in compatibility.
void fused_qk_norm_rope_v4_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,        // UNUSED (API compat)
    __nv_bfloat16* k_out,        // UNUSED (API compat)
    __nv_bfloat16* q_temp,       // RoPE output
    __nv_bfloat16* k_temp,       // RoPE output
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
