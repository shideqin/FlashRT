// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 GEMM with fused residual add in epilogue,
// BF16 output, SM120a.
//
// Thin wrapper around the existing NVFP4 GEMM. When residual_bf16 is
// non-null the epilogue computes D = alpha*(A@B^T) + residual_bf16
// (beta=1.0) instead of the default D = alpha*(A@B^T) (beta=0.0).
//
// Used by OmniVoice V23 to fuse the per-layer "GEMM + residual add"
// into a single kernel, saving one round-trip through global memory
// (~1.4 MB read/write per layer).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// NVFP4 W4A16 GEMM with optional residual add, BF16 output, SM120a.
//
//   A_packed      : (M, K/2)   u8   row-major  (FP4 e2m1, 2 per byte)
//   B_packed      : (N, K/2)   u8   row-major  (read as ColumnMajor (K, N))
//   D_bf16        : (M, N)     bf16 row-major   output
//   residual_bf16 : (M, N)     bf16 row-major   optional, nullptr → beta=0
//   SFA           : (M, K/16)  e4m3 (CUTLASS blockscaled atom layout)
//   SFB           : (N, K/16)  e4m3 (CUTLASS blockscaled atom layout)
//   alpha         : fp32 scalar
//
// Cooperative schedule (default).
void fp4_w4a16_gemm_residual_bf16out_sm120(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  residual_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

// Pingpong schedule variant. Use when N >= 4096 (QKV, GateUp).
void fp4_w4a16_gemm_residual_bf16out_sm120_pingpong(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  residual_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
