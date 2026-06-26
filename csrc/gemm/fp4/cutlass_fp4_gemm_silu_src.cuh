// ============================================================================
//  FlashRT — NVFP4 GEMM with fused silu(C) * acc → fp4 + SFA epilogue.
//
//  APPROACH: Pass gate values as C source matrix (Sm90SrcFetch) instead of
//  aux buffer (Sm90AuxLoad, which has a null-pointer bug in CUTLASS 4.3.1).
//
//  EVT tree:
//    block_scale_store(
//      silu(C) * acc
//    )
//
//  where C = gate_bf16 [M, N] (same shape as GEMM output), and acc is the
//  accumulator from X @ Wu^T.
//
//  This fuses SiLU+Mul+NVFP4 quantize into the Up GEMM epilogue, eliminating
//  the standalone SiLU+Mul+NVFP4 kernel (~11.2μs/layer on OmniVoice).
//
//  Additive: does NOT modify existing cutlass_fp4_gemm.cu / variants.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// Run NVFP4 GEMM with fused silu_mul_src epilogue.
//
//   D[M, N/2] (fp4 packed)  =  blockscale(silu(C_gate_bf16[M, N]) * acc)
//   SFA out                  =  per-16-element UE4M3 scales
//
// Inputs: A (activation NVFP4), B (weight NVFP4), C_gate (gate BF16 source).
// Output: D (packed FP4), D_SFA (block scales).
//
// Returns 0 on success, nonzero CUTLASS status code on error.
int cutlass_fp4_gemm_silu_src_fp4(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* C_gate_bf16,
    void*       D_packed,
    void*       D_SFA,
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
