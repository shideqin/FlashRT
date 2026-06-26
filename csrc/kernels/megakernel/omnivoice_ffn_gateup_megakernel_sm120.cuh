// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice FFN GateUp Megakernel — Phase 1 v2 (Agent B)
//
// Single-kernel fusion of GateUp GEMM + SiLU+Mul + NVFP4 quantize.
// Eliminates the 4.37MB BF16 intermediate that the V27 two-launch
// chain (FP4 GEMM → silu_mul_merged) writes/reads through HBM.
//
// V2: Uses FlashRT native swizzled SF format throughout — no conversions needed.
//     Drop-in compatible with the rest of the FP4 pipeline (rms_norm_to_nvfp4_swizzled,
//     CUTLASS GEMM, etc.)
//
// NVFP4 MMA instruction (Blackwell SM120):
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//   .row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Tile: M=16 × N=64 × K=64, 2-stage cp.async, 4 warps.
//
// Reference kernels:
//   motus_fp4_conv3d_v19sf_sm120.cu — NVFP4 MMA pattern + SF handling
//   und_ffn_megakernel_v5t_sm120.cu — cp.async pipeline architecture
//   quantize.cu                       — swizzled SF layout definition

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace megakernel {

// OmniVoice GateUp megakernel v2: GateUp GEMM + SiLU+Mul + NVFP4 quantize.
//
// All SF pointers use FlashRT native swizzled layout (same as
// rms_norm_to_nvfp4_swizzled_bf16 output).
//
//   inp_packed  : (M, K/2)     uint8  NVFP4 activations (row-major, 2 FP4/byte)
//   inp_sfa     : swizzled     uint8  UE4M3 scale factors (FlashRT-swizzled)
//   gu_packed   : (2*FFN, K/2) uint8  merged GateUp weight (row-major)
//   gu_sfb      : swizzled     uint8  UE4M3 weight SF (FlashRT-swizzled)
//   out_packed  : (M, FFN/2)   uint8  NVFP4 output (row-major, SiLU(gate)*up)
//   out_sfa     : swizzled     uint8  UE4M3 scale factors (FlashRT-swizzled)
//
//   M           : batch * seq_len (356 for OmniVoice)
//   FFN         : 3072 (intermediate dimension)
//   K           : 1024 (hidden dimension)
//   alpha       : fp32 global scale
//
// Returns 0 on success, nonzero on argument error.
int omnivoice_ffn_gateup_megakernel_sm120(
    const void* inp_packed,  const void* inp_sfa,
    const void* gu_packed,   const void* gu_sfb,
    void*       out_packed,  void*       out_sfa,
    int M, int FFN, int K,
    float alpha,
    cudaStream_t stream);

}  // namespace megakernel
}  // namespace flash_rt
