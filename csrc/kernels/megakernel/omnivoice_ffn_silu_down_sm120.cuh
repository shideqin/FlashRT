// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice FFN SiLU+Down Fused Kernel (V40 Agent B).
//
// Fuses SiLU(gate)*up + NVFP4 quantize + Down GEMM + residual add
// into a single kernel launch.
//
// Inputs:
//   gateup_bf16  : [M, 2*FFN] BF16 — gate/up from cuBLASLt GateUp GEMM
//   down_packed  : [D, FFN/2] uint8 — FP4 down projection weight
//   down_sf      : [D, FFN/16] uint8 — scale factors (swizzled layout)
//   residual_bf16: [M, D] BF16 — residual from attention path
//
// Output:
//   out_bf16     : [M, D] BF16 — residual + Down(silu(gate)*up)
//
// Shape constraints: FFN=3072, D=1024.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace megakernel {

/// SiLU+Down fused: computes silu(gate)*up, quantizes to FP4,
/// runs Down GEMM with hand-written NVFP4 MMA, adds residual.
///
/// Returns 0 on success, nonzero on error.
int omnivoice_ffn_silu_down_sm120(
    const void* gateup_bf16,
    const void* down_packed,  const void* down_sf,
    const void* residual_bf16,
    void*       out_bf16,
    int M, int FFN, int D,
    float alpha,
    cudaStream_t stream);

}  // namespace megakernel
}  // namespace flash_rt
