// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — true-SiLU variant of silu_mul_two_fp4_to_fp4.
//
// Identical to the GEGLU version except uses sigmoid(g)*g*u instead of
// tanh-approx GELU(g)*u. For Qwen3-based models that use SwiGLU (SiLU gate).
//
// Additive: does NOT modify existing kernels.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// True-SiLU combiner: silu(gate) * up → FP4 + SFA.
// Gate and up are FP4 packed + SFA (CUTLASS tile-interleaved layout).
void silu_mul_two_fp4_to_fp4_true_silu(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int half_dim,
    cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
