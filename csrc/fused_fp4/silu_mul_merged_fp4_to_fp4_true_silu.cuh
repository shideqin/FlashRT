// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — True-SiLU combiner for merged GateUp fp4out GEMM output.
//
// Reads a single merged FP4 buffer [S, 2*H] (gate in cols [0,H), up in cols [H,2H))
// with a unified SFA, applies true SiLU: silu(gate) * up → FP4 + SFA output [S, H].
//
// Eliminates the SF-split kernel that would otherwise be required to feed
// silu_mul_two_fp4_to_fp4_true_silu with separate gate/up SFA buffers.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// True-SiLU merged combiner: silu(gate) * up → FP4 + SFA.
// merged_packed/merged_sfa: from fp4out GateUp GEMM, shape (seq_len, 2*half_dim).
// out_packed/out_sfa: output FP4, shape (seq_len, half_dim).
void silu_mul_merged_fp4_to_fp4_true_silu(
    const uint8_t* merged_packed, const uint8_t* merged_sfa,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int half_dim,
    cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
