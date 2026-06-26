// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — Pad FP4 packed + SFA from shape (src_rows, D) to (dst_rows, D).
//
// dst_rows >= src_rows. Padding rows are zero-filled (packed=0, SFA=0).
// Handles CUTLASS Sm1xxBlockScaledConfig tile-interleaved SFA layout remapping.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// Pad FP4 activation: [src_rows, D] → [dst_rows, D].
// packed/sfa use CUTLASS Sm1xxBlockScaledConfig<16> tile-interleaved layout.
void pad_fp4_sfa(
    const uint8_t* src_packed, const uint8_t* src_sfa,
    uint8_t* dst_packed, uint8_t* dst_sfa,
    int src_rows, int dst_rows, int D,
    cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
