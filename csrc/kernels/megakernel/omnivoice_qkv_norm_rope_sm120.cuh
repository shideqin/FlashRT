// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice QKV+QK Megakernel — single-kernel fusion of QKV GEMM + QK Norm+RoPE.
//
// Replaces two kernel launches:
//   1. QKV GEMM (NVFP4 W4A16 → BF16)
//   2. QK Norm+RoPE v4 (BF16 → BF16, with RMSNorm + RoPE)
//
// Key design: BLOCK_N = 128 = HD, so each N-tile covers exactly one head.
// This allows per-head RMSNorm entirely within a single block.
//
// N-tile mapping (total 32 tiles):
//   Tiles 0..15  (global_col 0..2047):     Q heads 0..15
//   Tiles 16..23 (global_col 2048..3071):  K heads 0..7
//   Tiles 24..31 (global_col 3072..4095):  V heads 0..7

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt {
namespace megakernel {

void omnivoice_qkv_norm_rope_sm120(
    const uint8_t* act_packed,   // [BS, D/2] NVFP4 packed
    const uint8_t* act_sf,       // swizzled SF [BS, D]
    const uint8_t* w_packed,     // [QKVD, D/2] NVFP4 packed
    const uint8_t* w_sf,         // swizzled SF [QKVD, D]
    const __nv_bfloat16* q_weight, // [HD] Q per-head norm weight
    const __nv_bfloat16* k_weight, // [HD] K per-head norm weight
    const __nv_bfloat16* cos,      // [BS, HD] RoPE cos table
    const __nv_bfloat16* sin,      // [BS, HD] RoPE sin table
    __nv_bfloat16* q_out,        // [BS*NH, HD] RoPE'd Q output
    __nv_bfloat16* k_out,        // [BS*NKV, HD] RoPE'd K output
    __nv_bfloat16* v_out,        // [BS*NKV, HD] raw V output
    int BS, int D, int QKVD,
    int NH, int NKV, int HD,
    float eps,
    cudaStream_t stream);

}  // namespace megakernel
}  // namespace flash_rt
