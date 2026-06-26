// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice FFN GateUp+SiLU Megakernel v5 — Merged-B (Agent B)
//
// Single-kernel fusion: GateUp GEMM + SiLU+Mul + NVFP4 quantize.
// Uses a SINGLE merged B matrix [2*FFN, K] (same layout as cuBLASLt GEMM).
// Epilogue splits gate/up halves from accumulator, applies SiLU(gate)*up,
// then block-quantizes to NVFP4.
//
// Key difference from v4: v4 used dual-B (separate gate and up B matrices)
// which doubled B loads and MMA calls. v5 uses merged-B + split-in-epilogue,
// matching the cuBLASLt pattern while eliminating the BF16 intermediate.
//
// Tile: M=64 × N=128 × K=64, 2-stage cp.async, 4 warps, pingpong.
// N=128 covers 2 gate columns AND 2 up columns (64 gate + 64 up per half-tile).

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace megakernel {

// OmniVoice GateUp+SiLU megakernel v5 (merged-B).
//
// inp_packed : (M, K/2)     uint8  NVFP4 activations (row-major)
// inp_sfa    : swizzled     uint8  UE4M3 scale factors (FlashRT-swizzled)
// gu_packed  : (2*FFN, K/2) uint8  merged GateUp weight (row-major, gate[0:FFN], up[FFN:2*FFN])
// gu_sfb     : swizzled     uint8  UE4M3 weight SF (FlashRT-swizzled)
// out_packed : (M, FFN/2)   uint8  NVFP4 output (SiLU(gate)*up, row-major)
// out_sfa    : swizzled     uint8  UE4M3 scale factors (FlashRT-swizzled)
//
// M   : batch * seq_len (356 for OmniVoice)
// FFN : 3072 (intermediate dimension)
// K   : 1024 (hidden dimension)
// alpha: fp32 global scale
//
// Returns 0 on success, nonzero on error.
int omnivoice_ffn_gateup_silu_v5_sm120(
    const void* inp_packed,  const void* inp_sfa,
    const void* gu_packed,   const void* gu_sfb,
    void*       out_packed,  void*       out_sfa,
    int M, int FFN, int K,
    float alpha,
    cudaStream_t stream);

}  // namespace megakernel
}  // namespace flash_rt
