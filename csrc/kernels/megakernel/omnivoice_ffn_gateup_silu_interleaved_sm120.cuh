// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice FFN GateUp+SiLU Megakernel — Interleaved-B (Agent B v6)
//
// Single-kernel fusion: GateUp GEMM + SiLU+Mul + NVFP4 quantize.
// Key innovation: INTERLEAVED weight layout enables gate/up pairs
// to be co-located in the same accumulator tile, enabling efficient
// epilogue fusion without dual-B overhead.
//
// Weight layout: B[2*i] = gate[i], B[2*i+1] = up[i] (interleaved)
// vs standard: B[i] = gate[i], B[i+FFN] = up[i] (contiguous)
//
// Tile: M=64 × N=128 × K=64, 2-stage cp.async, 4 warps, 128 threads.
// N=128 covers 64 gate/up pairs per tile.
//
// Reference:
//   omnivoice_ffn_gateup_megakernel_sm120.cu — MMA pattern, SF handling

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace megakernel {

// OmniVoice GateUp+SiLU megakernel v6 (interleaved-B).
//
// inp_packed  : (M, K/2)       uint8  NVFP4 activations (row-major)
// inp_sfa     : swizzled       uint8  UE4M3 scale factors (FlashRT-swizzled)
// gu_packed   : (2*FFN, K/2)   uint8  INTERLEAVED weight (gate[0],up[0],gate[1],up[1],...)
// gu_sfb      : swizzled       uint8  UE4M3 weight SF (FlashRT-swizzled, same layout as gu_packed)
// out_packed  : (M, FFN/2)     uint8  NVFP4 output (SiLU(gate)*up, row-major)
// out_sfa     : swizzled       uint8  UE4M3 scale factors (FlashRT-swizzled)
//
// M   : batch * seq_len (356 for OmniVoice)
// FFN : 3072 (intermediate dimension)
// K   : 1024 (hidden dimension)
// alpha: fp32 global scale
//
// Returns 0 on success, nonzero on error.
int omnivoice_ffn_gateup_silu_interleaved_sm120(
    const void* inp_packed,  const void* inp_sfa,
    const void* gu_packed,   const void* gu_sfb,
    void*       out_packed,  void*       out_sfa,
    int M, int FFN, int K,
    float alpha,
    cudaStream_t stream);

}  // namespace megakernel
}  // namespace flash_rt
