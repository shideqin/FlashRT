// ================================================================
// FlashRT — Fused Q/K RMSNorm kernel
// Reads Q and K from strided Dq buffer, applies per-head RMSNorm,
// writes to flat RoPE-ready output buffers.
//
// Dq layout: [BS, QKVD] = [BS, NQK + 2*KVD]
//   Q slice at row offset 0,      len NQK = NH * HD
//   K slice at row offset NQK,    len KVD = NKV * HD
//   V slice at row offset NQK+KVD (not touched by this kernel)
//
// Output: q_out [BS * NH, HD]  (contiguous, RoPE-ready)
//         k_out [BS * NKV, HD] (contiguous, RoPE-ready)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

void fused_qk_norm_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream = 0);
