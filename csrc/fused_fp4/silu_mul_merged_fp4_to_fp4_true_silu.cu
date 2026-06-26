// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — True-SiLU merged combiner for fp4out GateUp GEMM output.
//
// Single kernel: reads merged GateUp FP4 [S, 2*H] (one buffer, one SFA),
// applies silu(gate_cols) * up_cols, writes FP4 [S, H] + SFA.
//
// Avoids an SF-split kernel launch and keeps the entire FFN activation
// chain in FP4, eliminating the 4.37 MB BF16 round-trip from GateUp → SiLU.

#include "silu_mul_merged_fp4_to_fp4_true_silu.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_fp4 {

#if FV_HAVE_CUTLASS

using CfgF4 = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ float e2m1_to_fp32(uint8_t v) {
    static constexpr float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (v & 0x8) ? -mags[v & 0x7] : mags[v & 0x7];
}

__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

// True SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// With BF16 round-trip on silu(g) to match Qwen3 reference.
__device__ __forceinline__ float true_silu_mul(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    float silu_g_bf = static_cast<float>(__float2bfloat16(silu_g));
    return silu_g_bf * u;
}

// Kernel: 1 thread = 1 NVFP4 block (16 elements).
// Reads gate scale from merged_sfa at (row, col_block) and up scale from
// merged_sfa at (row, col_block + H/16) where H = half_dim.
template <class LayoutSFMerged, class LayoutSFOut>
__global__ void silu_mul_merged_fp4_to_fp4_true_silu_kernel(
    const uint8_t* __restrict__ merged_packed,   // [S, H] bytes (packed 2H elements)
    const uint8_t* __restrict__ merged_sfa,
    uint8_t* __restrict__ out_packed,            // [S, H/2] bytes
    uint8_t* __restrict__ out_sfa,
    LayoutSFMerged layout_merged,                 // for shape (S, 2H)
    LayoutSFOut layout_out,                       // for shape (S, H)
    int H) {
    // H = FFN (half of merged width)
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;  // blocks in output (gate/up each have H/16 blocks)
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;

    // Gate scale: merged layout at block (row, col_block)
    int gate_sfa_off = layout_merged(row, col_base, 0);
    // Up scale: merged layout at block (row, col_block + H)
    int up_sfa_off   = layout_merged(row, col_base + H, 0);

    uint8_t gate_sf_byte = merged_sfa[gate_sfa_off];
    uint8_t up_sf_byte   = merged_sfa[up_sfa_off];
    __nv_fp8_e4m3 gate_bs_q, up_bs_q;
    *reinterpret_cast<uint8_t*>(&gate_bs_q) = gate_sf_byte;
    *reinterpret_cast<uint8_t*>(&up_bs_q)   = up_sf_byte;
    float gate_scale = static_cast<float>(gate_bs_q);
    float up_scale   = static_cast<float>(up_bs_q);

    // Gate packed data starts at row * (H) bytes (since merged has 2H elements → H bytes)
    // Gate occupies bytes [0, H/2) of the merged row
    // Up   occupies bytes [H/2, H) of the merged row
    const int H_bytes = H / 2;  // H elements → H/2 packed bytes
    const uint8_t* gate_row = merged_packed + row * (H_bytes * 2) + block_idx * 8;
    const uint8_t* up_row   = merged_packed + row * (H_bytes * 2) + H_bytes + block_idx * 8;

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t gb = gate_row[p];
        uint8_t ub = up_row[p];
        float g_lo = e2m1_to_fp32(gb & 0xF) * gate_scale;
        float g_hi = e2m1_to_fp32(gb >> 4)  * gate_scale;
        float u_lo = e2m1_to_fp32(ub & 0xF) * up_scale;
        float u_hi = e2m1_to_fp32(ub >> 4)  * up_scale;
        float v0 = true_silu_mul(g_lo, u_lo);
        float v1 = true_silu_mul(g_hi, u_hi);
        vals[2*p]   = v0;
        vals[2*p+1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    // Per-block quantize
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);

    int out_sfa_off = layout_out(row, col_base, 0);
    out_sfa[out_sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* op = out_packed + row * H_bytes + block_idx * 8;
    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1(vals[2*p]   * inv_bs);
        uint8_t hi = fp32_to_e2m1(vals[2*p+1] * inv_bs);
        op[p] = lo | (hi << 4);
    }
}

#endif  // FV_HAVE_CUTLASS

void silu_mul_merged_fp4_to_fp4_true_silu(
    const uint8_t* merged_packed, const uint8_t* merged_sfa,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    // Layout for merged input: shape (seq_len, 2*H)
    auto shape_merged = cute::make_shape(seq_len, 1, 2 * H, 1);
    auto layout_merged = CfgF4::tile_atom_to_shape_SFA(shape_merged);

    // Layout for output: shape (seq_len, H)
    auto shape_out = cute::make_shape(seq_len, 1, H, 1);
    auto layout_out = CfgF4::tile_atom_to_shape_SFA(shape_out);

    const int n_blocks = H / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    silu_mul_merged_fp4_to_fp4_true_silu_kernel<<<grid, block, 0, stream>>>(
        merged_packed, merged_sfa, out_packed, out_sfa,
        layout_merged, layout_out, H);
#else
    (void)merged_packed; (void)merged_sfa; (void)out_packed; (void)out_sfa;
    (void)seq_len; (void)H; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
