// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice QKV+QK Megakernel v2 — QKV GEMM + QK Norm+RoPE fusion.
//
// Replaces: QKV GEMM (FP4→BF16, 26μs) + QK Norm+RoPE v4 (BF16, ~30μs)
// With:     Single NVFP4 MMA kernel with RMSNorm+RoPE epilogue.
//
// v2: Complete RoPE epilogue using register-based atom pairing.
//     atom a ↔ a+8 maps to column col ↔ col+64=HD/2, both in same thread.
//     No shared memory needed for RoPE cross-atom access.
//
// Key design: BLOCK_N = 128 = HD, each N-tile covers exactly one head.
// This allows per-head RMSNorm within a single block.
//
// MMA pattern: identical to gateup megakernel (4 warps, m16n8k64 × 4-atom).
//   - BLOCK_M = 64, BLOCK_N = 128, BLOCK_K = 64
//   - 4 warps × 32 threads = 128 threads
//   - Each warp: 16 rows × 128 cols = 2048 f32 output values
//   - N_GROUPS = 4 (128/32), N_ATOMS = 16 (128/8)
//
// N-tile mapping (32 tiles total):
//   0..15:  Q heads 0..15 (QKVD cols 0..2047)
//   16..23: K heads 0..7  (QKVD cols 2048..3071)
//   24..31: V heads 0..7  (QKVD cols 3072..4095)

#include "omnivoice_qkv_norm_rope_sm120.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace flash_rt {
namespace megakernel {
namespace {

constexpr int BLOCK_M      = 64;
constexpr int BLOCK_N      = 128;
constexpr int BLOCK_K      = 64;
constexpr int NUM_WARPS    = 4;
constexpr int THREADS      = NUM_WARPS * 32;  // 128
constexpr int STAGES       = 2;

constexpr int N_ATOMS      = BLOCK_N / 8;      // 16
constexpr int N_GROUPS     = N_ATOMS / 4;       // 4

constexpr int SMEM_K_STRIDE = BLOCK_K / 2 + 16;  // 48 bytes (with padding)
constexpr int SF_K_PER_ROW  = BLOCK_K / 16;       // 4

// ── FlashRT swizzled SF helpers ──
__device__ __forceinline__ int swizzled_sf_base(int row, int kg, int total_groups) {
    int rb = row / 128;
    int ri = row % 128;
    int cb = kg / 4;
    int n_col_blocks = (total_groups + 3) / 4;
    return (rb * n_col_blocks + cb) * 512 + (ri % 32) * 16 + (ri / 32) * 4;
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_int, const uint8_t* src) {
    int b = (src == nullptr) ? 0 : 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_int), "l"(src), "r"(b));
}

__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// NVFP4 MMA 4-atom call
__device__ __forceinline__ void mma_nvfp4_4atom(
    float &dA0, float &dB0, float &dC0, float &dD0,
    float &dA1, float &dB1, float &dC1, float &dD1,
    float &dA2, float &dB2, float &dC2, float &dD2,
    float &dA3, float &dB3, float &dC3, float &dD3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    uint32_t sfa, uint32_t sfb)
{
    constexpr uint16_t bidA = 0, tidA = 0, bidB = 0;
    constexpr uint16_t tidB0 = 0, tidB1 = 1, tidB2 = 2, tidB3 = 3;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
        ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "+f"(dA0), "+f"(dB0), "+f"(dC0), "+f"(dD0)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(dA0), "f"(dB0), "f"(dC0), "f"(dD0),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB0));
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
        ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "+f"(dA1), "+f"(dB1), "+f"(dC1), "+f"(dD1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3),
          "f"(dA1), "f"(dB1), "f"(dC1), "f"(dD1),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB1));
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
        ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "+f"(dA2), "+f"(dB2), "+f"(dC2), "+f"(dD2)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5),
          "f"(dA2), "f"(dB2), "f"(dC2), "f"(dD2),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB2));
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
        ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "+f"(dA3), "+f"(dB3), "+f"(dC3), "+f"(dD3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7),
          "f"(dA3), "f"(dB3), "f"(dC3), "f"(dD3),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB3));
}

// ── Main kernel ──
__global__ void __launch_bounds__(THREADS, 2)
qkv_norm_rope_kernel(
    const uint8_t* __restrict__ act_packed,
    const uint8_t* __restrict__ act_sf,
    const uint8_t* __restrict__ w_packed,
    const uint8_t* __restrict__ w_sf,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int BS, int D, int QKVD,
    int NH, int NKV, int HD,
    float eps)
{
    __shared__ __align__(16) uint8_t A_smem [STAGES][BLOCK_M * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t B_smem [STAGES][BLOCK_N * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t SFA_smem[STAGES][BLOCK_M * SF_K_PER_ROW];
    __shared__ __align__(16) uint8_t SFB_smem[STAGES][BLOCK_N * SF_K_PER_ROW];

    // ── Thread indexing ──
    const int t       = threadIdx.x;
    const int warp_id = t / 32;
    const int lane    = t % 32;
    const int l       = lane % 4;
    const int h       = lane / 4;

    // ── Tile indexing ──
    const int N_TILES = QKVD / BLOCK_N;            // 32
    int m_idx = blockIdx.x / N_TILES;
    int n_idx = blockIdx.x % N_TILES;
    int m_base = m_idx * BLOCK_M;
    int n_base = n_idx * BLOCK_N;                  // global column start

    if (m_base >= BS || n_base >= QKVD) return;

    // Determine tile type
    bool is_q = (n_idx < NH);                              // tiles 0..15
    bool is_k = (n_idx >= NH && n_idx < NH + NKV);         // tiles 16..23
    bool is_v = (n_idx >= NH + NKV);                       // tiles 24..31
    int head_idx = is_q ? n_idx : (is_k ? (n_idx - NH) : (n_idx - NH - NKV));

    // Derived dims
    const int NQK = NH * HD;    // 2048
    const int KVD = NKV * HD;   // 1024
    const int K_half = D / 2;   // 512
    const int D_groups = D / 16;      // 64
    (void)QKVD;  // QKVD/16 not currently needed

    // ── Issue load: load A + B + SFA + SFB for a K-stage ──
    auto issue_load = [&](int stage, int k_base) {
        const int k_byte_off = k_base / 2;

        // A (activation): [BS, D] FP4
        {
            int row_a = t / 2;
            int boff  = (t % 2) * 16;
            int m_glob = m_base + row_a;
            const uint8_t* src = nullptr;
            if (m_glob < BS && k_base < D) {
                src = act_packed + (m_glob * K_half + k_byte_off + boff);
            }
            cp_async_16(to_smem(&A_smem[stage][row_a * SMEM_K_STRIDE + boff]), src);
        }

        // B (weight): [QKVD, D] FP4 — each thread loads one full row (32 bytes via 2×cp_async)
        {
            int row = t;
            if (row < BLOCK_N) {
                int m_glob = n_base + row;
                const uint8_t* src = nullptr;
                if (m_glob < QKVD && k_base < D) {
                    src = w_packed + (m_glob * K_half + k_byte_off);
                }
                cp_async_16(to_smem(&B_smem[stage][row * SMEM_K_STRIDE + 0]), src);
                cp_async_16(to_smem(&B_smem[stage][row * SMEM_K_STRIDE + 16]),
                            src ? src + 16 : nullptr);
            }
        }

        // SFA: swizzled, 4 contiguous bytes
        if (t < BLOCK_M) {
            int m_glob = m_base + t;
            uint32_t sf_packed = 0;
            if (m_glob < BS) {
                int kg = k_base / 16;
                int base = swizzled_sf_base(m_glob, kg, D_groups);
                sf_packed = *reinterpret_cast<const uint32_t*>(&act_sf[base]);
            }
            *reinterpret_cast<uint32_t*>(&SFA_smem[stage][t * SF_K_PER_ROW]) = sf_packed;
        }

        // SFB: swizzled, 4 contiguous bytes per row
        // Each of 128 threads loads SFB for one row (BLOCK_N=128)
        {
            int n_glob = n_base + t;
            uint32_t sf_packed = 0;
            if (n_glob < QKVD) {
                int kg = k_base / 16;
                int base = swizzled_sf_base(n_glob, kg, D_groups);
                sf_packed = *reinterpret_cast<const uint32_t*>(&w_sf[base]);
            }
            *reinterpret_cast<uint32_t*>(&SFB_smem[stage][t * SF_K_PER_ROW]) = sf_packed;
        }
    };

    // ── Accumulators ──
    float dA[N_ATOMS] = {0}, dB[N_ATOMS] = {0};
    float dC[N_ATOMS] = {0}, dD[N_ATOMS] = {0};

    // ── Main loop ──
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;
    const int K_iters = D / BLOCK_K;

    for (int k_iter = 0; k_iter < K_iters; ++k_iter) {
        int next_stage = compute_stage ^ 1;
        int k_next = (k_iter + 1) * BLOCK_K;
        if (k_next < D) issue_load(next_stage, k_next);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        const int warp_M_off = warp_id * 16;
        const int kA0 = 4 * l;
        const int kA2 = 4 * l + 16;

        int rA0 = warp_M_off + h;
        int rA1 = warp_M_off + h + 8;
        uint32_t A0 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA0 * SMEM_K_STRIDE + kA0]);
        uint32_t A1 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA1 * SMEM_K_STRIDE + kA0]);
        uint32_t A2 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA0 * SMEM_K_STRIDE + kA2]);
        uint32_t A3 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA1 * SMEM_K_STRIDE + kA2]);

        // SFA row selection
        int sfa_m_row;
        if ((lane & 3) == 1) sfa_m_row = warp_M_off + h + 8;
        else                 sfa_m_row = warp_M_off + h;
        uint32_t SFA_val = *reinterpret_cast<const uint32_t*>(
            &SFA_smem[compute_stage][sfa_m_row * SF_K_PER_ROW]);

        #pragma unroll
        for (int g = 0; g < N_GROUPS; ++g) {
            int base = g * 4;
            int co0 = (base + 0) * 8 + h;
            int co1 = (base + 1) * 8 + h;
            int co2 = (base + 2) * 8 + h;
            int co3 = (base + 3) * 8 + h;

            // B operands (8 uint32 per group)
            uint32_t B0 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co0 * SMEM_K_STRIDE + kA0]);
            uint32_t B1 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co0 * SMEM_K_STRIDE + kA2]);
            uint32_t B2 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co1 * SMEM_K_STRIDE + kA0]);
            uint32_t B3 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co1 * SMEM_K_STRIDE + kA2]);
            uint32_t B4 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co2 * SMEM_K_STRIDE + kA0]);
            uint32_t B5 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co2 * SMEM_K_STRIDE + kA2]);
            uint32_t B6 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co3 * SMEM_K_STRIDE + kA0]);
            uint32_t B7 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co3 * SMEM_K_STRIDE + kA2]);

            // SFB for this N-group
            int sfb_n = g * 32 + l * 8 + h;
            uint32_t SFB_val = *reinterpret_cast<const uint32_t*>(
                &SFB_smem[compute_stage][sfb_n * SF_K_PER_ROW]);

            mma_nvfp4_4atom(
                dA[base+0], dB[base+0], dC[base+0], dD[base+0],
                dA[base+1], dB[base+1], dC[base+1], dD[base+1],
                dA[base+2], dB[base+2], dC[base+2], dD[base+2],
                dA[base+3], dB[base+3], dC[base+3], dD[base+3],
                A0, A1, A2, A3,
                B0, B1, B2, B3, B4, B5, B6, B7,
                SFA_val, SFB_val);
        }

        compute_stage = next_stage;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // ── Epilogue: RMSNorm + RoPE → output ──
    // Each thread in the warp holds 4 values per atom (dA, dB, dC, dD)
    // Mapping: for atom a, l ∈ {0,1,2,3}:
    //   dA[a] → row (warp_M_off + h),       col (n_base + a*8 + 2*l)
    //   dB[a] → row (warp_M_off + h),       col (n_base + a*8 + 2*l + 1)
    //   dC[a] → row (warp_M_off + h + 8),   col (n_base + a*8 + 2*l)
    //   dD[a] → row (warp_M_off + h + 8),   col (n_base + a*8 + 2*l + 1)

    const int warp_M_off = warp_id * 16;

    if (is_v) {
        // ── V tile: direct f32→bf16 write ──
        // v_flat layout: [BS*NKV, HD], indexed as (row*NKV + head_idx)*HD + col
        // col is head-local (0..HD-1), computed from atom index and lane
        #pragma unroll
        for (int a = 0; a < N_ATOMS; ++a) {
            int col_local = a * 8 + 2 * l;  // 0..126, within-head column
            int row0 = m_base + warp_M_off + h;
            int row1 = m_base + warp_M_off + h + 8;

            if (row0 < BS) {
                int idx0 = (row0 * NKV + head_idx) * HD + col_local;
                v_out[idx0]     = __float2bfloat16(dA[a]);
                v_out[idx0 + 1] = __float2bfloat16(dB[a]);
            }
            if (row1 < BS) {
                int idx1 = (row1 * NKV + head_idx) * HD + col_local;
                v_out[idx1]     = __float2bfloat16(dC[a]);
                v_out[idx1 + 1] = __float2bfloat16(dD[a]);
            }
        }
    } else {
        // ── Q/K tile: RMSNorm + RoPE (v2: register-based RoPE) ──
        // Key insight: paired columns (col ↔ col+64) map to atom a ↔ a+8,
        // both in the same thread's registers. No shared memory needed for RoPE.

        // Step 1: compute local sum of squares per row (same as v1)
        float local_sum0 = 0.0f, local_sum1 = 0.0f;
        #pragma unroll
        for (int a = 0; a < N_ATOMS; ++a) {
            local_sum0 += dA[a]*dA[a] + dB[a]*dB[a];
            local_sum1 += dC[a]*dC[a] + dD[a]*dD[a];
        }

        // Step 2: warp reduce within h-group (4 threads with same h=lane/4)
        unsigned mask = 0xFu << (h * 4);
        local_sum0 += __shfl_down_sync(mask, local_sum0, 1);
        local_sum0 += __shfl_down_sync(mask, local_sum0, 2);
        local_sum1 += __shfl_down_sync(mask, local_sum1, 1);
        local_sum1 += __shfl_down_sync(mask, local_sum1, 2);

        // Only thread with l==0 writes the reduced sum to shared memory
        __shared__ float smem_rms[BLOCK_M];
        int row0 = warp_M_off + h;
        int row1 = warp_M_off + h + 8;
        if (l == 0) {
            smem_rms[row0] = local_sum0;
            smem_rms[row1] = local_sum1;
        }
        __syncthreads();

        // Step 3: read rms values (all threads)
        float rms0 = rsqrtf(smem_rms[row0] / HD + eps);
        float rms1 = rsqrtf(smem_rms[row1] / HD + eps);

        // Setup output pointers
        int global_row0 = m_base + row0;
        int global_row1 = m_base + row1;
        bool valid0 = (global_row0 < BS);
        bool valid1 = (global_row1 < BS);
        if (!valid0 && !valid1) return;  // nothing to do for this block

        const __nv_bfloat16* nw = is_q ? q_weight : k_weight;
        int out_stride = is_q ? NQK : KVD;
        __nv_bfloat16* out_base = is_q ? q_out : k_out;
        int out_col_offset = head_idx * HD;

        // Step 4: Apply RMSNorm + q_weight/k_weight, then RoPE
        // cos[col] == cos[col+64], sin[col] == sin[col+64] — reuse for pairs
        // RoPE formula (matching fused_qk_norm_rope_v4):
        //   low  (col < 64):  result = bf16(bf16(-partner * sin) + xv * cos)
        //   high (col >= 64): result = bf16(bf16(xv * sin) + partner * cos)

        #pragma unroll
        for (int a = 0; a < 8; ++a) {
            // Column offsets for this atom pair
            int col_even = a * 8 + 2 * l;
            int col_odd  = col_even + 1;
            int col_even_high = col_even + 64;
            int col_odd_high  = col_odd + 64;

            // cos/sin: same for paired columns (cos[c]==cos[c+64])
            float c_even = 0, s_even = 0, c_odd = 0, s_odd = 0;
            float c_even1 = 0, s_even1 = 0, c_odd1 = 0, s_odd1 = 0;
            if (valid0) {
                c_even = __bfloat162float(cos[global_row0 * HD + col_even]);
                s_even = __bfloat162float(sin[global_row0 * HD + col_even]);
                c_odd  = __bfloat162float(cos[global_row0 * HD + col_odd]);
                s_odd  = __bfloat162float(sin[global_row0 * HD + col_odd]);
            }
            if (valid1) {
                c_even1 = __bfloat162float(cos[global_row1 * HD + col_even]);
                s_even1 = __bfloat162float(sin[global_row1 * HD + col_even]);
                c_odd1  = __bfloat162float(cos[global_row1 * HD + col_odd]);
                s_odd1  = __bfloat162float(sin[global_row1 * HD + col_odd]);
            }

            // Norm weights (same for all rows since per-head, not per-row)
            float nw_even = __bfloat162float(nw[col_even]);
            float nw_odd  = __bfloat162float(nw[col_odd]);
            float nw_even_h = __bfloat162float(nw[col_even_high]);
            float nw_odd_h  = __bfloat162float(nw[col_odd_high]);

            // ── Row 0 ──
            if (valid0) {
                // Apply RMSNorm + weight scaling
                float xv_even  = dA[a]     * rms0 * nw_even;
                float xv_odd   = dB[a]     * rms0 * nw_odd;
                float pt_even  = dA[a + 8] * rms0 * nw_even_h;
                float pt_odd   = dB[a + 8] * rms0 * nw_odd_h;

                // RoPE low half (col < 64): rot = -partner
                float rot_sin_even = __bfloat162float(__float2bfloat16(-pt_even * s_even));
                float rot_sin_odd  = __bfloat162float(__float2bfloat16(-pt_odd * s_odd));
                float r_low_even = __float2bfloat16(rot_sin_even + xv_even * c_even);
                float r_low_odd  = __float2bfloat16(rot_sin_odd + xv_odd * c_odd);

                // RoPE high half (col >= 64): rot = +xv
                float rot_sin_even_h = __bfloat162float(__float2bfloat16(xv_even * s_even));
                float rot_sin_odd_h  = __bfloat162float(__float2bfloat16(xv_odd * s_odd));
                float r_high_even = __float2bfloat16(rot_sin_even_h + pt_even * c_even);
                float r_high_odd  = __float2bfloat16(rot_sin_odd_h + pt_odd * c_odd);

                // Write row 0
                out_base[global_row0 * out_stride + out_col_offset + col_even]      = r_low_even;
                out_base[global_row0 * out_stride + out_col_offset + col_odd]       = r_low_odd;
                out_base[global_row0 * out_stride + out_col_offset + col_even_high] = r_high_even;
                out_base[global_row0 * out_stride + out_col_offset + col_odd_high]  = r_high_odd;
            }

            // ── Row 1 ──
            if (valid1) {
                // Apply RMSNorm + weight scaling
                float xv_even1  = dC[a]     * rms1 * nw_even;
                float xv_odd1   = dD[a]     * rms1 * nw_odd;
                float pt_even1  = dC[a + 8] * rms1 * nw_even_h;
                float pt_odd1   = dD[a + 8] * rms1 * nw_odd_h;

                // RoPE low half
                float rot_sin_even1 = __bfloat162float(__float2bfloat16(-pt_even1 * s_even1));
                float rot_sin_odd1  = __bfloat162float(__float2bfloat16(-pt_odd1 * s_odd1));
                float r_low_even1 = __float2bfloat16(rot_sin_even1 + xv_even1 * c_even1);
                float r_low_odd1  = __float2bfloat16(rot_sin_odd1 + xv_odd1 * c_odd1);

                // RoPE high half
                float rot_sin_even_h1 = __bfloat162float(__float2bfloat16(xv_even1 * s_even1));
                float rot_sin_odd_h1  = __bfloat162float(__float2bfloat16(xv_odd1 * s_odd1));
                float r_high_even1 = __float2bfloat16(rot_sin_even_h1 + pt_even1 * c_even1);
                float r_high_odd1  = __float2bfloat16(rot_sin_odd_h1 + pt_odd1 * c_odd1);

                // Write row 1
                out_base[global_row1 * out_stride + out_col_offset + col_even]      = r_low_even1;
                out_base[global_row1 * out_stride + out_col_offset + col_odd]       = r_low_odd1;
                out_base[global_row1 * out_stride + out_col_offset + col_even_high] = r_high_even1;
                out_base[global_row1 * out_stride + out_col_offset + col_odd_high]  = r_high_odd1;
            }
        }
    }
}

}  // anonymous namespace

void omnivoice_qkv_norm_rope_sm120(
    const uint8_t* act_packed,
    const uint8_t* act_sf,
    const uint8_t* w_packed,
    const uint8_t* w_sf,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    int BS, int D, int QKVD,
    int NH, int NKV, int HD,
    float eps,
    cudaStream_t stream)
{
    if (BS <= 0 || D <= 0 || QKVD <= 0) return;
    if (QKVD % BLOCK_N != 0) {
        fprintf(stderr, "[qkv_norm_rope] QKVD=%d not divisible by BLOCK_N=%d\n", QKVD, BLOCK_N);
        return;
    }

    int M_tiles = (BS + BLOCK_M - 1) / BLOCK_M;
    int N_tiles = QKVD / BLOCK_N;
    int total_tiles = M_tiles * N_tiles;

    dim3 grid(total_tiles);
    dim3 block(THREADS);

    qkv_norm_rope_kernel<<<grid, block, 0, stream>>>(
        act_packed, act_sf, w_packed, w_sf,
        q_weight, k_weight, cos, sin,
        q_out, k_out, v_out,
        BS, D, QKVD, NH, NKV, HD, eps);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[qkv_norm_rope] launch err: %s\n", cudaGetErrorString(e));
    }
}

}  // namespace megakernel
}  // namespace flash_rt
