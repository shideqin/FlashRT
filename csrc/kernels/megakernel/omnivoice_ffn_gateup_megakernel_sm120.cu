// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice FFN GateUp Megakernel v4 — Optimized dual-B (Agent B)
//
// Single-kernel fusion: GateUp GEMM + SiLU+Mul + NVFP4 quantize.
// Dual B matrices (gate + up) with optimized SF loading:
//   - For K=1024, BLOCK_K=64, kg values within a K-block are always
//     [kg, kg+1, kg+2, kg+3] where kg%4==0 → contiguous swizzled SF indices.
//   - Single swizzled_sf_idx call per row per K-block, then read 4 contiguous bytes.
//
// FlashRT native swizzled SF format throughout.

#include "megakernel/omnivoice_ffn_gateup_megakernel_sm120.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace flash_rt {
namespace megakernel {
namespace {

constexpr int BLOCK_M      = 64;
constexpr int BLOCK_N      = 64;
constexpr int BLOCK_K      = 64;
constexpr int NUM_WARPS    = 4;
constexpr int THREADS      = NUM_WARPS * 32;   // 128
constexpr int STAGES       = 2;

constexpr int N_ATOMS      = BLOCK_N / 8;       // 8
constexpr int N_GROUPS     = N_ATOMS / 4;       // 2

constexpr int SMEM_K_STRIDE = BLOCK_K / 2 + 16;  // 48
constexpr int SF_K_PER_ROW  = BLOCK_K / 16;      // 4

// FlashRT swizzled SF index for a given (row, kg).
// For our use case, kg is always a multiple of 4 (k_base/16 with BLOCK_K=64).
__device__ __forceinline__ int swizzled_sf_base(
    int row, int kg, int total_groups)
{
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

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint8_t mag;
    if      (a < 0.25f)  mag = 0;
    else if (a < 0.75f)  mag = 1;
    else if (a < 1.25f)  mag = 2;
    else if (a < 1.75f)  mag = 3;
    else if (a < 2.5f)   mag = 4;
    else if (a < 3.5f)   mag = 5;
    else if (a < 5.0f)   mag = 6;
    else                 mag = 7;
    return sign | mag;
}

__device__ __forceinline__ uint8_t float_to_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    uint32_t bits = __float_as_uint(v);
    int exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;
    if (exp < -9) return 0;
    if (exp > 7) return 0x7F;
    uint8_t result;
    if (exp < 0) {
        int shift = -exp;
        uint32_t mant = (frac | 0x800000) >> (23 + shift - 2);
        uint32_t trunc_mask = (1u << (23 + shift - 2)) - 1;
        if ((frac & trunc_mask) != 0) mant += 1;
        result = (mant & 0x3);
    } else {
        uint32_t trunc_mask = (1u << 21) - 1;
        uint32_t mant2 = (frac >> 21) & 0x3;
        if ((frac & trunc_mask) != 0) {
            mant2 += 1;
            if (mant2 > 3) { mant2 = 0; exp += 1; }
        }
        if (exp > 7) return 0x7F;
        result = (exp << 2) | mant2;
    }
    return result;
}

// ── Main kernel: dual-B with optimized SF loading ──

__global__ void __launch_bounds__(THREADS, 2)
gateup_megakernel(
    const uint8_t* __restrict__ inp_packed,
    const uint8_t* __restrict__ inp_sfa,
    const uint8_t* __restrict__ gu_packed,
    const uint8_t* __restrict__ gu_sfb,
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_sfa,
    int M, int FFN, int K,
    float alpha)
{
    __shared__ __align__(16) uint8_t A_smem   [STAGES][BLOCK_M * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t Bg_smem  [STAGES][BLOCK_N * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t Bu_smem  [STAGES][BLOCK_N * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t SFA_smem [STAGES][BLOCK_M * SF_K_PER_ROW];
    __shared__ __align__(16) uint8_t SFBg_smem[STAGES][BLOCK_N * SF_K_PER_ROW];
    __shared__ __align__(16) uint8_t SFBu_smem[STAGES][BLOCK_N * SF_K_PER_ROW];

    const int t       = threadIdx.x;
    const int warp_id = t / 32;
    const int lane    = t % 32;
    const int l       = lane % 4;
    const int h       = lane / 4;

    const int K_groups   = K / 16;       // 64
    const int K_half     = K / 2;        // 512
    const int FFN_half   = FFN / 2;      // 1536
    const int FFN_groups = FFN / 16;     // 192
    const int GU_N       = 2 * FFN;      // 6144

    int n_tiles = (FFN + BLOCK_N - 1) / BLOCK_N;
    int m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    int m_idx  = blockIdx.x / n_tiles;
    int n_idx  = blockIdx.x % n_tiles;
    int m_base = m_idx * BLOCK_M;
    int n_base = n_idx * BLOCK_N;
    if (m_base >= M || n_base >= FFN) return;

    // Optimized SF loading: for BLOCK_K=64, kg values [kg, kg+1, kg+2, kg+3]
    // are all in the same cb group (kg%4==0 for k_base multiples of 64).
    // So we compute ONE swizzled base index and read 4 contiguous bytes.
    auto load_sf4_swizzled = [&](const uint8_t* sf_ptr, int row, int kg_start,
                                  int total_groups, uint8_t* dst) {
        int base = swizzled_sf_base(row, kg_start, total_groups);
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(&sf_ptr[base]);
    };

    auto issue_load = [&](int stage, int k_base) {
        const int k_byte_off = k_base / 2;

        // A: row-major packed data
        {
            int row_a = t / 2;
            int boff  = (t % 2) * 16;
            int m_glob = m_base + row_a;
            const uint8_t* src = nullptr;
            if (m_glob < M && k_base < K) {
                src = inp_packed + (m_glob * K_half + k_byte_off + boff);
            }
            cp_async_16(to_smem(&A_smem[stage][row_a * SMEM_K_STRIDE + boff]), src);
        }

        // B_gate: row-major
        {
            int row_b = t / 2;
            int boff  = (t % 2) * 16;
            int n_glob = n_base + row_b;
            const uint8_t* src = nullptr;
            if (n_glob < FFN && k_base < K) {
                src = gu_packed + (n_glob * K_half + k_byte_off + boff);
            }
            cp_async_16(to_smem(&Bg_smem[stage][row_b * SMEM_K_STRIDE + boff]), src);
        }

        // B_up: row-major, offset by FFN rows
        {
            int row_b = t / 2;
            int boff  = (t % 2) * 16;
            int n_glob = n_base + row_b + FFN;
            const uint8_t* src = nullptr;
            if (n_glob < GU_N && k_base < K) {
                src = gu_packed + (n_glob * K_half + k_byte_off + boff);
            }
            cp_async_16(to_smem(&Bu_smem[stage][row_b * SMEM_K_STRIDE + boff]), src);
        }

        // SFA: swizzled, 4 contiguous bytes (optimized)
        if (t < BLOCK_M) {
            int m_glob = m_base + t;
            uint32_t sf_packed = 0;
            if (m_glob < M) {
                int kg = k_base / 16;
                int base = swizzled_sf_base(m_glob, kg, K_groups);
                sf_packed = *reinterpret_cast<const uint32_t*>(&inp_sfa[base]);
            }
            *reinterpret_cast<uint32_t*>(&SFA_smem[stage][t * SF_K_PER_ROW]) = sf_packed;
        }

        // SFB_gate: swizzled, 4 contiguous bytes
        if (t >= 64 && t < 64 + BLOCK_N) {
            int n_glob = n_base + (t - 64);
            uint32_t sf_packed = 0;
            if (n_glob < FFN) {
                int kg = k_base / 16;
                int base = swizzled_sf_base(n_glob, kg, K_groups);
                sf_packed = *reinterpret_cast<const uint32_t*>(&gu_sfb[base]);
            }
            *reinterpret_cast<uint32_t*>(&SFBg_smem[stage][(t - 64) * SF_K_PER_ROW]) = sf_packed;
        }

        // SFB_up: swizzled, offset by FFN rows
        if (t < BLOCK_N) {
            int n_glob = n_base + t + FFN;
            uint32_t sf_packed = 0;
            if (n_glob < GU_N) {
                int kg = k_base / 16;
                int base = swizzled_sf_base(n_glob, kg, K_groups);
                sf_packed = *reinterpret_cast<const uint32_t*>(&gu_sfb[base]);
            }
            *reinterpret_cast<uint32_t*>(&SFBu_smem[stage][t * SF_K_PER_ROW]) = sf_packed;
        }
    };

    // Accumulators
    float dA_gate[N_ATOMS] = {0}, dB_gate[N_ATOMS] = {0};
    float dC_gate[N_ATOMS] = {0}, dD_gate[N_ATOMS] = {0};
    float dA_up[N_ATOMS]   = {0}, dB_up[N_ATOMS]   = {0};
    float dC_up[N_ATOMS]   = {0}, dD_up[N_ATOMS]   = {0};

    // Main loop
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;
    const int K_iters = K / BLOCK_K;

    for (int k_iter = 0; k_iter < K_iters; ++k_iter) {
        int next_stage = compute_stage ^ 1;
        int k_next = (k_iter + 1) * BLOCK_K;
        if (k_next < K) issue_load(next_stage, k_next);
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

            // Gate
            uint32_t B0g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co0 * SMEM_K_STRIDE + kA0]);
            uint32_t B1g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co0 * SMEM_K_STRIDE + kA2]);
            uint32_t B2g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co1 * SMEM_K_STRIDE + kA0]);
            uint32_t B3g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co1 * SMEM_K_STRIDE + kA2]);
            uint32_t B4g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co2 * SMEM_K_STRIDE + kA0]);
            uint32_t B5g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co2 * SMEM_K_STRIDE + kA2]);
            uint32_t B6g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co3 * SMEM_K_STRIDE + kA0]);
            uint32_t B7g = *reinterpret_cast<const uint32_t*>(
                &Bg_smem[compute_stage][co3 * SMEM_K_STRIDE + kA2]);

            int sfb_n = g * 32 + l * 8 + h;
            uint32_t SFB_gate = *reinterpret_cast<const uint32_t*>(
                &SFBg_smem[compute_stage][sfb_n * SF_K_PER_ROW]);

            mma_nvfp4_4atom(
                dA_gate[base+0], dB_gate[base+0], dC_gate[base+0], dD_gate[base+0],
                dA_gate[base+1], dB_gate[base+1], dC_gate[base+1], dD_gate[base+1],
                dA_gate[base+2], dB_gate[base+2], dC_gate[base+2], dD_gate[base+2],
                dA_gate[base+3], dB_gate[base+3], dC_gate[base+3], dD_gate[base+3],
                A0, A1, A2, A3,
                B0g, B1g, B2g, B3g, B4g, B5g, B6g, B7g,
                SFA_val, SFB_gate);

            // Up
            uint32_t B0u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co0 * SMEM_K_STRIDE + kA0]);
            uint32_t B1u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co0 * SMEM_K_STRIDE + kA2]);
            uint32_t B2u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co1 * SMEM_K_STRIDE + kA0]);
            uint32_t B3u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co1 * SMEM_K_STRIDE + kA2]);
            uint32_t B4u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co2 * SMEM_K_STRIDE + kA0]);
            uint32_t B5u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co2 * SMEM_K_STRIDE + kA2]);
            uint32_t B6u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co3 * SMEM_K_STRIDE + kA0]);
            uint32_t B7u = *reinterpret_cast<const uint32_t*>(
                &Bu_smem[compute_stage][co3 * SMEM_K_STRIDE + kA2]);

            uint32_t SFB_up = *reinterpret_cast<const uint32_t*>(
                &SFBu_smem[compute_stage][sfb_n * SF_K_PER_ROW]);

            mma_nvfp4_4atom(
                dA_up[base+0], dB_up[base+0], dC_up[base+0], dD_up[base+0],
                dA_up[base+1], dB_up[base+1], dC_up[base+1], dD_up[base+1],
                dA_up[base+2], dB_up[base+2], dC_up[base+2], dD_up[base+2],
                dA_up[base+3], dB_up[base+3], dC_up[base+3], dD_up[base+3],
                A0, A1, A2, A3,
                B0u, B1u, B2u, B3u, B4u, B5u, B6u, B7u,
                SFA_val, SFB_up);
        }

        compute_stage = next_stage;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // ── Epilogue: SiLU(gate) * up + NVFP4 quantize ──
    const int warp_M_off = warp_id * 16;
    #pragma unroll
    for (int n_atom = 0; n_atom < N_ATOMS; ++n_atom) {
        int col_pair = n_base + n_atom * 8 + 2 * l;
        int row0 = m_base + warp_M_off + h;
        int row1 = m_base + warp_M_off + h + 8;

        float g00 = dA_gate[n_atom] * alpha;
        float g01 = dB_gate[n_atom] * alpha;
        float g10 = dC_gate[n_atom] * alpha;
        float g11 = dD_gate[n_atom] * alpha;
        float u00 = dA_up[n_atom] * alpha;
        float u01 = dB_up[n_atom] * alpha;
        float u10 = dC_up[n_atom] * alpha;
        float u11 = dD_up[n_atom] * alpha;

        float v00 = __bfloat162float(__float2bfloat16(silu_f32(g00))) * u00;
        float v01 = __bfloat162float(__float2bfloat16(silu_f32(g01))) * u01;
        float v10 = __bfloat162float(__float2bfloat16(silu_f32(g10))) * u10;
        float v11 = __bfloat162float(__float2bfloat16(silu_f32(g11))) * u11;

        if (row0 < M && col_pair < FFN) {
            int out_byte_off = row0 * FFN_half + col_pair / 2;
            if (col_pair % 2 == 0) {
                uint8_t lo = float_to_fp4_e2m1(v00);
                uint8_t hi = float_to_fp4_e2m1(v01);
                out_packed[out_byte_off] = lo | (hi << 4);
            }
        }
        if (row1 < M && col_pair < FFN) {
            int out_byte_off = row1 * FFN_half + col_pair / 2;
            if (col_pair % 2 == 0) {
                uint8_t lo = float_to_fp4_e2m1(v10);
                uint8_t hi = float_to_fp4_e2m1(v11);
                out_packed[out_byte_off] = lo | (hi << 4);
            }
        }

        int sf_group = col_pair / 16;
        if (col_pair % 16 == 0 && sf_group < FFN_groups) {
            float amax0 = fmaxf(fabsf(v00), fabsf(v01));
            float amax1 = fmaxf(fabsf(v10), fabsf(v11));
            float amax = fmaxf(amax0, amax1);
            float desired_scale = amax / 6.0f;
            if (desired_scale < 1e-12f) desired_scale = 1e-12f;
            uint8_t sf_byte = float_to_ue4m3_ceil(desired_scale);

            if (row0 < M) {
                int sf_idx = swizzled_sf_base(row0, sf_group, FFN_groups);
                out_sfa[sf_idx] = sf_byte;
            }
            if (row1 < M) {
                int sf_idx = swizzled_sf_base(row1, sf_group, FFN_groups);
                out_sfa[sf_idx] = sf_byte;
            }
        }
    }
}

}  // anonymous namespace

int omnivoice_ffn_gateup_megakernel_sm120(
    const void* inp_packed,  const void* inp_sfa,
    const void* gu_packed,   const void* gu_sfb,
    void*       out_packed,  void*       out_sfa,
    int M, int FFN, int K,
    float alpha,
    cudaStream_t stream)
{
    if (FFN != 3072 || K != 1024) {
        fprintf(stderr, "[omnivoice_ffn_gateup] FFN=%d K=%d (expected 3072,1024)\n", FFN, K);
        return -1;
    }
    if (M <= 0) return -2;

    int n_tiles = (FFN + BLOCK_N - 1) / BLOCK_N;   // 48
    int m_tiles = (M + BLOCK_M - 1) / BLOCK_M;      // 6
    int total_tiles = m_tiles * n_tiles;             // 288

    dim3 grid(total_tiles);
    dim3 block(THREADS);

    gateup_megakernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(inp_packed),
        reinterpret_cast<const uint8_t*>(inp_sfa),
        reinterpret_cast<const uint8_t*>(gu_packed),
        reinterpret_cast<const uint8_t*>(gu_sfb),
        reinterpret_cast<uint8_t*>(out_packed),
        reinterpret_cast<uint8_t*>(out_sfa),
        M, FFN, K, alpha);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[omnivoice_ffn_gateup] launch err: %s\n",
                cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

}  // namespace megakernel
}  // namespace flash_rt
