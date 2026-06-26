// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice FFN GateUp+SiLU Megakernel v6 — Interleaved-B (Agent B)
//
// Single-kernel fusion: GateUp GEMM + SiLU+Mul + NVFP4 quantize.
// Uses INTERLEAVED weight layout: B[2*i]=gate[i], B[2*i+1]=up[i].
// This co-locates gate/up pairs in the same accumulator tile.
//
// Unlike v4 (dual-B, BLOCK_N=64): single B matrix, single MMA, BLOCK_N=128.
// The epilogue naturally pairs adjacent columns as gate/up.

#include "megakernel/omnivoice_ffn_gateup_silu_interleaved_sm120.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace flash_rt {
namespace megakernel {
namespace {

constexpr int BLOCK_M      = 64;
constexpr int BLOCK_N      = 128;  // 64 gate/up pairs in interleaved layout
constexpr int BLOCK_K      = 64;
constexpr int NUM_WARPS    = 4;
constexpr int THREADS      = NUM_WARPS * 32;   // 128
constexpr int STAGES       = 2;

constexpr int N_ATOMS      = BLOCK_N / 8;       // 16 (shared across all warps)
constexpr int N_GROUPS     = N_ATOMS / 4;       // 4

constexpr int SMEM_K_STRIDE = BLOCK_K / 2 + 16;  // 48
constexpr int SF_K_PER_ROW  = BLOCK_K / 16;      // 4

// ── Helpers (same as V28 v4) ──

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

// ── Main kernel: interleaved-B, single B matrix, single MMA per K-iteration ──

__global__ void __launch_bounds__(THREADS, 2)
gateup_silu_interleaved_kernel(
    const uint8_t* __restrict__ inp_packed,
    const uint8_t* __restrict__ inp_sfa,
    const uint8_t* __restrict__ gu_packed,
    const uint8_t* __restrict__ gu_sfb,
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_sfa,
    int M, int FFN, int K,
    float alpha)
{
    // Shared memory: single B (interleaved), no dual-B
    __shared__ __align__(16) uint8_t A_smem  [STAGES][BLOCK_M * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t B_smem  [STAGES][BLOCK_N * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t SFA_smem[STAGES][BLOCK_M * SF_K_PER_ROW];
    __shared__ __align__(16) uint8_t SFB_smem[STAGES][BLOCK_N * SF_K_PER_ROW];

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

    int n_tiles = (GU_N + BLOCK_N - 1) / BLOCK_N;  // 48
    int m_idx  = blockIdx.x / n_tiles;
    int n_idx  = blockIdx.x % n_tiles;
    int m_base = m_idx * BLOCK_M;
    int n_base = n_idx * BLOCK_N;
    if (m_base >= M || n_base >= GU_N) return;

    // ── Issue load ──
    auto issue_load = [&](int stage, int k_base) {
        const int k_byte_off = k_base / 2;

        // A: 64 rows, 128 threads → 2 threads per row, each loads 16 bytes
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

        // B: 128 rows, 128 threads → 1 thread per row, loads 2×16 bytes
        {
            int row_b = t;
            int n_glob = n_base + row_b;
            const uint8_t* src0 = nullptr;
            const uint8_t* src1 = nullptr;
            if (n_glob < GU_N && k_base < K) {
                src0 = gu_packed + (n_glob * K_half + k_byte_off);
                src1 = gu_packed + (n_glob * K_half + k_byte_off + 16);
            }
            cp_async_16(to_smem(&B_smem[stage][row_b * SMEM_K_STRIDE + 0]),  src0);
            cp_async_16(to_smem(&B_smem[stage][row_b * SMEM_K_STRIDE + 16]), src1);
        }

        // SFA
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

        // SFB
        if (t < BLOCK_N) {
            int n_glob = n_base + t;
            uint32_t sf_packed = 0;
            if (n_glob < GU_N) {
                int kg = k_base / 16;
                int base = swizzled_sf_base(n_glob, kg, K_groups);
                sf_packed = *reinterpret_cast<const uint32_t*>(&gu_sfb[base]);
            }
            *reinterpret_cast<uint32_t*>(&SFB_smem[stage][t * SF_K_PER_ROW]) = sf_packed;
        }
    };

    // ── Accumulators (single set, interleaved gate/up in adjacent columns) ──
    float dA[N_ATOMS] = {0}, dB[N_ATOMS] = {0};
    float dC[N_ATOMS] = {0}, dD[N_ATOMS] = {0};

    // ── Main loop ──
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;
    const int K_iters = K / BLOCK_K;  // 16

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

            // Single B matrix load (not dual-B)
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

            int sfb_n = g * 32 + l * 8 + h;
            uint32_t SFB_val = *reinterpret_cast<const uint32_t*>(
                &SFB_smem[compute_stage][sfb_n * SF_K_PER_ROW]);

            // Single MMA call (not dual)
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

    // ── Epilogue: SiLU(gate) * up for interleaved columns ──
    // Interleaved layout: accumulator columns [2*i]=gate, [2*i+1]=up
    // dA[n_atom] = gate at interleaved column (n_base + n_atom*8 + 2*l)
    // dB[n_atom] = up   at interleaved column (n_base + n_atom*8 + 2*l + 1)
    // dC[n_atom] = gate at same columns, row+8
    // dD[n_atom] = up   at same columns, row+8
    //
    // Output column: interleaved_col / 2 = (n_base + n_atom*8 + 2*l) / 2
    //               = n_base/2 + n_atom*4 + l

    const int warp_M_off = warp_id * 16;
    #pragma unroll
    for (int n_atom = 0; n_atom < N_ATOMS; ++n_atom) {
        // Output column for this thread's gate/up pair
        int out_col = n_base / 2 + n_atom * 4 + l;
        int row0 = m_base + warp_M_off + h;
        int row1 = m_base + warp_M_off + h + 8;

        // Gate and up from adjacent accumulator columns
        float gate0 = dA[n_atom] * alpha;
        float up0   = dB[n_atom] * alpha;
        float gate1 = dC[n_atom] * alpha;
        float up1   = dD[n_atom] * alpha;

        float v0 = __bfloat162float(__float2bfloat16(silu_f32(gate0))) * up0;
        float v1 = __bfloat162float(__float2bfloat16(silu_f32(gate1))) * up1;

        // Pack two adjacent output columns into one byte (standard FP4 layout)
        // Threads with even l write low nibble, odd l write high nibble
        // Use __shfl_xor_sync to pair with neighbor thread
        if (row0 < M && out_col < FFN) {
            int out_byte_off = row0 * FFN_half + out_col / 2;
            if (out_col % 2 == 0) {
                // Even column: we write low nibble, need high nibble from l+1 thread
                float v0_next = __shfl_xor_sync(0xffffffff, v0, 1);
                uint8_t lo = float_to_fp4_e2m1(v0);
                uint8_t hi = float_to_fp4_e2m1(v0_next);
                out_packed[out_byte_off] = lo | (hi << 4);
            }
        }
        if (row1 < M && out_col < FFN) {
            int out_byte_off = row1 * FFN_half + out_col / 2;
            if (out_col % 2 == 0) {
                float v1_next = __shfl_xor_sync(0xffffffff, v1, 1);
                uint8_t lo = float_to_fp4_e2m1(v1);
                uint8_t hi = float_to_fp4_e2m1(v1_next);
                out_packed[out_byte_off] = lo | (hi << 4);
            }
        }

        // SF output (same pattern as V28)
        int sf_group = out_col / 16;
        if (out_col % 16 == 0 && sf_group < FFN_groups) {
            // Compute amax across the 16 columns in this SF group
            float amax0 = fmaxf(fabsf(v0), fabsf(v1));
            // Warp reduce to get max across all threads in the same SF group
            #pragma unroll
            for (int offset = 1; offset < 16; offset <<= 1) {
                float other = __shfl_xor_sync(0xffffffff, amax0, offset);
                amax0 = fmaxf(amax0, other);
            }
            float desired_scale = amax0 / 6.0f;
            if (desired_scale < 1e-12f) desired_scale = 1e-12f;
            uint8_t sf_byte = float_to_ue4m3_ceil(desired_scale);

            if (row0 < M && (lane % 16) == 0) {
                int sf_idx = swizzled_sf_base(row0, sf_group, FFN_groups);
                out_sfa[sf_idx] = sf_byte;
            }
            if (row1 < M && (lane % 16) == 0) {
                int sf_idx = swizzled_sf_base(row1, sf_group, FFN_groups);
                out_sfa[sf_idx] = sf_byte;
            }
        }
    }
}

}  // anonymous namespace

int omnivoice_ffn_gateup_silu_interleaved_sm120(
    const void* inp_packed,  const void* inp_sfa,
    const void* gu_packed,   const void* gu_sfb,
    void*       out_packed,  void*       out_sfa,
    int M, int FFN, int K,
    float alpha,
    cudaStream_t stream)
{
    if (FFN != 3072 || K != 1024) {
        fprintf(stderr, "[omnivoice_ffn_gateup_silu_interleaved] FFN=%d K=%d (expected 3072,1024)\n", FFN, K);
        return -1;
    }
    if (M <= 0) return -2;

    int GU_N = 2 * FFN;  // 6144
    int n_tiles = (GU_N + BLOCK_N - 1) / BLOCK_N;   // 48
    int m_tiles = (M + BLOCK_M - 1) / BLOCK_M;       // 6
    int total_tiles = m_tiles * n_tiles;              // 288

    dim3 grid(total_tiles);
    dim3 block(THREADS);

    gateup_silu_interleaved_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(inp_packed),
        reinterpret_cast<const uint8_t*>(inp_sfa),
        reinterpret_cast<const uint8_t*>(gu_packed),
        reinterpret_cast<const uint8_t*>(gu_sfb),
        reinterpret_cast<uint8_t*>(out_packed),
        reinterpret_cast<uint8_t*>(out_sfa),
        M, FFN, K, alpha);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[omnivoice_ffn_gateup_silu_interleaved] launch err: %s\n",
                cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

}  // namespace megakernel
}  // namespace flash_rt
