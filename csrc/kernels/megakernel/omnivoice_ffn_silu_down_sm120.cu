// SPDX-License-Identifier: Apache-2.0
//
// OmniVoice FFN SiLU+Down Fused Kernel (V40 Agent B).
//
// Fuses SiLU(gate)*up + NVFP4 quantize + Down GEMM + residual add.
//
// Chunked approach: process FFN in 3 chunks of 1024 columns each.
// Shared memory per chunk: ~44KB (fits in default 48KB carveout).
//
// Per chunk: Pre-compute SiLU/Mul → FP4 → smem, then MMA loop.

#include "megakernel/omnivoice_ffn_silu_down_sm120.cuh"

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
constexpr int THREADS      = 128;
constexpr int STAGES       = 2;

constexpr int N_ATOMS      = BLOCK_N / 8;       // 8
constexpr int N_GROUPS     = N_ATOMS / 4;       // 2

constexpr int SMEM_K_STRIDE = BLOCK_K / 2 + 16;  // 48
constexpr int SF_K_PER_ROW  = BLOCK_K / 16;      // 4

// Chunk size: 1024 columns at a time (3 chunks for FFN=3072)
constexpr int CHUNK_K       = 1024;
constexpr int CHUNK_K_HALF  = CHUNK_K / 2;       // 512
constexpr int CHUNK_K_GROUPS = CHUNK_K / 16;     // 64
constexpr int CHUNK_ITERS   = CHUNK_K / BLOCK_K; // 16

// Chunk shared memory:
//   Act:  64 × 512  = 32,768
//   SFA:  64 × 64   =  4,096
//   B:    2 × 64 × 48 = 6,144
//   SFB:  2 × 64 × 4  =   512
//   Total: ~43.5KB
constexpr int CHUNK_ACT_SZ  = BLOCK_M * CHUNK_K_HALF;
constexpr int CHUNK_SFA_SZ  = BLOCK_M * CHUNK_K_GROUPS;
constexpr int CHUNK_B_SZ    = STAGES * BLOCK_N * SMEM_K_STRIDE;
constexpr int CHUNK_SFB_SZ  = STAGES * BLOCK_N * SF_K_PER_ROW;

// ── Helpers ──

__device__ __forceinline__ int swizzled_sf_base(int row, int kg, int total_groups) {
    int rb = row / 128, ri = row % 128, cb = kg / 4;
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

__device__ __forceinline__ float silu_f32(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u; float a = fabsf(v); uint8_t mag;
    if (a < 0.25f) mag = 0; else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2; else if (a < 1.75f) mag = 3;
    else if (a < 2.5f) mag = 4; else if (a < 3.5f) mag = 5;
    else if (a < 5.0f) mag = 6; else mag = 7;
    return sign | mag;
}

__device__ __forceinline__ uint8_t float_to_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    uint32_t bits = __float_as_uint(v); int exp = ((bits>>23)&0xFF)-127; uint32_t frac = bits&0x7FFFFF;
    if (exp < -9) return 0; if (exp > 7) return 0x7F;
    if (exp < 0) {
        int shift = -exp;
        uint32_t mant = (frac|0x800000)>>(23+shift-2);
        if ((frac & ((1u<<(23+shift-2))-1)) != 0) mant+=1;
        return (mant&0x3);
    } else {
        uint32_t mant2 = (frac>>21)&0x3;
        if ((frac&((1u<<21)-1))!=0) { mant2+=1; if(mant2>3){mant2=0;exp+=1;} }
        if(exp>7) return 0x7F;
        return (exp<<2)|mant2;
    }
}

// ── Main kernel ──

__global__ void __launch_bounds__(THREADS, 2)
silu_down_kernel(
    const __nv_bfloat16* __restrict__ gateup_bf16,
    const uint8_t*       __restrict__ down_packed,
    const uint8_t*       __restrict__ down_sf,
    const __nv_bfloat16* __restrict__ residual_bf16,
    __nv_bfloat16*       __restrict__ out_bf16,
    int M, int FFN, int D_val, float alpha)
{
    const int K_val = FFN, K_half = K_val / 2, K_groups = K_val / 16, GU_N = 2 * FFN;
    const int NUM_CHUNKS = K_val / CHUNK_K;  // 3

    // Shared memory (static, fits within 48KB)
    __shared__ __align__(16) uint8_t Act_smem [CHUNK_ACT_SZ];   // 32768
    __shared__ __align__(16) uint8_t SFA_smem [CHUNK_SFA_SZ];   // 4096
    __shared__ __align__(16) uint8_t B_smem   [CHUNK_B_SZ];     // 6144
    __shared__ __align__(16) uint8_t SFB_smem [CHUNK_SFB_SZ];   // 512

    const int t = threadIdx.x, warp_id = t / 32, lane = t % 32, l = lane % 4, h = lane / 4;

    int n_tiles = (D_val + BLOCK_N - 1) / BLOCK_N;
    int m_idx = blockIdx.x / n_tiles, n_idx = blockIdx.x % n_tiles;
    int m_base = m_idx * BLOCK_M, n_base = n_idx * BLOCK_N;
    if (m_base >= M || n_base >= D_val) return;

    // Accumulators (persist across chunks)
    float dA[N_ATOMS]={0}, dB[N_ATOMS]={0}, dC[N_ATOMS]={0}, dD[N_ATOMS]={0};

    // ═══ Process each chunk of FFN ═══
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
        int k_chunk_base = chunk * CHUNK_K;

        // ── Phase 1: SiLU/Mul/Quantize this chunk ──
        // 64 rows, 128 threads → 2 threads per row, each handles 512 columns
        {
            int row_a = t / 2, m_glob = m_base + row_a;
            int my_start = (t % 2) * 512;

            if (m_glob < M) {
                const __nv_bfloat16* gate_row = gateup_bf16 + m_glob * GU_N + k_chunk_base;
                const __nv_bfloat16* up_row   = gateup_bf16 + m_glob * GU_N + FFN + k_chunk_base;

                // Process 32 elements at a time (16 bytes output)
                for (int c = my_start; c < my_start + 512 && c < CHUNK_K; c += 32) {
                    float vals[32];
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float g = __bfloat162float(gate_row[c + i]);
                        float u = __bfloat162float(up_row[c + i]);
                        vals[i] = __bfloat162float(__float2bfloat16(silu_f32(g))) * u;
                    }
                    int byte_base = c / 2;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        uint8_t lo = float_to_fp4_e2m1(vals[2*i]);
                        uint8_t hi = float_to_fp4_e2m1(vals[2*i+1]);
                        Act_smem[row_a * CHUNK_K_HALF + byte_base + i] = lo | (hi << 4);
                    }
                }
            } else {
                // Zero padding
                for (int i = my_start / 2; i < (my_start + 512) / 2 && i < CHUNK_K_HALF; ++i)
                    Act_smem[row_a * CHUNK_K_HALF + i] = 0;
            }
        }

        // SFA computation
        {
            int row_a = t / 2, m_glob = m_base + row_a;
            int my_start = (t % 2) * 32;

            for (int g = my_start; g < my_start + 32 && g < CHUNK_K_GROUPS; ++g) {
                uint8_t sf_byte = 0;
                if (m_glob < M) {
                    int col_base = k_chunk_base + g * 16;
                    const __nv_bfloat16* gate_row = gateup_bf16 + m_glob * GU_N;
                    const __nv_bfloat16* up_row   = gateup_bf16 + m_glob * GU_N + FFN;
                    float amax = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 16; ++e) {
                        float gv = __bfloat162float(gate_row[col_base + e]);
                        float uv = __bfloat162float(up_row[col_base + e]);
                        float v = __bfloat162float(__float2bfloat16(silu_f32(gv))) * uv;
                        amax = fmaxf(amax, fabsf(v));
                    }
                    float ds = amax / 6.0f; if (ds < 1e-12f) ds = 1e-12f;
                    sf_byte = float_to_ue4m3_ceil(ds);
                }
                SFA_smem[row_a * CHUNK_K_GROUPS + g] = sf_byte;
            }
        }
        __syncthreads();

        // ── Phase 2: MMA loop for this chunk ──
        // Load first weight tile (stage 0)
        {
            int k_global = k_chunk_base;
            int k_boff = k_global / 2;
            int row_b = t / 2, boff = (t % 2) * 16;
            int n_glob = n_base + row_b;
            const uint8_t* src = (n_glob < D_val) ? down_packed + (n_glob * K_half + k_boff + boff) : nullptr;
            cp_async_16(to_smem(&B_smem[0 * BLOCK_N * SMEM_K_STRIDE + row_b * SMEM_K_STRIDE + boff]), src);
            if (t < BLOCK_N) {
                uint32_t sf = 0;
                if (n_base + t < D_val) {
                    int base = swizzled_sf_base(n_base + t, k_global / 16, K_groups);
                    sf = *reinterpret_cast<const uint32_t*>(&down_sf[base]);
                }
                *reinterpret_cast<uint32_t*>(&SFB_smem[0 * BLOCK_N * SF_K_PER_ROW + t * SF_K_PER_ROW]) = sf;
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        int comp_stage = 0;
        for (int k_iter = 0; k_iter < CHUNK_ITERS; ++k_iter) {
            int next_stage = comp_stage ^ 1;
            int k_next = k_chunk_base + (k_iter + 1) * BLOCK_K;

            if (k_next < K_val) {
                int k_boff = k_next / 2;
                int row_b = t / 2, boff = (t % 2) * 16;
                int n_glob = n_base + row_b;
                const uint8_t* src = (n_glob < D_val) ? down_packed + (n_glob * K_half + k_boff + boff) : nullptr;
                cp_async_16(to_smem(&B_smem[next_stage * BLOCK_N * SMEM_K_STRIDE + row_b * SMEM_K_STRIDE + boff]), src);
                if (t < BLOCK_N) {
                    uint32_t sf = 0;
                    if (n_base + t < D_val) {
                        int base = swizzled_sf_base(n_base + t, k_next / 16, K_groups);
                        sf = *reinterpret_cast<const uint32_t*>(&down_sf[base]);
                    }
                    *reinterpret_cast<uint32_t*>(&SFB_smem[next_stage * BLOCK_N * SF_K_PER_ROW + t * SF_K_PER_ROW]) = sf;
                }
                asm volatile("cp.async.commit_group;\n" ::);
            }

            asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();

            const int warp_M_off = warp_id * 16, kA0 = 4*l, kA2 = 4*l+16;
            const int k_act_off = k_iter * (BLOCK_K/2);
            int rA0 = warp_M_off + h, rA1 = warp_M_off + h + 8;

            uint32_t A0 = *reinterpret_cast<const uint32_t*>(&Act_smem[rA0 * CHUNK_K_HALF + k_act_off + kA0]);
            uint32_t A1 = *reinterpret_cast<const uint32_t*>(&Act_smem[rA1 * CHUNK_K_HALF + k_act_off + kA0]);
            uint32_t A2 = *reinterpret_cast<const uint32_t*>(&Act_smem[rA0 * CHUNK_K_HALF + k_act_off + kA2]);
            uint32_t A3 = *reinterpret_cast<const uint32_t*>(&Act_smem[rA1 * CHUNK_K_HALF + k_act_off + kA2]);

            int sfa_row = warp_M_off + h, sfa_kg = k_iter * 4;
            uint32_t SFA_val = *reinterpret_cast<const uint32_t*>(&SFA_smem[sfa_row * CHUNK_K_GROUPS + sfa_kg]);

            uint8_t* B_stage_ptr = &B_smem[comp_stage * BLOCK_N * SMEM_K_STRIDE];
            uint8_t* SFB_stage_ptr = &SFB_smem[comp_stage * BLOCK_N * SF_K_PER_ROW];

            #pragma unroll
            for (int g = 0; g < N_GROUPS; ++g) {
                int base = g * 4;
                int c0=(base+0)*8+h, c1=(base+1)*8+h, c2=(base+2)*8+h, c3=(base+3)*8+h;

                uint32_t B0 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c0*SMEM_K_STRIDE+kA0]);
                uint32_t B1 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c0*SMEM_K_STRIDE+kA2]);
                uint32_t B2 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c1*SMEM_K_STRIDE+kA0]);
                uint32_t B3 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c1*SMEM_K_STRIDE+kA2]);
                uint32_t B4 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c2*SMEM_K_STRIDE+kA0]);
                uint32_t B5 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c2*SMEM_K_STRIDE+kA2]);
                uint32_t B6 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c3*SMEM_K_STRIDE+kA0]);
                uint32_t B7 = *reinterpret_cast<const uint32_t*>(&B_stage_ptr[c3*SMEM_K_STRIDE+kA2]);

                int sfb_n = g * 32 + l * 8 + h;
                uint32_t SFB_val = *reinterpret_cast<const uint32_t*>(&SFB_stage_ptr[sfb_n*SF_K_PER_ROW]);

                mma_nvfp4_4atom(
                    dA[base+0],dB[base+0],dC[base+0],dD[base+0],
                    dA[base+1],dB[base+1],dC[base+1],dD[base+1],
                    dA[base+2],dB[base+2],dC[base+2],dD[base+2],
                    dA[base+3],dB[base+3],dC[base+3],dD[base+3],
                    A0,A1,A2,A3, B0,B1,B2,B3,B4,B5,B6,B7, SFA_val,SFB_val);
            }
            comp_stage = next_stage;
            __syncthreads();
        }
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();  // Ensure all threads done before next chunk
    }

    // ═══ Epilogue: alpha scale + residual → BF16 ═══
    const int wmo = warp_id * 16;
    #pragma unroll
    for (int n = 0; n < N_ATOMS; ++n) {
        int c0 = n_base + n*8 + 2*l, c1 = c0 + 1;
        int r0 = m_base + wmo + h, r1 = r0 + 8;

        float v00 = dA[n]*alpha, v01 = dB[n]*alpha, v10 = dC[n]*alpha, v11 = dD[n]*alpha;

        if (r0 < M && c0 < D_val) { float r = __bfloat162float(residual_bf16[r0*D_val+c0]); out_bf16[r0*D_val+c0] = __float2bfloat16(v00+r); }
        if (r0 < M && c1 < D_val) { float r = __bfloat162float(residual_bf16[r0*D_val+c1]); out_bf16[r0*D_val+c1] = __float2bfloat16(v01+r); }
        if (r1 < M && c0 < D_val) { float r = __bfloat162float(residual_bf16[r1*D_val+c0]); out_bf16[r1*D_val+c0] = __float2bfloat16(v10+r); }
        if (r1 < M && c1 < D_val) { float r = __bfloat162float(residual_bf16[r1*D_val+c1]); out_bf16[r1*D_val+c1] = __float2bfloat16(v11+r); }
    }
}

}  // anonymous namespace

int omnivoice_ffn_silu_down_sm120(
    const void* gateup_bf16, const void* down_packed, const void* down_sf,
    const void* residual_bf16, void* out_bf16,
    int M, int FFN, int D_val, float alpha, cudaStream_t stream)
{
    if (FFN != 3072 || D_val != 1024) {
        fprintf(stderr, "[omnivoice_ffn_silu_down] FFN=%d D=%d (expected 3072,1024)\n", FFN, D_val);
        return -1;
    }
    if (M <= 0) return -2;

    int n_tiles = (D_val + BLOCK_N - 1) / BLOCK_N;
    int m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(m_tiles * n_tiles), block(THREADS);

    silu_down_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gateup_bf16),
        reinterpret_cast<const uint8_t*>(down_packed),
        reinterpret_cast<const uint8_t*>(down_sf),
        reinterpret_cast<const __nv_bfloat16*>(residual_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        M, FFN, D_val, alpha);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[omnivoice_ffn_silu_down] launch err: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

}  // namespace megakernel
}  // namespace flash_rt
