// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — Fused Q/K RMSNorm + RoPE kernel (V13: bit-exact with original).
//
// Combines fused_qk_norm_bf16 + qwen36_partial_rope_qk_bf16 into a single
// kernel launch, while maintaining bit-exact numerical results.
//
// Key design for bit-exactness:
//   - 256 threads/block, BS blocks (matching fused_qk_norm.cu)
//   - packed2 (__nv_bfloat162) loading/storing (matching original)
//   - block_reduce_sum for RMS reduction (matching original)
//   - Same rsqrtf(sum/HD + eps) formula
//   - Same partial_rope_value formula with bf16 rounding of rot*sv
//   - RoPE uses shared-memory buffering to avoid in-place read-write hazard
//
// Pipeline within each block:
//   1. Norm all Q heads → write to q_out (packed2, matching fused_qk_norm)
//   2. Norm all K heads → write to k_out (packed2)
//   3. __syncthreads()
//   4. For each Q head: load→shared→compute RoPE→write (safe in-place)
//   5. For each K head: load→shared→compute RoPE→write
// ================================================================

#include "fused_qk_norm_rope.cuh"
#include "common.cuh"

namespace flash_rt {
namespace kernels {

__global__ void fused_qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int NH, int NKV, int HD, int QKVD, float eps)
{
    using T = __nv_bfloat16;
    using T2 = __nv_bfloat162;

    int row = blockIdx.x;
    int HD2 = HD >> 1;
    int NQK = NH * HD;
    int KVD = NKV * HD;

    extern __shared__ float shared[];
    // shared[] is used for:
    //   Phase 1: block_reduce_sum (256 floats)
    //   Phase 2: RoPE head buffer (HD floats = 128 floats, reused per head)
    // So we need max(256, HD) floats. With HD=128 and 256 threads, 256 is enough.
    float* rope_buf = shared;  // reuse same shared memory

    const T2* w2_q = reinterpret_cast<const T2*>(q_weight);
    const T2* w2_k = reinterpret_cast<const T2*>(k_weight);

    // ── Phase 1: Q per-head RMSNorm (bit-exact copy from fused_qk_norm.cu) ──
    const T* dq_q = dq + row * QKVD;
    T* q_out_row = q_out + row * NQK;

    for (int h = 0; h < NH; ++h) {
        const T* head_start = dq_q + h * HD;
        T* head_out = q_out_row + h * HD;
        const T2* head2 = reinterpret_cast<const T2*>(head_start);
        T2* out2 = reinterpret_cast<T2*>(head_out);

        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 val = head2[i];
            float v0 = to_f32(val.x), v1 = to_f32(val.y);
            local_sum += v0 * v0 + v1 * v1;
        }
        float rms = rsqrtf(block_reduce_sum(local_sum, shared) / HD + eps);

        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 xv = head2[i], wv = w2_q[i];
            float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
            out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
        }
        __syncthreads();
    }

    // ── Phase 2: K per-head RMSNorm ──
    const T* dq_k = dq + row * QKVD + NQK;
    T* k_out_row = k_out + row * KVD;

    for (int h = 0; h < NKV; ++h) {
        const T* head_start = dq_k + h * HD;
        T* head_out = k_out_row + h * HD;
        const T2* head2 = reinterpret_cast<const T2*>(head_start);
        T2* out2 = reinterpret_cast<T2*>(head_out);

        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 val = head2[i];
            float v0 = to_f32(val.x), v1 = to_f32(val.y);
            local_sum += v0 * v0 + v1 * v1;
        }
        float rms = rsqrtf(block_reduce_sum(local_sum, shared) / HD + eps);

        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 xv = head2[i], wv = w2_k[i];
            float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
            out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
        }
        __syncthreads();
    }

    // Barrier: all norm outputs are in q_out/k_out global memory
    __syncthreads();

    // ── Phase 3: Q RoPE (safe via shared-memory buffering) ──
    // Process one head at a time:
    //   1. Load all HD elements from q_out into rope_buf[] as float
    //   2. __syncthreads()
    //   3. Compute RoPE from rope_buf (immutable) → write back to q_out
    //   4. __syncthreads()
    for (int h = 0; h < NH; ++h) {
        T* head_out = q_out_row + h * HD;

        // Load head elements into shared memory
        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            rope_buf[col] = __bfloat162float(head_out[col]);
        }
        __syncthreads();

        // Compute RoPE using rope_buf (read-only) and cos/sin
        int half = HD >> 1;
        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            if (col >= HD) continue;  // redundant check
            float cv = __bfloat162float(cos[row * HD + col]);
            float sv = __bfloat162float(sin[row * HD + col]);
            float xv = rope_buf[col];

            int rot_col = (col < half) ? (col + half) : (col - half);
            float rot = rope_buf[rot_col];
            if (col < half) rot = -rot;

            float rot_sin_bf = __bfloat162float(__float2bfloat16(rot * sv));
            head_out[col] = __float2bfloat16(rot_sin_bf + xv * cv);
        }
        __syncthreads();
    }

    // ── Phase 4: K RoPE ──
    for (int h = 0; h < NKV; ++h) {
        T* head_out = k_out_row + h * HD;

        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            rope_buf[col] = __bfloat162float(head_out[col]);
        }
        __syncthreads();

        int half = HD >> 1;
        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            float cv = __bfloat162float(cos[row * HD + col]);
            float sv = __bfloat162float(sin[row * HD + col]);
            float xv = rope_buf[col];

            int rot_col = (col < half) ? (col + half) : (col - half);
            float rot = rope_buf[rot_col];
            if (col < half) rot = -rot;

            float rot_sin_bf = __bfloat162float(__float2bfloat16(rot * sv));
            head_out[col] = __float2bfloat16(rot_sin_bf + xv * cv);
        }
        __syncthreads();
    }
}

void fused_qk_norm_rope_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream)
{
    if (BS <= 0 || NH <= 0 || NKV <= 0 || HD <= 0) return;

    constexpr int kThreads = 256;
    dim3 block(kThreads);
    dim3 grid(BS);

    // Shared memory: max(kThreads, HD) floats for block_reduce_sum + RoPE buffer
    constexpr int kSharedBytes = 256 * sizeof(float);

    fused_qk_norm_rope_kernel<<<grid, block, kSharedBytes, stream>>>(
        dq, q_weight, k_weight, cos, sin,
        q_out, k_out,
        NH, NKV, HD, QKVD, eps);
}

// ── V15: Optimized variant with temp buffers for RoPE ──
// Eliminates 48 __syncthreads() from the RoPE phase by writing RoPE output
// to separate temp buffers (q_temp/k_temp), avoiding the in-place read-write
// hazard without shared-memory buffering.
//
// Syncthreads: 16 (Q norm) + 8 (K norm) + 1 (barrier) = 25 (vs 73 in V13).

__global__ void fused_qk_norm_rope_v2_kernel(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ q_temp,
    __nv_bfloat16* __restrict__ k_temp,
    int NH, int NKV, int HD, int QKVD, float eps)
{
    using T = __nv_bfloat16;
    using T2 = __nv_bfloat162;

    int row = blockIdx.x;
    int HD2 = HD >> 1;
    int NQK = NH * HD;
    int KVD = NKV * HD;
    int half = HD >> 1;

    extern __shared__ float shared[];  // block_reduce_sum (256 floats)

    const T2* w2_q = reinterpret_cast<const T2*>(q_weight);
    const T2* w2_k = reinterpret_cast<const T2*>(k_weight);

    // ── Phase 1: Q per-head RMSNorm (same as V13) ──
    const T* dq_q = dq + row * QKVD;
    T* q_out_row = q_out + row * NQK;

    for (int h = 0; h < NH; ++h) {
        const T* head_start = dq_q + h * HD;
        T* head_out = q_out_row + h * HD;
        const T2* head2 = reinterpret_cast<const T2*>(head_start);
        T2* out2 = reinterpret_cast<T2*>(head_out);

        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 val = head2[i];
            float v0 = to_f32(val.x), v1 = to_f32(val.y);
            local_sum += v0 * v0 + v1 * v1;
        }
        float rms = rsqrtf(block_reduce_sum(local_sum, shared) / HD + eps);

        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 xv = head2[i], wv = w2_q[i];
            float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
            out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
        }
        __syncthreads();
    }

    // ── Phase 2: K per-head RMSNorm (same as V13) ──
    const T* dq_k = dq + row * QKVD + NQK;
    T* k_out_row = k_out + row * KVD;

    for (int h = 0; h < NKV; ++h) {
        const T* head_start = dq_k + h * HD;
        T* head_out = k_out_row + h * HD;
        const T2* head2 = reinterpret_cast<const T2*>(head_start);
        T2* out2 = reinterpret_cast<T2*>(head_out);

        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 val = head2[i];
            float v0 = to_f32(val.x), v1 = to_f32(val.y);
            local_sum += v0 * v0 + v1 * v1;
        }
        float rms = rsqrtf(block_reduce_sum(local_sum, shared) / HD + eps);

        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 xv = head2[i], wv = w2_k[i];
            float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
            out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
        }
        __syncthreads();
    }

    // Barrier: all norm outputs are in q_out/k_out global memory
    __syncthreads();

    // ── Phase 3: Q RoPE (read q_out → write q_temp, NO syncthreads!) ──
    // Since q_out is read-only and q_temp is write-only, there is no
    // in-place race condition. Each thread independently reads from
    // q_out and writes to q_temp.
    T* q_temp_row = q_temp + row * NQK;
    const T* cos_row = cos + row * HD;
    const T* sin_row = sin + row * HD;

    for (int h = 0; h < NH; ++h) {
        T* head_out = q_out_row + h * HD;
        T* head_tmp = q_temp_row + h * HD;

        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            float cv = to_f32(cos_row[col]);
            float sv = to_f32(sin_row[col]);
            float xv = to_f32(head_out[col]);

            int rot_col = (col < half) ? (col + half) : (col - half);
            float rot = to_f32(head_out[rot_col]);
            if (col < half) rot = -rot;

            float rot_sin_bf = to_f32(from_f32<T>(rot * sv));
            head_tmp[col] = from_f32<T>(rot_sin_bf + xv * cv);
        }
    }

    // ── Phase 4: K RoPE (read k_out → write k_temp, NO syncthreads!) ──
    T* k_temp_row = k_temp + row * KVD;

    for (int h = 0; h < NKV; ++h) {
        T* head_out = k_out_row + h * HD;
        T* head_tmp = k_temp_row + h * HD;

        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            float cv = to_f32(cos_row[col]);
            float sv = to_f32(sin_row[col]);
            float xv = to_f32(head_out[col]);

            int rot_col = (col < half) ? (col + half) : (col - half);
            float rot = to_f32(head_out[rot_col]);
            if (col < half) rot = -rot;

            float rot_sin_bf = to_f32(from_f32<T>(rot * sv));
            head_tmp[col] = from_f32<T>(rot_sin_bf + xv * cv);
        }
    }
}

void fused_qk_norm_rope_v2_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_temp,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream)
{
    if (BS <= 0 || NH <= 0 || NKV <= 0 || HD <= 0) return;

    constexpr int kThreads = 256;
    dim3 block(kThreads);
    dim3 grid(BS);

    // Shared memory for block_reduce_sum only (no RoPE buffer needed)
    constexpr int kSharedBytes = 256 * sizeof(float);

    fused_qk_norm_rope_v2_kernel<<<grid, block, kSharedBytes, stream>>>(
        dq, q_weight, k_weight, cos, sin,
        q_out, k_out, q_temp, k_temp,
        NH, NKV, HD, QKVD, eps);
}

// ── V21v3: Norm output stays in shared memory ──
// Eliminates q_out/k_out global memory writes by keeping norm'd data
// in shared memory for immediate RoPE consumption. Saves ~10 us/layer.
//
// Shared memory layout:
//   [0..1023]         reduce_buf (256 floats)
//   [1024..1279]      norm_buf  (128 bf16 scalars = 256 bytes)
// Total: 1280 bytes.

__global__ void fused_qk_norm_rope_v3_kernel(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_temp,
    __nv_bfloat16* __restrict__ k_temp,
    int NH, int NKV, int HD, int QKVD, float eps)
{
    using T = __nv_bfloat16;
    using T2 = __nv_bfloat162;

    int row = blockIdx.x;
    int HD2 = HD >> 1;
    int NQK = NH * HD;
    int KVD = NKV * HD;
    int half = HD >> 1;

    extern __shared__ float shared[];
    float* reduce_buf = shared;                   // [256] floats
    T*     norm_buf   = reinterpret_cast<T*>(shared + 256);  // [HD] bf16

    const T2* w2_q = reinterpret_cast<const T2*>(q_weight);
    const T2* w2_k = reinterpret_cast<const T2*>(k_weight);

    const T* cos_row = cos + row * HD;
    const T* sin_row = sin + row * HD;

    // ── Phase 1: Q heads — norm→shmem→RoPE→q_temp ──
    const T* dq_q = dq + row * QKVD;
    T* q_temp_row = q_temp + row * NQK;

    for (int h = 0; h < NH; ++h) {
        const T* head_start = dq_q + h * HD;
        const T2* head2 = reinterpret_cast<const T2*>(head_start);

        // 1a. Compute RMSNorm sum
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 val = head2[i];
            float v0 = to_f32(val.x), v1 = to_f32(val.y);
            local_sum += v0 * v0 + v1 * v1;
        }
        float rms = rsqrtf(block_reduce_sum(local_sum, reduce_buf) / HD + eps);

        // 1b. Norm → store scalars in shared memory (not global!)
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 xv = head2[i], wv = w2_q[i];
            float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
            norm_buf[i * 2]     = from_f32<T>(v0);
            norm_buf[i * 2 + 1] = from_f32<T>(v1);
        }
        __syncthreads();

        // 1c. RoPE: read norm_buf (shared) → write q_temp (global)
        T* head_tmp = q_temp_row + h * HD;
        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            float cv = to_f32(cos_row[col]);
            float sv = to_f32(sin_row[col]);
            float xv = to_f32(norm_buf[col]);

            int rot_col = (col < half) ? (col + half) : (col - half);
            float rot = to_f32(norm_buf[rot_col]);
            if (col < half) rot = -rot;

            float rot_sin_bf = to_f32(from_f32<T>(rot * sv));
            head_tmp[col] = from_f32<T>(rot_sin_bf + xv * cv);
        }
        __syncthreads();
    }

    // ── Phase 2: K heads — norm→shmem→RoPE→k_temp ──
    const T* dq_k = dq + row * QKVD + NQK;
    T* k_temp_row = k_temp + row * KVD;

    for (int h = 0; h < NKV; ++h) {
        const T* head_start = dq_k + h * HD;
        const T2* head2 = reinterpret_cast<const T2*>(head_start);

        // 2a. Compute RMSNorm sum
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 val = head2[i];
            float v0 = to_f32(val.x), v1 = to_f32(val.y);
            local_sum += v0 * v0 + v1 * v1;
        }
        float rms = rsqrtf(block_reduce_sum(local_sum, reduce_buf) / HD + eps);

        // 2b. Norm → store scalars in shared memory
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 xv = head2[i], wv = w2_k[i];
            float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
            norm_buf[i * 2]     = from_f32<T>(v0);
            norm_buf[i * 2 + 1] = from_f32<T>(v1);
        }
        __syncthreads();

        // 2c. RoPE: read norm_buf (shared) → write k_temp (global)
        T* head_tmp = k_temp_row + h * HD;
        for (int col = threadIdx.x; col < HD; col += blockDim.x) {
            float cv = to_f32(cos_row[col]);
            float sv = to_f32(sin_row[col]);
            float xv = to_f32(norm_buf[col]);

            int rot_col = (col < half) ? (col + half) : (col - half);
            float rot = to_f32(norm_buf[rot_col]);
            if (col < half) rot = -rot;

            float rot_sin_bf = to_f32(from_f32<T>(rot * sv));
            head_tmp[col] = from_f32<T>(rot_sin_bf + xv * cv);
        }
        __syncthreads();
    }
}

void fused_qk_norm_rope_v3_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,        // UNUSED
    __nv_bfloat16* k_out,        // UNUSED
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_temp,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream)
{
    if (BS <= 0 || NH <= 0 || NKV <= 0 || HD <= 0) return;

    constexpr int kThreads = 256;
    dim3 block(kThreads);
    dim3 grid(BS);

    // Shared memory: 256 floats (reduce) + 128 bf16 (norm_buf)
    constexpr int kSharedBytes = 256 * sizeof(float) + 128 * sizeof(__nv_bfloat16);

    fused_qk_norm_rope_v3_kernel<<<grid, block, kSharedBytes, stream>>>(
        dq, q_weight, k_weight, cos, sin,
        q_temp, k_temp,
        NH, NKV, HD, QKVD, eps);
}

// ── V26v4 (optimized): Warp-per-head parallelism, fully register-based ──
// Each warp processes one head independently using warp shuffle for
// RMS reduction. 8 warps × 1 head = 8 heads in parallel.
// Q (16 heads): 2 iterations, K (8 heads): 1 iteration.
// ZERO __syncthreads() — all data kept in registers.
//
// Key insight: for HD=128 with 32 threads/warp, each thread owns 4 elements
// at indices {lane, lane+32, lane+64, lane+96}. RoPE rotation pairs
// (col ↔ col+half) map to indices within the SAME thread (e.g., lane ↔ lane+64),
// so no cross-thread communication is needed after RMS reduction.
//
// Cos/Sin are cached in registers at the start and reused for Q and K phases.

__global__ void fused_qk_norm_rope_v4_kernel(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_temp,
    __nv_bfloat16* __restrict__ k_temp,
    int NH, int NKV, int HD, int QKVD, float eps)
{
    using T = __nv_bfloat16;

    constexpr int kWarpCount = 8;
    constexpr int kWarpSize = 32;
    constexpr int kElemsPerThread = 128 / kWarpSize;  // 4 for HD=128
    constexpr int kHalf = 128 >> 1;                    // 64
    constexpr int kRotSlotOffset = kHalf / kWarpSize;  // 2

    int row = blockIdx.x;
    int warp_id = threadIdx.x / kWarpSize;
    int lane_id = threadIdx.x % kWarpSize;
    int NQK = NH * HD;
    int KVD = NKV * HD;

    // ── Cache cos/sin for this row in registers ──
    const T* cos_row = cos + row * HD;
    const T* sin_row = sin + row * HD;
    float cos_vals[kElemsPerThread];
    float sin_vals[kElemsPerThread];
    #pragma unroll
    for (int e = 0; e < kElemsPerThread; ++e) {
        int col = lane_id + e * kWarpSize;
        cos_vals[e] = to_f32(cos_row[col]);
        sin_vals[e] = to_f32(sin_row[col]);
    }

    // ── Phase 1: Q heads (NH=16, 2 iterations of 8) ──
    const T* dq_q = dq + row * QKVD;
    T* q_temp_row = q_temp + row * NQK;

    for (int iter = 0; iter < 2; ++iter) {
        int h = iter * kWarpCount + warp_id;
        if (h >= NH) continue;

        const T* head_src = dq_q + h * HD;
        T* head_dst = q_temp_row + h * HD;

        // Load elements into registers + compute RMS sum
        float vals[kElemsPerThread];
        float local_sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            vals[e] = to_f32(head_src[col]);
            local_sum += vals[e] * vals[e];
        }

        // Warp shuffle reduction for RMS (butterfly → all lanes hold the sum)
        float total = local_sum;
        total += __shfl_xor_sync(0xffffffff, total, 16);
        total += __shfl_xor_sync(0xffffffff, total, 8);
        total += __shfl_xor_sync(0xffffffff, total, 4);
        total += __shfl_xor_sync(0xffffffff, total, 2);
        total += __shfl_xor_sync(0xffffffff, total, 1);
        float rms = rsqrtf(total / HD + eps);

        // Normalize in registers
        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            vals[e] = vals[e] * rms * to_f32(q_weight[col]);
        }

        // Apply RoPE + write output (all from registers, no shared memory)
        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            float cv = cos_vals[e];
            float sv = sin_vals[e];
            float xv = vals[e];

            int rot_slot = (col < kHalf) ? (e + kRotSlotOffset) : (e - kRotSlotOffset);
            float rot_val = vals[rot_slot];
            if (col < kHalf) rot_val = -rot_val;

            float rot_sin_bf = to_f32(from_f32<T>(rot_val * sv));
            head_dst[col] = from_f32<T>(rot_sin_bf + xv * cv);
        }
        // No __syncthreads() needed — all data in registers, each warp independent
    }

    // ── Phase 2: K heads (NKV=8, 1 iteration of 8) ──
    const T* dq_k = dq + row * QKVD + NQK;
    T* k_temp_row = k_temp + row * KVD;

    {
        int h = warp_id;
        if (h < NKV) {
            const T* head_src = dq_k + h * HD;
            T* head_dst = k_temp_row + h * HD;

            // Load + RMS sum
            float vals[kElemsPerThread];
            float local_sum = 0.0f;
            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                vals[e] = to_f32(head_src[col]);
                local_sum += vals[e] * vals[e];
            }

            // Warp shuffle reduction (butterfly → all lanes hold the sum)
            float total = local_sum;
            total += __shfl_xor_sync(0xffffffff, total, 16);
            total += __shfl_xor_sync(0xffffffff, total, 8);
            total += __shfl_xor_sync(0xffffffff, total, 4);
            total += __shfl_xor_sync(0xffffffff, total, 2);
            total += __shfl_xor_sync(0xffffffff, total, 1);
            float rms = rsqrtf(total / HD + eps);

            // Normalize in registers
            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                vals[e] = vals[e] * rms * to_f32(k_weight[col]);
            }

            // RoPE + write output
            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                float cv = cos_vals[e];
                float sv = sin_vals[e];
                float xv = vals[e];

                int rot_slot = (col < kHalf) ? (e + kRotSlotOffset) : (e - kRotSlotOffset);
                float rot_val = vals[rot_slot];
                if (col < kHalf) rot_val = -rot_val;

                float rot_sin_bf = to_f32(from_f32<T>(rot_val * sv));
                head_dst[col] = from_f32<T>(rot_sin_bf + xv * cv);
            }
        }
        // No __syncthreads() — each warp independent
    }
}

void fused_qk_norm_rope_v4_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,        // UNUSED (API compat)
    __nv_bfloat16* k_out,        // UNUSED (API compat)
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_temp,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream)
{
    if (BS <= 0 || NH <= 0 || NKV <= 0 || HD <= 0) return;

    constexpr int kThreads = 256;
    dim3 block(kThreads);
    dim3 grid(BS);

    // No shared memory needed — fully register-based
    fused_qk_norm_rope_v4_kernel<<<grid, block, 0, stream>>>(
        dq, q_weight, k_weight, cos, sin,
        q_temp, k_temp,
        NH, NKV, HD, QKVD, eps);
}

}  // namespace kernels
}  // namespace flash_rt
