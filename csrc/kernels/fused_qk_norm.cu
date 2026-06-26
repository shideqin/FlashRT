// ================================================================
// FlashRT — Fused Q/K Per-Head RMSNorm kernel (V7 fix: per-head norm)
// ================================================================

#include "fused_qk_norm.cuh"
#include "common.cuh"

template<typename T>
__global__ void fused_qk_norm_kernel(
    const T* __restrict__ dq,
    const T* __restrict__ q_weight,
    const T* __restrict__ k_weight,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    int NH, int NKV, int HD, int QKVD, float eps) {

    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;                  // token index [0, BS)
    int NQK = NH * HD;
    int KVD = NKV * HD;
    int HD2 = HD >> 1;                    // packed pairs per head
    int NQK2 = NQK >> 1;                  // packed pairs for all Q heads
    int KVD2 = KVD >> 1;

    extern __shared__ float shared[];
    const T2* w2_q = reinterpret_cast<const T2*>(q_weight);
    const T2* w2_k = reinterpret_cast<const T2*>(k_weight);

    // ── Q norm: per-head (each head = HD elements) ──
    const T* dq_q = dq + row * QKVD;
    T* q_out_row = q_out + row * NQK;

    for (int h = 0; h < NH; ++h) {
        const T* head_start = dq_q + h * HD;
        T* head_out = q_out_row + h * HD;
        const T2* head2 = reinterpret_cast<const T2*>(head_start);
        T2* out2 = reinterpret_cast<T2*>(head_out);

        // Compute RMS for this head (HD elements)
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 val = head2[i];
            float v0 = to_f32(val.x), v1 = to_f32(val.y);
            local_sum += v0 * v0 + v1 * v1;
        }
        float rms = rsqrtf(block_reduce_sum(local_sum, shared) / HD + eps);

        // Normalize + weight
        for (int i = threadIdx.x; i < HD2; i += blockDim.x) {
            T2 xv = head2[i], wv = w2_q[i];
            float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
            out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
        }
        __syncthreads();
    }

    // ── K norm: per-head ──
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
}

template __global__ void fused_qk_norm_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);

void fused_qk_norm_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream) {
    fused_qk_norm_kernel<__nv_bfloat16>
        <<<BS, 256, 256 * sizeof(float), stream>>>(
            dq, q_weight, k_weight, q_out, k_out,
            NH, NKV, HD, QKVD, eps);
}
