// ================================================================
// FlashRT — MaskGIT cross-position select kernel. See maskgit_select.cuh.
// ================================================================

#include "maskgit_select.cuh"
#include <curand_kernel.h>

#define SEL_THREADS 256
#define SEL_MAX_POW2 4096   // covers C*T up to 4096 (T up to 512 for C=8)

__global__ void maskgit_select_topk_kernel(
    const __nv_bfloat16* __restrict__ confidence,
    const int* __restrict__ pred_tokens,
    int* __restrict__ sample_tokens,
    const int* __restrict__ k_dev,
    int C, int T, float lpf, float pt, int mask_id, unsigned long long seed)
{
    int b = blockIdx.x;
    int ct = C * T;
    // smallest pow2 >= ct (bounded by SEL_MAX_POW2)
    int n = 1; while (n < ct) n <<= 1;
    if (n > SEL_MAX_POW2) n = SEL_MAX_POW2;

    extern __shared__ char smem_raw[];
    float* sval = reinterpret_cast<float*>(smem_raw);           // [n]
    int*   sidx = reinterpret_cast<int*>(sval + n);             // [n]

    const __nv_bfloat16* conf_b = confidence + b * ct;
    const int* pred_b = pred_tokens + b * ct;
    int* tok_b = sample_tokens + b * ct;

    curandStatePhilox4_32_10_t rst;
    curand_init(seed, (unsigned long long)b * SEL_THREADS + threadIdx.x, 0, &rst);

    // Load + compute scores (with position gumbel) + mask filled positions.
    for (int i = threadIdx.x; i < n; i += SEL_THREADS) {
        if (i < ct) {
            float c = __bfloat162float(conf_b[i]);
            int codebook = i / T;
            float raw = c - codebook * lpf;
            float sc;
            if (pt > 0.0f) {
                float u = curand_uniform(&rst);
                float g = -logf(-logf(u + 1e-10f) + 1e-10f);
                sc = raw / pt + g;
            } else {
                sc = raw;
            }
            if (tok_b[i] != mask_id) sc = -1e30f;   // already filled → never selected
            sval[i] = sc; sidx[i] = i;
        } else {
            sval[i] = -1e30f; sidx[i] = -1;
        }
    }
    __syncthreads();

    // Bitonic sort DESCENDING by sval.
    for (int kk = 2; kk <= n; kk <<= 1) {
        for (int j = kk >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += SEL_THREADS) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool desc = ((i & kk) == 0);
                    bool swap = desc ? (sval[i] < sval[ixj]) : (sval[i] > sval[ixj]);
                    if (swap) {
                        float tv = sval[i]; sval[i] = sval[ixj]; sval[ixj] = tv;
                        int ti = sidx[i]; sidx[i] = sidx[ixj]; sidx[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Read k from device, scatter pred into top-k positions.
    int k = *k_dev;
    if (k > ct) k = ct;
    for (int i = threadIdx.x; i < k; i += SEL_THREADS) {
        int pos = sidx[i];
        if (pos >= 0) tok_b[pos] = pred_b[pos];
    }
}

void maskgit_select_topk_bf16(
    const __nv_bfloat16* confidence, const int* pred_tokens, int* sample_tokens,
    const int* k_dev, int B, int C, int T, float layer_penalty_factor,
    float position_temp, int mask_id, unsigned long long seed, cudaStream_t stream)
{
    if (B <= 0) return;
    int n = 1; while (n < C * T) n <<= 1;
    if (n > SEL_MAX_POW2) n = SEL_MAX_POW2;
    dim3 grid(B), block(SEL_THREADS);
    int shared_bytes = n * (sizeof(float) + sizeof(int));
    maskgit_select_topk_kernel<<<grid, block, shared_bytes, stream>>>(
        confidence, pred_tokens, sample_tokens, k_dev,
        C, T, layer_penalty_factor, position_temp, mask_id, seed);
}
