// ================================================================
// FlashRT — MaskGIT per-row sample kernel (BF16). See maskgit_sample.cuh.
// ================================================================

#include "maskgit_sample.cuh"
#include <curand_kernel.h>

#define MS_THREADS 256
#define MS_POW2 2048   // next pow2 >= 1025 (V); bitonic sort size

namespace {

// Bitonic sort of shared pairs (val, idx), DESCENDING by val.
// n = MS_POW2, t = threadIdx.x over MS_THREADS. Each thread owns n/MS_THREADS = 8 pairs.
__device__ __forceinline__ void bitonic_sort_desc(float* vals, int* idxs, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += MS_THREADS) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool desc = ((i & k) == 0);           // descending within block when k-bit clear
                    float a = vals[i], b = vals[ixj];
                    int ia = idxs[i], ib = idxs[ixj];
                    bool swap = desc ? (a < b) : (a > b);
                    if (swap) {
                        vals[i] = b; vals[ixj] = a;
                        idxs[i] = ib; idxs[ixj] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }
}

}  // namespace

__global__ void maskgit_sample_row_kernel(
    const __nv_bfloat16* __restrict__ log_probs,
    int* __restrict__ pred_tokens,
    __nv_bfloat16* __restrict__ confidence,
    int V, int num_filt, int mask_id, float ct, unsigned long long seed)
{
    int row = blockIdx.x;
    if (row >= gridDim.x) return;

    extern __shared__ char smem_raw[];
    float* sval = reinterpret_cast<float*>(smem_raw);            // [MS_POW2]
    int*   sidx = reinterpret_cast<int*>(sval + MS_POW2);        // [MS_POW2]

    const __nv_bfloat16* lp = log_probs + row * V;

    // Load row into shared (pad with -inf, idx=-1)
    for (int i = threadIdx.x; i < MS_POW2; i += MS_THREADS) {
        if (i < V) { sval[i] = __bfloat162float(lp[i]); sidx[i] = i; }
        else       { sval[i] = -1e30f; sidx[i] = -1; }
    }
    __syncthreads();

    bitonic_sort_desc(sval, sidx, MS_POW2);
    // After sort: sval[0] = max (confidence), sval[0..num_filt-1] = top num_filt.

    // confidence = max → sval[0]
    if (threadIdx.x == 0) {
        confidence[row] = __float2bfloat16(sval[0]);
    }

    if (ct <= 0.0f) {
        // No gumbel: pred = global argmax (sort already placed it at sidx[0]).
        if (threadIdx.x == 0) pred_tokens[row] = sidx[0];
        return;
    }

    // gumbel-sample among top num_filt: g_i = sval[i]/ct + (-log(-log(u+eps)+eps)); argmax g → pred.
    // Each thread handles <= ceil(num_filt/MS_THREADS) candidates.
    curandStatePhilox4_32_10_t st;
    curand_init(seed, (unsigned long long)row * MS_THREADS + threadIdx.x, 0, &st);
    float best_g = -1e30f; int best_idx = -1;
    for (int i = threadIdx.x; i < num_filt; i += MS_THREADS) {
        float u = curand_uniform(&st);                 // (0,1]
        float gumbel = -logf(-logf(u + 1e-10f) + 1e-10f);
        float g = sval[i] / ct + gumbel;
        if (g > best_g) { best_g = g; best_idx = sidx[i]; }
    }
    // Warp + block reduce to find global argmax(best_g)
    __shared__ float red_v[MS_THREADS]; __shared__ int red_i[MS_THREADS];
    red_v[threadIdx.x] = best_g; red_i[threadIdx.x] = best_idx;
    __syncthreads();
    for (int off = MS_THREADS >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            if (red_v[threadIdx.x + off] > red_v[threadIdx.x]) {
                red_v[threadIdx.x] = red_v[threadIdx.x + off];
                red_i[threadIdx.x] = red_i[threadIdx.x + off];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pred_tokens[row] = red_i[0];
    }
}

void maskgit_sample_row_bf16(
    const __nv_bfloat16* log_probs,
    int* pred_tokens, __nv_bfloat16* confidence,
    int rows, int V, int num_filt, int mask_id, float class_temp,
    unsigned long long seed, cudaStream_t stream)
{
    if (rows <= 0) return;
    dim3 grid(rows), block(MS_THREADS);
    int shared_bytes = MS_POW2 * (sizeof(float) + sizeof(int));
    maskgit_sample_row_kernel<<<grid, block, shared_bytes, stream>>>(
        log_probs, pred_tokens, confidence, V, num_filt, mask_id, class_temp, seed);
}
