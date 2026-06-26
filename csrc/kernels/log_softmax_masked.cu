// ================================================================
// FlashRT — Masked Log-Softmax kernel (BF16)
//
// Fused: log_softmax + mask. Warp-per-row, BF16 I/O, FP32 compute.
// ================================================================

#include "log_softmax_masked.cuh"

#define LSM_WARP_SIZE 32
#define LSM_MAX_COLS 2048
#define LSM_ITERS (LSM_MAX_COLS / LSM_WARP_SIZE)  // 64

__global__ void log_softmax_masked_bf16_kernel(
    __nv_bfloat16* data, int rows, int cols, int mask_col) {

    int lane = threadIdx.x % LSM_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    __nv_bfloat16* src = data + row * cols;

    // Pass 1: load + find max (in float)
    float reg[LSM_ITERS];
    float mx = -1e30f;

    #pragma unroll
    for (int it = 0; it < LSM_ITERS; it++) {
        int c = it * LSM_WARP_SIZE + lane;
        if (c < cols) {
            float v = __bfloat162float(src[c]);
            reg[it] = v;
            mx = fmaxf(mx, v);
        } else {
            reg[it] = -1e30f;
        }
    }

    // Warp reduce max
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    // Pass 2: exp + sum
    float sm = 0;
    #pragma unroll
    for (int it = 0; it < LSM_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    // Pass 3: log_softmax = (x - mx) - log(sum(exp(x - mx)))
    float log_sm = logf(sm + 1e-12f);

    #pragma unroll
    for (int it = 0; it < LSM_ITERS; it++) {
        int c = it * LSM_WARP_SIZE + lane;
        if (c < cols) {
            float val;
            if (c == mask_col) {
                val = -1e30f;  // masked token → -inf in log-space
            } else {
                float x = __bfloat162float(src[c]);
                val = (x - mx) - log_sm;
            }
            src[c] = __float2bfloat16(val);
        }
    }
}

void log_softmax_masked_bf16(
    __nv_bfloat16* data, int rows, int cols, int mask_col,
    cudaStream_t stream) {
    dim3 grid(rows);
    dim3 block(LSM_WARP_SIZE);
    log_softmax_masked_bf16_kernel<<<grid, block, 0, stream>>>(
        data, rows, cols, mask_col);
}
