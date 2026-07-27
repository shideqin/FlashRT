#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::kernels {

// Run a row-major BF16 GEMM for M active rows while retaining the cuBLASLt
// algorithm selected for reference_M rows:
//
//   out[M, N] = x[M, K] @ weight[N, K]^T
//
// OmniVoice's padded CFG batch needs only the active prefix after attention,
// but PyTorch selects a different GEMM algorithm when the row count is
// reduced directly.  Holding the reference-shape algorithm fixed preserves
// the BF16 reduction order while avoiding work on the padded tail.
void omnivoice_bf16_matmul_reference_algo(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    int reference_M,
    cudaStream_t stream);

// As above, but select a specific cuBLASLt heuristic rank. This is used only
// for shape/rank pairs that are both faster and bit-identical to PyTorch's
// rank-zero BF16 result on the target architecture.
void omnivoice_bf16_matmul_heuristic_algo(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    int reference_M,
    int algorithm_rank,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
