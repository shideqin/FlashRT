#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::kernels {

// Experimental exact-BF16 dual GEMM:
//   out = BF16(BF16(silu(BF16(x @ gate.T))) * BF16(x @ up.T))
// The two projections share the A tile and the intermediate projections stay
// inside the epilogue.
int omnivoice_bf16_dual_swiglu(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight,
    __nv_bfloat16* out,
    int rows,
    int hidden_dim,
    int input_dim,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
