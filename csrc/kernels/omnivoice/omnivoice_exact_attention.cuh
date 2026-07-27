#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt::kernels {

// OmniVoice's exact single-request CFG attention. Q/K/V contain the two
// sequences packed back-to-back and cu_seqlens is [0, conditional, total].
// The implementation keeps PyTorch's 128-key memory-efficient CUTLASS math
// and exact reduction order, while specializing the query tile to 32 rows
// for the short no-bias/no-dropout inference shape.
void omnivoice_exact_packed_attention_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* output,
    const int32_t* cu_seqlens,
    int total_rows,
    int max_seqlen,
    int num_sequences,
    int num_heads,
    int head_dim,
    float scale,
    cudaStream_t stream);

// Exact two-sequence CFG specialization with a compact CTA grid. The first
// sequence has max_seqlen rows and the second occupies the remaining packed
// rows. This removes empty query blocks for the shorter sequence without
// changing the 32x128 attention tile or its arithmetic.
void omnivoice_exact_packed_attention_compact_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* output,
    int total_rows,
    int max_seqlen,
    int num_heads,
    int head_dim,
    float scale,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
