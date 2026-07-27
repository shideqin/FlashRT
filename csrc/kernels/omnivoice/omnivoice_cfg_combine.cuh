// ================================================================
// FlashRT — OmniVoice fused kernels.
//
// Replaces 3× torch log_softmax in the MaskGIT CFG path:
//   c_lp = log_softmax(c_logits)        with mask_col → -inf
//   u_lp = log_softmax(u_logits)        (NOT masked)
//   out  = log_softmax(c_lp + gs*(c_lp - u_lp))
//
// Warp-per-row, BF16 I/O, FP32 compute, shfl_xor reductions.
// Covers up to CC_ITERS*32 columns (1536).
// ================================================================

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt { namespace kernels {

// (kept in namespace for organization; .cu defines at global scope via using)
}}

void omnivoice_cfg_logsoftmax_bf16(
    const __nv_bfloat16* c_logits,   // [rows, cols]
    const __nv_bfloat16* u_logits,   // [rows, cols]
    __nv_bfloat16* out,              // [rows, cols]
    int rows, int cols, int mask_col, float guidance_scale,
    cudaStream_t stream);

void omnivoice_cfg_combine_f32(
    const float* conditional_log_probs,
    const float* unconditional_log_probs,
    float* out,
    int elements,
    float guidance_scale,
    cudaStream_t stream);

void omnivoice_logits_bf16_cfg_argmax_scores_f32(
    const __nv_bfloat16* packed_logits,
    const float* layer_penalty,
    int64_t* predictions,
    float* scores,
    int rows,
    int cols,
    int target_length,
    int mask_col,
    float guidance_scale,
    cudaStream_t stream);

void omnivoice_embed_codebooks_bf16(
    const int64_t* tokens,
    const __nv_bfloat16* embedding,
    const int64_t* codebook_offsets,
    __nv_bfloat16* conditional_out,
    __nv_bfloat16* unconditional_out,
    int num_codebooks,
    int token_stride,
    int target_length,
    int embedding_dim,
    cudaStream_t stream);

void omnivoice_prepare_cfg_embeddings_bf16(
    const int64_t* tokens,
    const __nv_bfloat16* embedding,
    const int64_t* codebook_offsets,
    const __nv_bfloat16* base_input,
    __nv_bfloat16* out,
    int num_codebooks,
    int token_stride,
    int sequence_length,
    int target_length,
    int embedding_dim,
    cudaStream_t stream);

void omnivoice_argmax_scores_f32(
    const float* log_probs,
    const float* layer_penalty,
    int64_t* predictions,
    float* scores,
    int rows,
    int cols,
    int target_length,
    int mask_col,
    cudaStream_t stream);

void omnivoice_maskgit_topk_update(
    const float* scores,
    const int64_t* predictions,
    int64_t* tokens,
    int elements,
    int k,
    int64_t mask_id,
    cudaStream_t stream);

void omnivoice_maskgit_topk_gumbel_update(
    const float* scores,
    const float* uniforms,
    const int64_t* predictions,
    int64_t* tokens,
    int elements,
    int k,
    int64_t mask_id,
    float temperature,
    cudaStream_t stream);

void omnivoice_maskgit_topk_scheduled_update(
    const float* scores,
    const int64_t* predictions,
    int64_t* tokens,
    const int* schedule,
    int* step,
    int elements,
    int schedule_length,
    int64_t mask_id,
    cudaStream_t stream);

void omnivoice_maskgit_topk_gumbel_scheduled_update(
    const float* scores,
    const float* uniforms,
    const int64_t* predictions,
    int64_t* tokens,
    const int* schedule,
    int* step,
    int elements,
    int schedule_length,
    int64_t mask_id,
    float temperature,
    cudaStream_t stream);
