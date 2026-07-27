// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — OmniVoice fused kernels.
//
// Single-kernel warp-per-head Q/K normalization + RoPE rotation.
// Replaces qkv_split + rms_norm + rope_apply (3 kernel launches → 1).
//
// Pipeline:
//   1. Read Q and K from strided Dq buffer
//   2. Apply per-head RMSNorm (Q and K separately)
//   3. Apply RoPE — warp-shuffle RMS reduction, register-based
//   4. Write RoPE results to q_temp/k_temp output buffers

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

void omnivoice_rms_apply_exact_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const float* inv_rms,
    __nv_bfloat16* out,
    int rows, int dim,
    cudaStream_t stream);

void omnivoice_rms_apply_from_mean_exact_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const float* mean_square,
    __nv_bfloat16* out,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream);

void omnivoice_square_bf16_f32(
    const __nv_bfloat16* x,
    float* out,
    int elements,
    cudaStream_t stream);

void omnivoice_residual_add_square_bf16_f32(
    __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    float* square,
    int elements,
    cudaStream_t stream);

void omnivoice_square_mean_1024_bf16_f32(
    const __nv_bfloat16* x,
    float* mean_square,
    int rows,
    cudaStream_t stream);

void omnivoice_residual_add_square_mean_1024_bf16_f32(
    __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    float* mean_square,
    int rows,
    cudaStream_t stream);

void omnivoice_rms_norm_exact_1024_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int rows,
    float eps,
    cudaStream_t stream);

void omnivoice_residual_add_rms_norm_exact_1024_bf16(
    __nv_bfloat16* residual,
    const __nv_bfloat16* update,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int rows,
    float eps,
    cudaStream_t stream);

void omnivoice_silu_mul_merged_bf16(
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* out,
    int rows,
    int dim,
    cudaStream_t stream);

void omnivoice_expand_kv_bf16(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* k_expanded,
    __nv_bfloat16* v_expanded,
    int batch,
    int seq,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int v_row_stride,
    cudaStream_t stream);

void omnivoice_qk_norm_rope_bf16(
    const __nv_bfloat16* dq,         // [BS, QKVD] = [BS, NH*HD + 2*NKV*HD]
    const __nv_bfloat16* q_weight,   // [HD]  Q norm weight
    const __nv_bfloat16* k_weight,   // [HD]  K norm weight
    const __nv_bfloat16* cos,        // [BS, rope_dim]  RoPE cos
    const __nv_bfloat16* sin,        // [BS, rope_dim]  RoPE sin
    __nv_bfloat16* q_temp,           // RoPE output
    __nv_bfloat16* k_temp,           // RoPE output
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream);

void omnivoice_qk_norm_rope_expand_kv_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_expanded,
    __nv_bfloat16* v_expanded,
    int batch,
    int seq,
    int NH,
    int NKV,
    int HD,
    int QKVD,
    int active_rows,
    float eps,
    cudaStream_t stream);

void omnivoice_qk_norm_rope_expand_packed_kv_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_expanded,
    __nv_bfloat16* v_expanded,
    int batch,
    int seq,
    int NH,
    int NKV,
    int HD,
    int QKVD,
    int active_rows,
    float eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
