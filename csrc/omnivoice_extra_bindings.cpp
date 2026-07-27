// ================================================================
// flash_rt_omnivoice — standalone pybind module for OmniVoice-specific
// fused kernels (kept separate from flash_rt_kernels so they can be
// added/rebuilt independently without touching the main bindings).
//
// Kernels: exact RMS helpers, CFG combine, QK norm/RoPE.
// ================================================================
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/omnivoice/omnivoice_cfg_combine.cuh"
#include "kernels/omnivoice/omnivoice_bf16_dual_swiglu.cuh"
#include "kernels/omnivoice/omnivoice_bf16_matmul.cuh"
#include "kernels/omnivoice/omnivoice_exact_attention.cuh"
#include "kernels/omnivoice/omnivoice_qk_norm_rope.cuh"

namespace py = pybind11;

static inline cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

PYBIND11_MODULE(flash_rt_omnivoice, m) {
    m.def("omnivoice_bf16_dual_swiglu",
        [](uintptr_t x, uintptr_t gate_weight, uintptr_t up_weight,
           uintptr_t out, int rows, int hidden_dim, int input_dim,
           uintptr_t stream) {
            return flash_rt::kernels::omnivoice_bf16_dual_swiglu(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(gate_weight),
                reinterpret_cast<const __nv_bfloat16*>(up_weight),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows,hidden_dim,input_dim,to_stream(stream));
        },
        py::arg("x"),py::arg("gate_weight"),py::arg("up_weight"),
        py::arg("out"),py::arg("rows"),py::arg("hidden_dim"),
        py::arg("input_dim"),py::arg("stream")=0);

    m.def("omnivoice_bf16_matmul_reference_algo",
        [](uintptr_t x, uintptr_t weight, uintptr_t out,
           int M, int N, int K, int reference_M, uintptr_t stream) {
            flash_rt::kernels::omnivoice_bf16_matmul_reference_algo(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                reinterpret_cast<__nv_bfloat16*>(out),
                M, N, K, reference_M,
                reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("weight"), py::arg("out"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("reference_M"), py::arg("stream") = 0);

    m.def("omnivoice_bf16_matmul_heuristic_algo",
        [](uintptr_t x, uintptr_t weight, uintptr_t out,
           int M, int N, int K, int reference_M, int algorithm_rank,
           uintptr_t stream) {
            flash_rt::kernels::omnivoice_bf16_matmul_heuristic_algo(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                reinterpret_cast<__nv_bfloat16*>(out),
                M, N, K, reference_M, algorithm_rank,
                to_stream(stream));
        },
        py::arg("x"), py::arg("weight"), py::arg("out"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("reference_M"), py::arg("algorithm_rank"),
        py::arg("stream") = 0);

    m.def("omnivoice_exact_packed_attention_bf16",
        [](uintptr_t query, uintptr_t key, uintptr_t value,
           uintptr_t output, uintptr_t cu_seqlens, int total_rows,
           int max_seqlen, int num_sequences, int num_heads,
           int head_dim, float scale, uintptr_t stream) {
            flash_rt::kernels::omnivoice_exact_packed_attention_bf16(
                reinterpret_cast<const __nv_bfloat16*>(query),
                reinterpret_cast<const __nv_bfloat16*>(key),
                reinterpret_cast<const __nv_bfloat16*>(value),
                reinterpret_cast<__nv_bfloat16*>(output),
                reinterpret_cast<const int32_t*>(cu_seqlens),
                total_rows, max_seqlen, num_sequences, num_heads,
                head_dim, scale, to_stream(stream));
        },
        py::arg("query"), py::arg("key"), py::arg("value"),
        py::arg("output"), py::arg("cu_seqlens"),
        py::arg("total_rows"), py::arg("max_seqlen"),
        py::arg("num_sequences"), py::arg("num_heads"),
        py::arg("head_dim"), py::arg("scale"),
        py::arg("stream") = 0,
        "Run exact dropout-free packed BF16 attention.");

    m.def("omnivoice_exact_packed_attention_compact_bf16",
        [](uintptr_t query, uintptr_t key, uintptr_t value,
           uintptr_t output, int total_rows, int max_seqlen,
           int num_heads, int head_dim, float scale, uintptr_t stream) {
            flash_rt::kernels::
                omnivoice_exact_packed_attention_compact_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(query),
                    reinterpret_cast<const __nv_bfloat16*>(key),
                    reinterpret_cast<const __nv_bfloat16*>(value),
                    reinterpret_cast<__nv_bfloat16*>(output),
                    total_rows, max_seqlen, num_heads, head_dim,
                    scale, to_stream(stream));
        },
        py::arg("query"), py::arg("key"), py::arg("value"),
        py::arg("output"), py::arg("total_rows"),
        py::arg("max_seqlen"), py::arg("num_heads"),
        py::arg("head_dim"), py::arg("scale"),
        py::arg("stream") = 0,
        "Run exact two-sequence packed BF16 attention on a compact grid.");

    m.doc() = "OmniVoice fused kernels";

    m.def("omnivoice_maskgit_topk_update",
        [](uintptr_t scores, uintptr_t predictions, uintptr_t tokens,
           int elements, int k, int64_t mask_id, uintptr_t stream) {
            omnivoice_maskgit_topk_update(
                reinterpret_cast<const float*>(scores),
                reinterpret_cast<const int64_t*>(predictions),
                reinterpret_cast<int64_t*>(tokens),
                elements, k, mask_id, to_stream(stream));
        },
        py::arg("scores"), py::arg("predictions"), py::arg("tokens"),
        py::arg("elements"), py::arg("k"), py::arg("mask_id"),
        py::arg("stream") = 0,
        "Fuse MaskGIT masking, top-k selection, and token update.");

    m.def("omnivoice_maskgit_topk_gumbel_update",
        [](uintptr_t scores, uintptr_t uniforms,
           uintptr_t predictions, uintptr_t tokens,
           int elements, int k, int64_t mask_id,
           float temperature, uintptr_t stream) {
            omnivoice_maskgit_topk_gumbel_update(
                reinterpret_cast<const float*>(scores),
                reinterpret_cast<const float*>(uniforms),
                reinterpret_cast<const int64_t*>(predictions),
                reinterpret_cast<int64_t*>(tokens),
                elements, k, mask_id, temperature, to_stream(stream));
        },
        py::arg("scores"), py::arg("uniforms"),
        py::arg("predictions"), py::arg("tokens"),
        py::arg("elements"), py::arg("k"), py::arg("mask_id"),
        py::arg("temperature"), py::arg("stream") = 0,
        "Fuse position Gumbel scoring, masking, top-k, and token update.");

    m.def("omnivoice_maskgit_topk_scheduled_update",
        [](uintptr_t scores, uintptr_t predictions, uintptr_t tokens,
           uintptr_t schedule, uintptr_t step, int elements,
           int schedule_length, int64_t mask_id, uintptr_t stream) {
            omnivoice_maskgit_topk_scheduled_update(
                reinterpret_cast<const float*>(scores),
                reinterpret_cast<const int64_t*>(predictions),
                reinterpret_cast<int64_t*>(tokens),
                reinterpret_cast<const int*>(schedule),
                reinterpret_cast<int*>(step),
                elements, schedule_length, mask_id, to_stream(stream));
        },
        py::arg("scores"), py::arg("predictions"), py::arg("tokens"),
        py::arg("schedule"), py::arg("step"), py::arg("elements"),
        py::arg("schedule_length"), py::arg("mask_id"),
        py::arg("stream") = 0,
        "Run scheduled Top-K/token update and advance a device step.");

    m.def("omnivoice_maskgit_topk_gumbel_scheduled_update",
        [](uintptr_t scores, uintptr_t uniforms,
           uintptr_t predictions, uintptr_t tokens,
           uintptr_t schedule, uintptr_t step, int elements,
           int schedule_length, int64_t mask_id,
           float temperature, uintptr_t stream) {
            omnivoice_maskgit_topk_gumbel_scheduled_update(
                reinterpret_cast<const float*>(scores),
                reinterpret_cast<const float*>(uniforms),
                reinterpret_cast<const int64_t*>(predictions),
                reinterpret_cast<int64_t*>(tokens),
                reinterpret_cast<const int*>(schedule),
                reinterpret_cast<int*>(step),
                elements, schedule_length, mask_id, temperature,
                to_stream(stream));
        },
        py::arg("scores"), py::arg("uniforms"),
        py::arg("predictions"), py::arg("tokens"),
        py::arg("schedule"), py::arg("step"), py::arg("elements"),
        py::arg("schedule_length"), py::arg("mask_id"),
        py::arg("temperature"), py::arg("stream") = 0,
        "Run scheduled Gumbel Top-K/token update and advance a device step.");

    m.def("omnivoice_cfg_combine_f32",
        [](uintptr_t conditional_log_probs,
           uintptr_t unconditional_log_probs,
           uintptr_t out, int elements, float guidance_scale,
           uintptr_t stream) {
            omnivoice_cfg_combine_f32(
                reinterpret_cast<const float*>(conditional_log_probs),
                reinterpret_cast<const float*>(unconditional_log_probs),
                reinterpret_cast<float*>(out),
                elements, guidance_scale, to_stream(stream));
        },
        py::arg("conditional_log_probs"),
        py::arg("unconditional_log_probs"), py::arg("out"),
        py::arg("elements"), py::arg("guidance_scale"),
        py::arg("stream") = 0,
        "Fuse the exact FP32 CFG subtract, scale, and add operations.");

    m.def("omnivoice_logits_bf16_cfg_argmax_scores_f32",
        [](uintptr_t packed_logits, uintptr_t layer_penalty,
           uintptr_t predictions, uintptr_t scores,
           int rows, int cols, int target_length, int mask_col,
           float guidance_scale, uintptr_t stream) {
            omnivoice_logits_bf16_cfg_argmax_scores_f32(
                reinterpret_cast<const __nv_bfloat16*>(packed_logits),
                reinterpret_cast<const float*>(layer_penalty),
                reinterpret_cast<int64_t*>(predictions),
                reinterpret_cast<float*>(scores),
                rows, cols, target_length, mask_col, guidance_scale,
                to_stream(stream));
        },
        py::arg("packed_logits"), py::arg("layer_penalty"),
        py::arg("predictions"), py::arg("scores"), py::arg("rows"),
        py::arg("cols"), py::arg("target_length"), py::arg("mask_col"),
        py::arg("guidance_scale"), py::arg("stream") = 0,
        "Fuse BF16 logits promotion, three exact FP32 log-softmax "
        "operations, CFG, and selection.");

    m.def("omnivoice_embed_codebooks_bf16",
        [](uintptr_t tokens, uintptr_t embedding,
           uintptr_t codebook_offsets, uintptr_t conditional_out,
           uintptr_t unconditional_out, int num_codebooks,
           int token_stride, int target_length, int embedding_dim,
           uintptr_t stream) {
            omnivoice_embed_codebooks_bf16(
                reinterpret_cast<const int64_t*>(tokens),
                reinterpret_cast<const __nv_bfloat16*>(embedding),
                reinterpret_cast<const int64_t*>(codebook_offsets),
                reinterpret_cast<__nv_bfloat16*>(conditional_out),
                reinterpret_cast<__nv_bfloat16*>(unconditional_out),
                num_codebooks, token_stride, target_length,
                embedding_dim, to_stream(stream));
        },
        py::arg("tokens"), py::arg("embedding"),
        py::arg("codebook_offsets"), py::arg("conditional_out"),
        py::arg("unconditional_out"), py::arg("num_codebooks"),
        py::arg("token_stride"), py::arg("target_length"),
        py::arg("embedding_dim"), py::arg("stream") = 0,
        "Fuse codebook embedding gather, FP32 sum, and CFG input copies.");

    m.def("omnivoice_prepare_cfg_embeddings_bf16",
        [](uintptr_t tokens, uintptr_t embedding,
           uintptr_t codebook_offsets, uintptr_t base_input,
           uintptr_t out, int num_codebooks, int token_stride,
           int sequence_length, int target_length, int embedding_dim,
           uintptr_t stream) {
            omnivoice_prepare_cfg_embeddings_bf16(
                reinterpret_cast<const int64_t*>(tokens),
                reinterpret_cast<const __nv_bfloat16*>(embedding),
                reinterpret_cast<const int64_t*>(codebook_offsets),
                reinterpret_cast<const __nv_bfloat16*>(base_input),
                reinterpret_cast<__nv_bfloat16*>(out),
                num_codebooks, token_stride, sequence_length,
                target_length, embedding_dim, to_stream(stream));
        },
        py::arg("tokens"), py::arg("embedding"),
        py::arg("codebook_offsets"), py::arg("base_input"),
        py::arg("out"), py::arg("num_codebooks"),
        py::arg("token_stride"), py::arg("sequence_length"),
        py::arg("target_length"), py::arg("embedding_dim"),
        py::arg("stream") = 0,
        "Restore static CFG rows and generate both target embedding spans.");

    m.def("omnivoice_argmax_scores_f32",
        [](uintptr_t log_probs, uintptr_t layer_penalty,
           uintptr_t predictions, uintptr_t scores,
           int rows, int cols, int target_length, int mask_col,
           uintptr_t stream) {
            omnivoice_argmax_scores_f32(
                reinterpret_cast<const float*>(log_probs),
                reinterpret_cast<const float*>(layer_penalty),
                reinterpret_cast<int64_t*>(predictions),
                reinterpret_cast<float*>(scores),
                rows, cols, target_length, mask_col,
                to_stream(stream));
        },
        py::arg("log_probs"), py::arg("layer_penalty"),
        py::arg("predictions"), py::arg("scores"),
        py::arg("rows"), py::arg("cols"), py::arg("target_length"),
        py::arg("mask_col"), py::arg("stream") = 0,
        "Fuse MaskGIT argmax, confidence max, and layer penalty.");

    m.def("omnivoice_rms_apply_exact_bf16",
        [](uintptr_t x, uintptr_t weight, uintptr_t inv_rms,
           uintptr_t out, int rows, int dim, uintptr_t stream) {
            flash_rt::kernels::omnivoice_rms_apply_exact_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                reinterpret_cast<const float*>(inv_rms),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, dim, to_stream(stream));
        },
        py::arg("x"), py::arg("weight"), py::arg("inv_rms"),
        py::arg("out"), py::arg("rows"), py::arg("dim"),
        py::arg("stream") = 0,
        "Apply a PyTorch-computed inverse RMS with exact BF16 rounding.");

    m.def("omnivoice_rms_apply_from_mean_exact_bf16",
        [](uintptr_t x, uintptr_t weight, uintptr_t mean_square,
           uintptr_t out, int rows, int dim, float eps, uintptr_t stream) {
            flash_rt::kernels::omnivoice_rms_apply_from_mean_exact_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                reinterpret_cast<const float*>(mean_square),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, dim, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("weight"), py::arg("mean_square"),
        py::arg("out"), py::arg("rows"), py::arg("dim"),
        py::arg("eps"), py::arg("stream") = 0,
        "Apply exact RMSNorm from a PyTorch-reduced FP32 mean square.");

    m.def("omnivoice_square_bf16_f32",
        [](uintptr_t x, uintptr_t out,
           int elements, uintptr_t stream) {
            flash_rt::kernels::omnivoice_square_bf16_f32(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<float*>(out),
                elements, to_stream(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("elements"),
        py::arg("stream") = 0,
        "Convert BF16 to FP32 and square with PyTorch-compatible rounding.");

    m.def("omnivoice_residual_add_square_bf16_f32",
        [](uintptr_t residual, uintptr_t update, uintptr_t square,
           int elements, uintptr_t stream) {
            flash_rt::kernels::omnivoice_residual_add_square_bf16_f32(
                reinterpret_cast<__nv_bfloat16*>(residual),
                reinterpret_cast<const __nv_bfloat16*>(update),
                reinterpret_cast<float*>(square),
                elements, to_stream(stream));
        },
        py::arg("residual"), py::arg("update"), py::arg("square"),
        py::arg("elements"), py::arg("stream") = 0,
        "BF16 residual add followed by exact FP32 square preparation.");

    m.def("omnivoice_square_mean_1024_bf16_f32",
        [](uintptr_t x, uintptr_t mean_square, int rows,
           uintptr_t stream) {
            flash_rt::kernels::omnivoice_square_mean_1024_bf16_f32(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<float*>(mean_square),
                rows, to_stream(stream));
        },
        py::arg("x"), py::arg("mean_square"), py::arg("rows"),
        py::arg("stream") = 0,
        "Exact ATen-order square/mean reduction for rows of width 1024.");

    m.def("omnivoice_residual_add_square_mean_1024_bf16_f32",
        [](uintptr_t residual, uintptr_t update, uintptr_t mean_square,
           int rows, uintptr_t stream) {
            flash_rt::kernels::
                omnivoice_residual_add_square_mean_1024_bf16_f32(
                    reinterpret_cast<__nv_bfloat16*>(residual),
                    reinterpret_cast<const __nv_bfloat16*>(update),
                    reinterpret_cast<float*>(mean_square),
                    rows, to_stream(stream));
        },
        py::arg("residual"), py::arg("update"),
        py::arg("mean_square"), py::arg("rows"),
        py::arg("stream") = 0,
        "Exact BF16 residual add and ATen-order square/mean reduction.");

    m.def("omnivoice_rms_norm_exact_1024_bf16",
        [](uintptr_t x, uintptr_t weight, uintptr_t out,
           int rows, float eps, uintptr_t stream) {
            flash_rt::kernels::omnivoice_rms_norm_exact_1024_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("weight"), py::arg("out"),
        py::arg("rows"), py::arg("eps"), py::arg("stream") = 0,
        "Exact full RMSNorm for rows of width 1024.");

    m.def("omnivoice_residual_add_rms_norm_exact_1024_bf16",
        [](uintptr_t residual, uintptr_t update, uintptr_t weight,
           uintptr_t out, int rows, float eps, uintptr_t stream) {
            flash_rt::kernels::
                omnivoice_residual_add_rms_norm_exact_1024_bf16(
                    reinterpret_cast<__nv_bfloat16*>(residual),
                    reinterpret_cast<const __nv_bfloat16*>(update),
                    reinterpret_cast<const __nv_bfloat16*>(weight),
                    reinterpret_cast<__nv_bfloat16*>(out),
                    rows, eps, to_stream(stream));
        },
        py::arg("residual"), py::arg("update"), py::arg("weight"),
        py::arg("out"), py::arg("rows"), py::arg("eps"),
        py::arg("stream") = 0,
        "Exact BF16 residual add and full RMSNorm for width 1024.");

    m.def("omnivoice_silu_mul_merged_bf16",
        [](uintptr_t gate_up, uintptr_t out, int rows, int dim,
           uintptr_t stream) {
            flash_rt::kernels::omnivoice_silu_mul_merged_bf16(
                reinterpret_cast<const __nv_bfloat16*>(gate_up),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, dim, to_stream(stream));
        },
        py::arg("gate_up"), py::arg("out"), py::arg("rows"),
        py::arg("dim"), py::arg("stream") = 0,
        "Exact BF16 SiLU and multiply from a merged [gate, up] tensor.");

    m.def("omnivoice_expand_kv_bf16",
        [](uintptr_t k, uintptr_t v, uintptr_t k_expanded,
           uintptr_t v_expanded, int batch, int seq, int num_heads,
           int num_kv_heads, int head_dim, int v_row_stride,
           uintptr_t stream) {
            flash_rt::kernels::omnivoice_expand_kv_bf16(
                reinterpret_cast<const __nv_bfloat16*>(k),
                reinterpret_cast<const __nv_bfloat16*>(v),
                reinterpret_cast<__nv_bfloat16*>(k_expanded),
                reinterpret_cast<__nv_bfloat16*>(v_expanded),
                batch, seq, num_heads, num_kv_heads, head_dim,
                v_row_stride, to_stream(stream));
        },
        py::arg("k"), py::arg("v"), py::arg("k_expanded"),
        py::arg("v_expanded"), py::arg("batch"), py::arg("seq"),
        py::arg("num_heads"), py::arg("num_kv_heads"),
        py::arg("head_dim"), py::arg("v_row_stride"),
        py::arg("stream") = 0,
        "Expand GQA K/V heads into SDPA's contiguous [B,H,S,D] layout.");

    m.def("omnivoice_cfg_logsoftmax_bf16",
        [](uintptr_t c_logits, uintptr_t u_logits, uintptr_t out,
           int rows, int cols, int mask_col, double guidance_scale,
           uintptr_t stream) {
            omnivoice_cfg_logsoftmax_bf16(
                reinterpret_cast<const __nv_bfloat16*>(c_logits),
                reinterpret_cast<const __nv_bfloat16*>(u_logits),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, cols, mask_col, (float)guidance_scale, to_stream(stream));
        },
        py::arg("c_logits"), py::arg("u_logits"), py::arg("out"),
        py::arg("rows"), py::arg("cols"), py::arg("mask_col"),
        py::arg("guidance_scale"), py::arg("stream") = 0,
        "Fused CFG combine (BF16). Replaces 3x torch log_softmax.");

    m.def("omnivoice_qk_norm_rope_bf16",
        [](uintptr_t dq, uintptr_t q_weight, uintptr_t k_weight,
           uintptr_t cos, uintptr_t sin,
           uintptr_t q_temp, uintptr_t k_temp,
           int BS, int NH, int NKV, int HD, int QKVD, float eps,
           uintptr_t stream) {
            flash_rt::kernels::omnivoice_qk_norm_rope_bf16(
                reinterpret_cast<const __nv_bfloat16*>(dq),
                reinterpret_cast<const __nv_bfloat16*>(q_weight),
                reinterpret_cast<const __nv_bfloat16*>(k_weight),
                reinterpret_cast<const __nv_bfloat16*>(cos),
                reinterpret_cast<const __nv_bfloat16*>(sin),
                reinterpret_cast<__nv_bfloat16*>(q_temp),
                reinterpret_cast<__nv_bfloat16*>(k_temp),
                BS, NH, NKV, HD, QKVD, eps,
                reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("dq"), py::arg("q_weight"), py::arg("k_weight"),
        py::arg("cos"), py::arg("sin"),
        py::arg("q_temp"), py::arg("k_temp"),
        py::arg("BS"), py::arg("NH"), py::arg("NKV"), py::arg("HD"),
        py::arg("QKVD"), py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused Q/K RMSNorm + RoPE (warp-per-head). "
        "Replaces qkv_split + norm + rope (3 launches → 1).");

    m.def("omnivoice_qk_norm_rope_expand_kv_bf16",
        [](uintptr_t dq, uintptr_t q_weight, uintptr_t k_weight,
           uintptr_t cos, uintptr_t sin, uintptr_t q_temp,
           uintptr_t k_expanded, uintptr_t v_expanded,
           int batch, int seq, int NH, int NKV, int HD, int QKVD,
           int active_rows, float eps, uintptr_t stream) {
            flash_rt::kernels::omnivoice_qk_norm_rope_expand_kv_bf16(
                reinterpret_cast<const __nv_bfloat16*>(dq),
                reinterpret_cast<const __nv_bfloat16*>(q_weight),
                reinterpret_cast<const __nv_bfloat16*>(k_weight),
                reinterpret_cast<const __nv_bfloat16*>(cos),
                reinterpret_cast<const __nv_bfloat16*>(sin),
                reinterpret_cast<__nv_bfloat16*>(q_temp),
                reinterpret_cast<__nv_bfloat16*>(k_expanded),
                reinterpret_cast<__nv_bfloat16*>(v_expanded),
                batch, seq, NH, NKV, HD, QKVD, active_rows, eps,
                to_stream(stream));
        },
        py::arg("dq"), py::arg("q_weight"), py::arg("k_weight"),
        py::arg("cos"), py::arg("sin"), py::arg("q_temp"),
        py::arg("k_expanded"), py::arg("v_expanded"),
        py::arg("batch"), py::arg("seq"), py::arg("NH"),
        py::arg("NKV"), py::arg("HD"), py::arg("QKVD"),
        py::arg("active_rows"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Exact fused Q/K RMSNorm + RoPE with expanded SDPA K/V output.");

    m.def("omnivoice_qk_norm_rope_expand_packed_kv_bf16",
        [](uintptr_t dq, uintptr_t q_weight, uintptr_t k_weight,
           uintptr_t cos, uintptr_t sin, uintptr_t q_temp,
           uintptr_t k_expanded, uintptr_t v_expanded,
           int batch, int seq, int NH, int NKV, int HD, int QKVD,
           int active_rows, float eps, uintptr_t stream) {
            flash_rt::kernels::
                omnivoice_qk_norm_rope_expand_packed_kv_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(dq),
                    reinterpret_cast<const __nv_bfloat16*>(q_weight),
                    reinterpret_cast<const __nv_bfloat16*>(k_weight),
                    reinterpret_cast<const __nv_bfloat16*>(cos),
                    reinterpret_cast<const __nv_bfloat16*>(sin),
                    reinterpret_cast<__nv_bfloat16*>(q_temp),
                    reinterpret_cast<__nv_bfloat16*>(k_expanded),
                    reinterpret_cast<__nv_bfloat16*>(v_expanded),
                    batch, seq, NH, NKV, HD, QKVD, active_rows, eps,
                    to_stream(stream));
        },
        py::arg("dq"), py::arg("q_weight"), py::arg("k_weight"),
        py::arg("cos"), py::arg("sin"), py::arg("q_temp"),
        py::arg("k_expanded"), py::arg("v_expanded"),
        py::arg("batch"), py::arg("seq"), py::arg("NH"),
        py::arg("NKV"), py::arg("HD"), py::arg("QKVD"),
        py::arg("active_rows"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Exact fused Q/K RMSNorm + RoPE with packed variable-length K/V.");
}
