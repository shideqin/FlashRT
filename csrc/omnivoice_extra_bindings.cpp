// ================================================================
// flash_rt_omnivoice — standalone pybind module for OmniVoice-specific
// fused kernels (kept separate from flash_rt_kernels so they can be
// added/rebuilt independently without touching the main bindings).
//
// Kernels: cfg_combine_log_softmax_bf16, maskgit_sample_row_bf16.
// ================================================================
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/cfg_combine.cuh"
#include "kernels/maskgit_sample.cuh"
#include "kernels/maskgit_select.cuh"

namespace py = pybind11;

static inline cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

PYBIND11_MODULE(flash_rt_omnivoice, m) {
    m.doc() = "OmniVoice fused kernels (cfg_combine, maskgit_sample)";

    // Fused CFG combine: out = log_softmax(c_lp + gs*(c_lp - u_lp))
    m.def("cfg_combine_log_softmax_bf16",
        [](uintptr_t c_logits, uintptr_t u_logits, uintptr_t out,
           int rows, int cols, int mask_col, double guidance_scale,
           uintptr_t stream) {
            cfg_combine_log_softmax_bf16(
                reinterpret_cast<const __nv_bfloat16*>(c_logits),
                reinterpret_cast<const __nv_bfloat16*>(u_logits),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, cols, mask_col, (float)guidance_scale, to_stream(stream));
        },
        py::arg("c_logits"), py::arg("u_logits"), py::arg("out"),
        py::arg("rows"), py::arg("cols"), py::arg("mask_col"),
        py::arg("guidance_scale"), py::arg("stream") = 0,
        "Fused CFG combine (BF16). Replaces 3x torch log_softmax.");

    // MaskGIT per-row sample: filter_top_k + gumbel + argmax + confidence
    m.def("maskgit_sample_row_bf16",
        [](uintptr_t log_probs, uintptr_t pred_tokens, uintptr_t confidence,
           int rows, int V, int num_filt, int mask_id, double class_temp,
           unsigned long long seed, uintptr_t stream) {
            maskgit_sample_row_bf16(
                reinterpret_cast<const __nv_bfloat16*>(log_probs),
                reinterpret_cast<int*>(pred_tokens),
                reinterpret_cast<__nv_bfloat16*>(confidence),
                rows, V, num_filt, mask_id, (float)class_temp, seed,
                to_stream(stream));
        },
        py::arg("log_probs"), py::arg("pred_tokens"), py::arg("confidence"),
        py::arg("rows"), py::arg("V"), py::arg("num_filt"),
        py::arg("mask_id"), py::arg("class_temp"), py::arg("seed") = 0,
        py::arg("stream") = 0,
        "MaskGIT per-row sample (BF16): filter+gumbel+argmax+confidence.");

    // MaskGIT cross-position select: score + position-gumbel + mask + topk(k_dev) + scatter
    m.def("maskgit_select_topk_bf16",
        [](uintptr_t confidence, uintptr_t pred_tokens, uintptr_t sample_tokens,
           uintptr_t k_dev, int B, int C, int T, double layer_penalty_factor,
           double position_temp, int mask_id, unsigned long long seed,
           uintptr_t stream) {
            maskgit_select_topk_bf16(
                reinterpret_cast<const __nv_bfloat16*>(confidence),
                reinterpret_cast<const int*>(pred_tokens),
                reinterpret_cast<int*>(sample_tokens),
                reinterpret_cast<const int*>(k_dev),
                B, C, T, (float)layer_penalty_factor, (float)position_temp,
                mask_id, seed, to_stream(stream));
        },
        py::arg("confidence"), py::arg("pred_tokens"), py::arg("sample_tokens"),
        py::arg("k_dev"), py::arg("B"), py::arg("C"), py::arg("T"),
        py::arg("layer_penalty_factor"), py::arg("position_temp"),
        py::arg("mask_id"), py::arg("seed") = 0, py::arg("stream") = 0,
        "MaskGIT cross-position select (BF16): score+gumbel+mask+topk(k)+scatter.");
}
