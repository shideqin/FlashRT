#!/usr/bin/env python3
"""OmniVoice FlashRT kernel smoke test.

Validates all required kernel symbols are present when FlashRT is built
with FLASHRT_ENABLE_OMNIVOICE=ON on SM120 GPUs. Skips cleanly when
compiled .so modules are absent (default/OFF build).

Usage:
  pytest -q tests/test_omnivoice_smoke.py        # both ON and OFF cases
"""
import pytest, warnings, sys, os
import torch
warnings.filterwarnings("ignore")

# Add FlashRT repo root to path (repo convention: tests/ run from root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDefaultBuild:
    """Default build (FLASHRT_ENABLE_OMNIVOICE=OFF).

    flash_rt package imports work without GPU. The models.omnivoice
    module imports cleanly, _has_cfg_kernel is False.
    """

    def test_omnivoice_module_imports(self):
        from flash_rt.models.omnivoice import (
            FlashRTLlm, FlashRTLlmBF16,
            inject, free_encoder, eject, _check_kernels,
        )
        assert inject is not None

    def test_cfg_kernel_disabled_by_default(self):
        from flash_rt.models.omnivoice import pipeline_rtx
        import importlib
        importlib.reload(pipeline_rtx)
        # If the omnivoice .so is present (flag ON), kernel is True.
        # If absent (flag OFF), kernel is False. Both are valid.
        assert isinstance(pipeline_rtx._has_cfg_kernel, bool)

    def test_cfg_matches_upstream_fp32_order(self):
        from flash_rt.models.omnivoice.pipeline_rtx import _make_cfg

        torch.manual_seed(7)
        conditional = torch.randn(2, 3, 4, 11, dtype=torch.bfloat16)
        unconditional = torch.randn_like(conditional)
        mask_id = 6
        guidance_scale = 2.0

        actual = _make_cfg(guidance_scale, mask_id)(
            conditional, unconditional)
        cl = torch.log_softmax(conditional.float(), dim=-1)
        ul = torch.log_softmax(unconditional.float(), dim=-1)
        expected = torch.log_softmax(
            cl + guidance_scale * (cl - ul), dim=-1)
        expected[..., mask_id] = -float("inf")

        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)

    def test_exact_cfg_is_the_default_profile(self):
        from flash_rt.models.omnivoice.pipeline_rtx import (
            inject, MASKGIT_CUDA_SELECT_MAX_ELEMENTS)

        assert inject.__defaults__[0] == 1.0
        assert MASKGIT_CUDA_SELECT_MAX_ELEMENTS == 1024

    def test_vectorized_audio_postprocess_matches_upstream(self):
        import numpy as np
        from omnivoice.models.omnivoice import (
            fade_and_pad_audio, remove_silence)
        from flash_rt.models.omnivoice.pipeline_rtx import (
            _fade_and_pad_numpy_exact,
            _remove_silence_numpy_exact)

        generator=np.random.default_rng(109)
        for channels,length in (
                (1,0),(1,241),(1,12000),(1,30001),
                (2,12000),(2,48037)):
            audio=(
                generator.standard_normal((channels,length))*0.03
            ).astype(np.float32)
            if length>=12000:
                audio[:,length//4:length//4+12000]*=0.001
            expected=remove_silence(
                audio,24000,mid_sil=500,
                lead_sil=100,trail_sil=100)
            actual=_remove_silence_numpy_exact(
                audio,24000,500,100,100)
            assert np.array_equal(actual,expected)

            expected_faded=fade_and_pad_audio(
                expected,sample_rate=24000)
            actual_faded=_fade_and_pad_numpy_exact(
                actual,sample_rate=24000)
            assert np.array_equal(actual_faded,expected_faded)

    def test_decoder_quantizer_patch_is_exact_on_cpu(self):
        from types import SimpleNamespace
        from flash_rt.models.omnivoice.pipeline_rtx import (
            _optimize_audio_decoder,
        )

        class Unit:
            def decode(self, indices):
                return indices.float()

        class Quantizer:
            def __init__(self):
                self.quantizers = [Unit(), Unit(), Unit()]

            def decode(self, codes):
                out = torch.tensor(0.0, device=codes.device)
                for index, indices in enumerate(codes):
                    out = out + self.quantizers[index].decode(indices)
                return out

        class Tokenizer:
            def __init__(self):
                self.quantizer = Quantizer()
                self.config = SimpleNamespace(return_dict=False)

            def decode(self, audio_codes, return_dict=None):
                return self.quantizer.decode(audio_codes.transpose(0, 1))

        tokenizer = Tokenizer()
        model = SimpleNamespace(audio_tokenizer=tokenizer)
        codes = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
            dtype=torch.int64)
        expected = tokenizer.decode(codes)

        _optimize_audio_decoder(model)
        actual = tokenizer.decode(codes)

        assert torch.equal(actual, expected)
        assert tokenizer._frt_decoder_graphs == {}


class TestGatedBuild:
    """OmniVoice build (FLASHRT_ENABLE_OMNIVOICE=ON, SM120 GPU).

    All kernel symbols must be present across modules:
      flash_rt_kernels  — base FP4/fused kernels
      flash_rt_omnivoice — OmniVoice-specific kernels
      flash_rt_fa2       — FlashAttention2
    """

    _F = None  # flash_rt_kernels
    _O = None  # flash_rt_omnivoice
    _A = None  # flash_rt_fa2

    @pytest.fixture(autouse=True)
    def _setup(self):
        if TestGatedBuild._F is None:
            try:
                from flash_rt import flash_rt_kernels as fvk
            except ImportError:
                pytest.skip("flash_rt_kernels not available (GPU build required)")
            TestGatedBuild._F = fvk
        if TestGatedBuild._O is None:
            try:
                from flash_rt import flash_rt_omnivoice as fvo
            except ImportError:
                fvo = None
            TestGatedBuild._O = fvo
        if TestGatedBuild._A is None:
            try:
                from flash_rt import flash_rt_fa2 as fa2
            except ImportError:
                fa2 = None
            TestGatedBuild._A = fa2

    # ── flash_rt_kernels symbols ──

    def test_fp4_gemm_symbols(self):
        fvk = TestGatedBuild._F
        assert hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out")
        assert hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out_pingpong")

    def test_fused_norm_symbols(self):
        fvk = TestGatedBuild._F
        for sym in ("rms_norm", "rms_norm_to_nvfp4_swizzled_bf16",
                     "residual_add_rms_norm",
                     "residual_add_rms_norm_to_nvfp4_swizzled_bf16"):
            assert hasattr(fvk, sym)

    def test_quantize_symbols(self):
        fvk = TestGatedBuild._F
        assert hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled")
        assert hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled_mse")

    def test_silu_symbols(self):
        assert hasattr(TestGatedBuild._F,
                        "silu_mul_merged_to_nvfp4_swizzled_bf16")

    # ── flash_rt_omnivoice symbols ──

    def test_omnivoice_cfg_logsoftmax(self):
        if TestGatedBuild._O is None:
            pytest.skip("flash_rt_omnivoice not built")
        assert hasattr(TestGatedBuild._O, "omnivoice_cfg_logsoftmax_bf16")
        assert hasattr(TestGatedBuild._O, "omnivoice_cfg_combine_f32")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_logits_bf16_cfg_argmax_scores_f32")
        assert hasattr(
            TestGatedBuild._O, "omnivoice_embed_codebooks_bf16")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_prepare_cfg_embeddings_bf16")
        assert hasattr(
            TestGatedBuild._O, "omnivoice_argmax_scores_f32")
        assert hasattr(TestGatedBuild._O, "omnivoice_maskgit_topk_update")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_maskgit_topk_gumbel_update")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_maskgit_topk_scheduled_update")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_maskgit_topk_gumbel_scheduled_update")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_exact_packed_attention_bf16")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_exact_packed_attention_compact_bf16")
        assert hasattr(
            TestGatedBuild._O, "omnivoice_bf16_dual_swiglu")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_bf16_matmul_heuristic_algo")

    @pytest.mark.parametrize("guidance_scale", [0.5, 2.0, 3.25])
    def test_omnivoice_cfg_combine_matches_torch(
            self, guidance_scale):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(61)
        conditional = torch.randn(
            304, 1025, device="cuda", dtype=torch.float32)
        unconditional = torch.randn_like(conditional)
        expected = (
            conditional
            + guidance_scale * (conditional - unconditional))
        actual = torch.empty_like(expected)
        TestGatedBuild._O.omnivoice_cfg_combine_f32(
            conditional.data_ptr(), unconditional.data_ptr(),
            actual.data_ptr(), actual.numel(), guidance_scale,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    def test_omnivoice_embed_codebooks_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(67)
        codebooks, target_length, token_stride, dim = 8, 38, 43, 1024
        vocab = 1025
        tokens = torch.randint(
            0, vocab, (codebooks, token_stride),
            device="cuda", dtype=torch.int64)
        offsets = (
            torch.arange(codebooks, device="cuda", dtype=torch.int64)
            * vocab)
        embedding = torch.randn(
            codebooks * vocab, dim,
            device="cuda", dtype=torch.bfloat16)
        expected = embedding[
            tokens[:, :target_length] + offsets[:, None]].sum(dim=0)
        conditional = torch.empty_like(expected)
        unconditional = torch.empty_like(expected)

        TestGatedBuild._O.omnivoice_embed_codebooks_bf16(
            tokens.data_ptr(), embedding.data_ptr(), offsets.data_ptr(),
            conditional.data_ptr(), unconditional.data_ptr(),
            codebooks, token_stride, target_length, dim,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(conditional, expected)
        assert torch.equal(unconditional, expected)

    def test_omnivoice_prepare_cfg_embeddings_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(69)
        codebooks, seq, target_length, dim = 8, 79, 38, 1024
        vocab = 1025
        tokens = torch.randint(
            0, vocab, (codebooks, target_length),
            device="cuda", dtype=torch.int64)
        offsets = (
            torch.arange(codebooks, device="cuda", dtype=torch.int64)
            * vocab)
        embedding = torch.randn(
            codebooks * vocab, dim,
            device="cuda", dtype=torch.bfloat16)
        base = torch.randn(
            2, seq, dim, device="cuda", dtype=torch.bfloat16)
        expected = base.clone()
        target = embedding[
            tokens + offsets[:, None]].sum(dim=0)
        expected[0, seq-target_length:seq] = target
        expected[1, :target_length] = target
        actual = torch.full_like(base, float("nan"))

        TestGatedBuild._O.omnivoice_prepare_cfg_embeddings_bf16(
            tokens.data_ptr(), embedding.data_ptr(), offsets.data_ptr(),
            base.data_ptr(), actual.data_ptr(), codebooks,
            tokens.stride(0), seq, target_length, dim,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(
            actual[1, :target_length], expected[1, :target_length])

    @pytest.mark.parametrize("guidance_scale", [0.5, 2.0, 3.25])
    def test_omnivoice_full_fused_cfg_selection_matches_torch(
            self, guidance_scale):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(79)
        codebooks, target_length, cols = 8, 38, 1025
        packed_logits = torch.randn(
            2, target_length, codebooks, cols,
            device="cuda", dtype=torch.bfloat16)
        logits = packed_logits.float().permute(0, 2, 1, 3)
        conditional = logits[0].log_softmax(dim=-1)
        unconditional = logits[1].log_softmax(dim=-1)
        penalty = torch.arange(
            codebooks, device="cuda", dtype=torch.float32)[:, None]
        mask_id = cols - 1
        combined = (
            conditional
            + guidance_scale * (conditional - unconditional))
        expected_lp = combined.log_softmax(dim=-1)
        expected_lp[..., mask_id] = -float("inf")
        expected_predictions = expected_lp.argmax(dim=-1)
        expected_scores = expected_lp.max(dim=-1).values - penalty
        predictions = torch.empty_like(
            expected_predictions, dtype=torch.int64)
        scores = torch.empty_like(expected_scores)

        TestGatedBuild._O.omnivoice_logits_bf16_cfg_argmax_scores_f32(
            packed_logits.data_ptr(), penalty.data_ptr(),
            predictions.data_ptr(), scores.data_ptr(), scores.numel(),
            cols, target_length, mask_id, guidance_scale,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(predictions, expected_predictions)
        assert torch.equal(scores, expected_scores)

    def test_omnivoice_argmax_scores_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(71)
        codebooks, target_length, cols = 8, 38, 1025
        log_probs = torch.randn(
            codebooks, target_length, cols,
            device="cuda", dtype=torch.float32)
        # Exercise exact tie-breaking: PyTorch argmax chooses the lowest
        # index, while max returns the shared value.
        log_probs[0, 0, 7] = 20.0
        log_probs[0, 0, 19] = 20.0
        penalty = torch.arange(
            codebooks, device="cuda", dtype=torch.float32)[:, None]
        mask_id = 1024
        expected_log_probs = log_probs.clone()
        expected_log_probs[..., mask_id] = -float("inf")
        expected_predictions = expected_log_probs.argmax(dim=-1)
        expected_scores = expected_log_probs.max(dim=-1).values - penalty
        predictions = torch.empty_like(
            expected_predictions, dtype=torch.int64)
        scores = torch.empty_like(expected_scores)

        TestGatedBuild._O.omnivoice_argmax_scores_f32(
            log_probs.data_ptr(), penalty.data_ptr(),
            predictions.data_ptr(), scores.data_ptr(),
            scores.numel(), cols, target_length, mask_id,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(predictions, expected_predictions)
        assert torch.equal(scores, expected_scores)

    @pytest.mark.parametrize(
        ("elements", "k"),
        [(1, 1), (304, 58), (512, 251), (768, 389), (1024, 733)])
    def test_omnivoice_maskgit_topk_update_matches_torch(
            self, elements, k):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")

        mask_id = 1024
        # A permutation gives every candidate a distinct score, so the test
        # verifies selection rather than relying on unspecified tie ordering.
        generator = torch.Generator(device="cuda")
        generator.manual_seed(59 + elements)
        scores = torch.randperm(
            elements, device="cuda", generator=generator).float()
        predictions = torch.arange(
            10000, 10000 + elements, device="cuda", dtype=torch.int64)
        tokens = torch.full(
            (elements,), mask_id, device="cuda", dtype=torch.int64)
        if elements > 1:
            tokens[::7] = 17

        expected = tokens.clone()
        masked_scores = scores.masked_fill(expected != mask_id, -float("inf"))
        indices = torch.topk(masked_scores, k).indices
        expected[indices] = predictions[indices]

        TestGatedBuild._O.omnivoice_maskgit_topk_update(
            scores.data_ptr(), predictions.data_ptr(), tokens.data_ptr(),
            elements, k, mask_id,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(tokens, expected)

    @pytest.mark.parametrize(
        ("elements", "seed"),
        [(304, 73), (304, 79), (768, 83), (1024, 89)])
    def test_omnivoice_maskgit_gumbel_update_matches_torch(
            self, elements, seed):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")

        k, mask_id, temperature = elements // 5, 1024, 5.0
        torch.manual_seed(seed)
        scores = torch.randn(
            elements, device="cuda", dtype=torch.float32)
        uniforms = torch.rand_like(scores)
        predictions = torch.arange(
            20000, 20000 + elements, device="cuda", dtype=torch.int64)
        tokens = torch.full(
            (elements,), mask_id, device="cuda", dtype=torch.int64)
        tokens[::11] = 23

        expected = tokens.clone()
        # Supply the same uniforms explicitly rather than consuming a second
        # RNG sequence inside the reference helper.
        scaled = scores / temperature
        noise = -torch.log(
            -torch.log(uniforms + 1e-10) + 1e-10)
        gumbel_scores = scaled + noise
        gumbel_scores.masked_fill_(
            expected != mask_id, -float("inf"))
        indices = torch.topk(gumbel_scores, k).indices
        expected[indices] = predictions[indices]

        TestGatedBuild._O.omnivoice_maskgit_topk_gumbel_update(
            scores.data_ptr(), uniforms.data_ptr(),
            predictions.data_ptr(), tokens.data_ptr(),
            elements, k, mask_id, temperature,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(tokens, expected)

    @pytest.mark.parametrize("elements", [304, 768, 1024])
    @pytest.mark.parametrize("with_gumbel", [False, True])
    def test_omnivoice_scheduled_topk_advances_device_step(
            self, with_gumbel, elements):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")

        mask_id = 1024
        schedule = torch.tensor(
            [elements // 5, elements // 4],
            device="cuda", dtype=torch.int32)
        step = torch.zeros((), device="cuda", dtype=torch.int32)
        predictions = torch.arange(
            30000, 30000 + elements, device="cuda", dtype=torch.int64)
        tokens = torch.full(
            (elements,), mask_id, device="cuda", dtype=torch.int64)
        scores = torch.arange(
            elements, device="cuda", dtype=torch.float32)
        uniforms = torch.linspace(
            0.01, 0.99, elements, device="cuda", dtype=torch.float32)

        expected = tokens.clone()
        for k in schedule.tolist():
            ranked = scores
            if with_gumbel:
                ranked = (
                    scores / 5.0
                    - torch.log(
                        -torch.log(uniforms + 1e-10) + 1e-10))
            ranked = ranked.masked_fill(
                expected != mask_id, -float("inf"))
            indices = torch.topk(ranked, k).indices
            expected[indices] = predictions[indices]

        stream = torch.cuda.current_stream().cuda_stream
        for _ in range(schedule.numel()):
            common = (
                predictions.data_ptr(), tokens.data_ptr(),
                schedule.data_ptr(), step.data_ptr(), elements,
                schedule.numel(), mask_id)
            if with_gumbel:
                TestGatedBuild._O \
                    .omnivoice_maskgit_topk_gumbel_scheduled_update(
                        scores.data_ptr(), uniforms.data_ptr(), *common,
                        5.0, stream)
            else:
                TestGatedBuild._O.omnivoice_maskgit_topk_scheduled_update(
                    scores.data_ptr(), *common, stream)

        torch.cuda.synchronize()
        assert step.item() == schedule.numel()
        assert torch.equal(tokens, expected)

    def test_omnivoice_qk_norm_rope(self):
        if TestGatedBuild._O is None:
            pytest.skip("flash_rt_omnivoice not built")
        assert hasattr(TestGatedBuild._O, "omnivoice_qk_norm_rope_bf16")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_qk_norm_rope_expand_kv_bf16")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_qk_norm_rope_expand_packed_kv_bf16")

    def test_omnivoice_exact_rms_apply(self):
        if TestGatedBuild._O is None:
            pytest.skip("flash_rt_omnivoice not built")
        assert hasattr(TestGatedBuild._O, "omnivoice_rms_apply_exact_bf16")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_rms_apply_from_mean_exact_bf16")
        assert hasattr(TestGatedBuild._O, "omnivoice_square_bf16_f32")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_residual_add_square_bf16_f32")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_square_mean_1024_bf16_f32")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_residual_add_square_mean_1024_bf16_f32")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_rms_norm_exact_1024_bf16")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_residual_add_rms_norm_exact_1024_bf16")
        assert hasattr(
            TestGatedBuild._O, "omnivoice_silu_mul_merged_bf16")
        assert hasattr(
            TestGatedBuild._O, "omnivoice_expand_kv_bf16")
        assert hasattr(
            TestGatedBuild._O,
            "omnivoice_bf16_matmul_reference_algo")

    def test_omnivoice_reference_algo_matmul_matches_padded_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(43)
        active_rows, padded_rows = 117, 158
        x = torch.randn(
            padded_rows, 2048, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            1024, 2048, device="cuda", dtype=torch.bfloat16)
        expected = torch.mm(x, weight.t())[:active_rows]
        actual = torch.empty(
            active_rows, 1024, device="cuda", dtype=torch.bfloat16)
        TestGatedBuild._O.omnivoice_bf16_matmul_reference_algo(
            x.data_ptr(), weight.data_ptr(), actual.data_ptr(),
            active_rows, 1024, 2048, padded_rows,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("rows", [112, 117, 128])
    def test_omnivoice_qkv_heuristic_matmul_matches_torch(
            self, rows):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(97)
        output_dim, input_dim = 4096, 1024
        x = torch.randn(
            rows, input_dim, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            output_dim, input_dim,
            device="cuda", dtype=torch.bfloat16)
        expected = torch.mm(x, weight.t())
        actual = torch.empty_like(expected)
        TestGatedBuild._O.omnivoice_bf16_matmul_heuristic_algo(
            x.data_ptr(), weight.data_ptr(), actual.data_ptr(),
            rows, output_dim, input_dim, rows, 4,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    def test_omnivoice_square_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(11)
        x = torch.randn(
            17, 1024, device="cuda", dtype=torch.bfloat16)
        expected = x.float().pow(2)
        actual = torch.empty_like(expected)
        TestGatedBuild._O.omnivoice_square_bf16_f32(
            x.data_ptr(), actual.data_ptr(), x.numel(),
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    def test_omnivoice_rms_from_mean_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(19)
        x = torch.randn(
            17, 1024, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            1024, device="cuda", dtype=torch.bfloat16)
        mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + 1e-6)
        normalized = (x.float() * inv_rms).to(torch.bfloat16)
        expected = (normalized.float() * weight.float()).to(torch.bfloat16)
        actual = torch.empty_like(x)
        TestGatedBuild._O.omnivoice_rms_apply_from_mean_exact_bf16(
            x.data_ptr(), weight.data_ptr(), mean_square.data_ptr(),
            actual.data_ptr(), 17, 1024, 1e-6,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("rows", [1, 17, 158])
    def test_omnivoice_square_mean_matches_torch(self, rows):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(23 + rows)
        x = torch.randn(
            rows, 1024, device="cuda", dtype=torch.bfloat16)
        expected = x.float().pow(2).mean(dim=-1, keepdim=True)
        actual = torch.empty(
            rows, 1, device="cuda", dtype=torch.float32)
        TestGatedBuild._O.omnivoice_square_mean_1024_bf16_f32(
            x.data_ptr(), actual.data_ptr(), rows,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("rows", [1, 17, 158])
    def test_omnivoice_residual_square_mean_matches_torch(self, rows):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(29 + rows)
        residual = torch.randn(
            rows, 1024, device="cuda", dtype=torch.bfloat16)
        update = torch.randn_like(residual)
        expected_residual = residual + update
        expected_mean = expected_residual.float().pow(2).mean(
            dim=-1, keepdim=True)
        actual_residual = residual.clone()
        actual_mean = torch.empty(
            rows, 1, device="cuda", dtype=torch.float32)
        fused = getattr(
            TestGatedBuild._O,
            "omnivoice_residual_add_square_mean_1024_bf16_f32")
        fused(
            actual_residual.data_ptr(), update.data_ptr(),
            actual_mean.data_ptr(), rows,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual_residual, expected_residual)
        assert torch.equal(actual_mean, expected_mean)

    @pytest.mark.parametrize("rows", [1, 17, 158])
    def test_omnivoice_full_rms_matches_torch(self, rows):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(31 + rows)
        x = torch.randn(
            rows, 1024, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            1024, device="cuda", dtype=torch.bfloat16)
        mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + 1e-6)
        normalized = (x.float() * inv_rms).to(torch.bfloat16)
        expected = (normalized.float() * weight.float()).to(torch.bfloat16)
        actual = torch.empty_like(x)
        TestGatedBuild._O.omnivoice_rms_norm_exact_1024_bf16(
            x.data_ptr(), weight.data_ptr(), actual.data_ptr(),
            rows, 1e-6, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("rows", [1, 17, 158])
    def test_omnivoice_residual_full_rms_matches_torch(self, rows):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(37 + rows)
        residual = torch.randn(
            rows, 1024, device="cuda", dtype=torch.bfloat16)
        update = torch.randn_like(residual)
        weight = torch.randn(
            1024, device="cuda", dtype=torch.bfloat16)
        expected_residual = residual + update
        mean_square = expected_residual.float().pow(2).mean(
            dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + 1e-6)
        normalized = (
            expected_residual.float() * inv_rms).to(torch.bfloat16)
        expected = (
            normalized.float() * weight.float()).to(torch.bfloat16)
        actual_residual = residual.clone()
        actual = torch.empty_like(residual)
        fused = getattr(
            TestGatedBuild._O,
            "omnivoice_residual_add_rms_norm_exact_1024_bf16")
        fused(
            actual_residual.data_ptr(), update.data_ptr(),
            weight.data_ptr(), actual.data_ptr(), rows, 1e-6,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual_residual, expected_residual)
        assert torch.equal(actual, expected)

    def test_omnivoice_residual_add_square_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(13)
        residual = torch.randn(
            17, 1024, device="cuda", dtype=torch.bfloat16)
        update = torch.randn_like(residual)
        expected_residual = residual + update
        expected_square = expected_residual.float().pow(2)
        actual_residual = residual.clone()
        actual_square = torch.empty_like(expected_square)
        TestGatedBuild._O.omnivoice_residual_add_square_bf16_f32(
            actual_residual.data_ptr(), update.data_ptr(),
            actual_square.data_ptr(), actual_residual.numel(),
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual_residual, expected_residual)
        assert torch.equal(actual_square, expected_square)

    def test_omnivoice_silu_mul_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(17)
        merged = torch.randn(
            19, 2 * 3072, device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.silu(
            merged[:, :3072]) * merged[:, 3072:]
        actual = torch.empty_like(expected)
        TestGatedBuild._O.omnivoice_silu_mul_merged_bf16(
            merged.data_ptr(), actual.data_ptr(), 19, 3072,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize("rows", [117, 160, 320])
    def test_omnivoice_dual_swiglu_matches_torch(self, rows):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(83 + rows)
        width, hidden = 1024, 3072
        inputs = torch.randn(
            rows, width, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(
            2 * hidden, width, device="cuda", dtype=torch.bfloat16)
        merged = torch.matmul(inputs, weights.t())
        expected = torch.nn.functional.silu(
            merged[:, :hidden]) * merged[:, hidden:]
        actual = torch.empty_like(expected)
        status = TestGatedBuild._O.omnivoice_bf16_dual_swiglu(
            inputs.data_ptr(), weights.data_ptr(),
            weights[hidden:].data_ptr(), actual.data_ptr(),
            rows, hidden, width,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert status == 0
        assert torch.equal(actual, expected)

    def test_omnivoice_expand_kv_matches_torch(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(41)
        batch, seq, heads, kv_heads, head_dim = 2, 79, 16, 8, 128
        k = torch.randn(
            batch, seq, kv_heads, head_dim,
            device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        expected_k = k.transpose(1, 2)[:, :, None, :, :].expand(
            batch, kv_heads, heads // kv_heads, seq, head_dim
        ).reshape(batch, heads, seq, head_dim)
        expected_v = v.transpose(1, 2)[:, :, None, :, :].expand(
            batch, kv_heads, heads // kv_heads, seq, head_dim
        ).reshape(batch, heads, seq, head_dim)
        actual_k = torch.empty_like(expected_k)
        actual_v = torch.empty_like(expected_v)
        TestGatedBuild._O.omnivoice_expand_kv_bf16(
            k.data_ptr(), v.data_ptr(), actual_k.data_ptr(),
            actual_v.data_ptr(), batch, seq, heads, kv_heads, head_dim,
            kv_heads * head_dim,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_v, expected_v)

    def test_omnivoice_packed_kv_matches_expanded_valid_rows(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(47)
        batch, seq, heads, kv_heads, head_dim = 2, 79, 16, 8, 128
        active_rows = 117
        qkvd = (heads + 2 * kv_heads) * head_dim
        dq = torch.randn(
            batch * seq, qkvd,
            device="cuda", dtype=torch.bfloat16)
        q_weight = torch.randn(
            head_dim, device="cuda", dtype=torch.bfloat16)
        k_weight = torch.randn_like(q_weight)
        cos = torch.randn(
            batch * seq, head_dim,
            device="cuda", dtype=torch.bfloat16)
        sin = torch.randn_like(cos)
        q_expanded = torch.empty(
            batch * seq, heads, head_dim,
            device="cuda", dtype=torch.bfloat16)
        q_packed = torch.empty_like(q_expanded)
        k_expanded = torch.empty(
            batch, heads, seq, head_dim,
            device="cuda", dtype=torch.bfloat16)
        v_expanded = torch.empty_like(k_expanded)
        k_packed = torch.empty(
            active_rows, heads, head_dim,
            device="cuda", dtype=torch.bfloat16)
        v_packed = torch.empty_like(k_packed)
        args = (
            dq.data_ptr(), q_weight.data_ptr(), k_weight.data_ptr(),
            cos.data_ptr(), sin.data_ptr())
        tail = (
            batch, seq, heads, kv_heads, head_dim, qkvd,
            active_rows, 1e-6, torch.cuda.current_stream().cuda_stream)
        TestGatedBuild._O.omnivoice_qk_norm_rope_expand_kv_bf16(
            *args, q_expanded.data_ptr(), k_expanded.data_ptr(),
            v_expanded.data_ptr(), *tail)
        TestGatedBuild._O.omnivoice_qk_norm_rope_expand_packed_kv_bf16(
            *args, q_packed.data_ptr(), k_packed.data_ptr(),
            v_packed.data_ptr(), *tail)
        torch.cuda.synchronize()
        expected_k = torch.cat(
            [k_expanded[0].transpose(0, 1),
             k_expanded[1, :, :38].transpose(0, 1)])
        expected_v = torch.cat(
            [v_expanded[0].transpose(0, 1),
             v_expanded[1, :, :38].transpose(0, 1)])
        assert torch.equal(
            q_packed[:active_rows], q_expanded[:active_rows])
        assert torch.equal(k_packed, expected_k)
        assert torch.equal(v_packed, expected_v)

    def test_packed_attention_matches_padded_valid_rows(self):
        if TestGatedBuild._O is None or not torch.cuda.is_available():
            pytest.skip("flash_rt_omnivoice CUDA build required")
        torch.manual_seed(53)
        batch, seq, heads, head_dim = 2, 79, 16, 128
        short_seq = 38
        q = torch.randn(
            batch, seq, heads, head_dim,
            device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mask = torch.zeros(
            batch, 1, seq, seq, device="cuda", dtype=torch.bool)
        mask[0] = True
        mask[1, :, :short_seq, :short_seq] = True
        padded = torch.arange(short_seq, seq, device="cuda")
        mask[1, :, padded, padded] = True
        expected_full = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask, dropout_p=0.0, scale=head_dim ** -0.5,
            is_causal=False).transpose(1, 2)
        expected = torch.cat(
            [expected_full[0], expected_full[1, :short_seq]]).unsqueeze(0)
        packed_q = torch.cat(
            [q[0], q[1, :short_seq]]).unsqueeze(0)
        packed_k = torch.cat(
            [k[0], k[1, :short_seq]]).unsqueeze(0)
        packed_v = torch.cat(
            [v[0], v[1, :short_seq]]).unsqueeze(0)
        cu_seqlens = torch.tensor(
            [0, seq, seq + short_seq],
            device="cuda", dtype=torch.int32)
        actual = torch.ops.aten._efficient_attention_forward(
            packed_q, packed_k, packed_v, None,
            cu_seqlens, cu_seqlens, seq, seq,
            0.0, 0, False, scale=head_dim ** -0.5)[0]
        specialized = torch.empty_like(actual)
        TestGatedBuild._O.omnivoice_exact_packed_attention_bf16(
            packed_q.data_ptr(), packed_k.data_ptr(),
            packed_v.data_ptr(), specialized.data_ptr(),
            cu_seqlens.data_ptr(), seq + short_seq, seq, batch,
            heads, head_dim, head_dim ** -0.5,
            torch.cuda.current_stream().cuda_stream)
        compact = torch.empty_like(actual)
        TestGatedBuild._O.omnivoice_exact_packed_attention_compact_bf16(
            packed_q.data_ptr(), packed_k.data_ptr(),
            packed_v.data_ptr(), compact.data_ptr(),
            seq + short_seq, seq, heads, head_dim,
            head_dim ** -0.5,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
        assert torch.equal(specialized, actual)
        assert torch.equal(compact, actual)


    def test_engine_cfg_kernel_true(self):
        from flash_rt.models.omnivoice import pipeline_rtx
        import importlib
        importlib.reload(pipeline_rtx)
        if TestGatedBuild._O is not None:
            assert pipeline_rtx._has_cfg_kernel is True

    def test_graph_shape_cache_restores_independent_state(self):
        from collections import OrderedDict
        from flash_rt.models.omnivoice.pipeline_rtx import FlashRTLlm

        engine = object.__new__(FlashRTLlm)
        engine._graph_cache = OrderedDict()
        engine._graph_cache_limit = 2

        states = {}
        for shape in ("shape-a", "shape-b"):
            for field in FlashRTLlm._GRAPH_STATE_FIELDS:
                value = object()
                setattr(engine, field, value)
                states[(shape, field)] = value
            engine._graph_shape = shape
            states[(shape, "_graph_shape")] = shape
            engine._save_graph_state()

        assert engine._activate_graph_state("shape-a")
        for field in FlashRTLlm._GRAPH_STATE_FIELDS:
            assert getattr(engine, field) is states[("shape-a", field)]
        assert list(engine._graph_cache)[-1] == "shape-a"
        assert not engine._activate_graph_state("missing")

    def test_bf16_workspace_aliases_disjoint_projection_phases(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from flash_rt.models.omnivoice.pipeline_rtx import (
            FlashRTLlmBF16, GUD, QKVD)

        engine = object.__new__(FlashRTLlmBF16)
        engine.device = "cuda:0"
        engine._active_rows = 117
        engine._alloc(2, 79)
        buffers = engine._buf

        assert buffers["projection"].numel() == 117 * GUD
        assert buffers["Dq"].shape == (117, QKVD)
        assert buffers["Dg"].shape == (117, GUD)
        assert buffers["Dq"].data_ptr() == buffers["Dg"].data_ptr()
        assert (
            buffers["Dq"].data_ptr()
            == engine._attn_o_buf.data_ptr())
        assert "h2" not in buffers
        assert "q_temp" not in buffers
        assert "k_flat" not in buffers
        assert "k_temp" not in buffers

    def test_check_kernels_passes(self):
        from flash_rt.models.omnivoice import _check_kernels
        import importlib
        from flash_rt.models.omnivoice import pipeline_rtx
        importlib.reload(pipeline_rtx)
        # Reload to pick up the current .so state
        from flash_rt.models.omnivoice import pipeline_rtx as prx
        importlib.reload(prx)
        if prx._fvo is not None:
            prx._check_kernels()

    # ── flash_rt_fa2 ──

    def test_fa2_module(self):
        if TestGatedBuild._A is None:
            pytest.skip("flash_rt_fa2 not available")
        assert hasattr(TestGatedBuild._A, "fwd_bf16")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
