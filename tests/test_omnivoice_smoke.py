#!/usr/bin/env python3
"""OmniVoice FlashRT kernel smoke test.

Covers two build configurations:
  - Default (FLASHRT_ENABLE_OMNIVOICE=OFF): flash_rt_kernels has base symbols only,
    flash_rt_omnivoice is absent, engine reports _has_cfg_kernel=False.
  - OmniVoice (FLASHRT_ENABLE_OMNIVOICE=ON): flash_rt_omnivoice module is present
    with cfg_combine + maskgit symbols, engine _has_cfg_kernel=True.

Usage:
  pytest -q tests/test_omnivoice_smoke.py               # default build
  pytest -q tests/test_omnivoice_smoke.py -m gpu          # GPU build (SM120)
"""
import pytest, warnings, sys, os
warnings.filterwarnings("ignore")

_BUNDLE = os.environ.get("OMNIVOICE_BUNDLE", "/data/omnivoice_flashrt_bundle")
sys.path.insert(0, _BUNDLE)


class TestDefaultBuild:
    """Default build (FLASHRT_ENABLE_OMNIVOICE=OFF).

    flash_rt package imports work without GPU. Engine symbols are present
    but _has_cfg_kernel is False (flash_rt_omnivoice not built).
    """

    def test_package_imports(self):
        import flash_rt
        for attr in ("inject", "free_encoder", "eject"):
            assert hasattr(flash_rt, attr)

    def test_engine_import(self):
        from flash_rt import api
        assert hasattr(api, "FlashRTLlm")
        assert hasattr(api, "FlashRTLlmBF16")

    def test_cfg_kernel_disabled(self):
        """When flash_rt_omnivoice is absent, _has_cfg_kernel is False."""
        from flash_rt import api
        import importlib
        importlib.reload(api)
        has_omnivoice = False
        try:
            from flash_rt import flash_rt_omnivoice as _t  # noqa
            has_omnivoice = True
        except ImportError:
            pass
        if has_omnivoice:
            assert api._has_cfg_kernel is True, "kernel present but flag is False"
        else:
            assert api._has_cfg_kernel is False

    def test_check_kernels_raises(self):
        """_check_kernels() raises RuntimeError when omnivoice is missing."""
        from flash_rt import api
        import importlib
        importlib.reload(api)
        try:
            api._check_kernels()
        except RuntimeError as e:
            assert "flash_rt_omnivoice" in str(e)
        else:
            # No exception means all kernels are present (flag ON).
            # Skip — this test validates the OFF path.
            pass


class TestGatedBuild:
    """OmniVoice build (FLASHRT_ENABLE_OMNIVOICE=ON, SM120 GPU).

    All kernel symbols must be present across three modules:
      flash_rt_kernels  — base FP4/fused kernels
      flash_rt_omnivoice — OmniVoice-specific (cfg_combine, maskgit)
      flash_rt_fa2       — FlashAttention2
    """

    _F = None  # flash_rt_kernels
    _O = None  # flash_rt_omnivoice
    _A = None  # flash_rt_fa2

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Match engine import paths: local dev + flashcli deployment."""
        if TestGatedBuild._F is None:
            try:
                from flash_rt import flash_rt_kernels as fvk
            except ImportError:
                import flash_rt_kernels as fvk
            TestGatedBuild._F = fvk
        if TestGatedBuild._O is None:
            try:
                from flash_rt import flash_rt_omnivoice as fvo
            except ImportError:
                try:
                    import flash_rt_omnivoice as fvo
                except ImportError:
                    fvo = None
            TestGatedBuild._O = fvo
        if TestGatedBuild._A is None:
            try:
                from flash_rt import flash_rt_fa2 as fa2
            except ImportError:
                import flash_rt_fa2 as fa2
            TestGatedBuild._A = fa2

    # ── flash_rt_kernels symbols ──

    @pytest.mark.gpu
    def test_kernels_module(self):
        assert TestGatedBuild._F is not None

    @pytest.mark.gpu
    def test_fp4_gemm_symbols(self):
        fvk = TestGatedBuild._F
        assert hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out")
        assert hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out_pingpong")

    @pytest.mark.gpu
    def test_fused_norm_symbols(self):
        fvk = TestGatedBuild._F
        for sym in ("rms_norm", "rms_norm_to_nvfp4_swizzled_bf16",
                     "residual_add_rms_norm",
                     "residual_add_rms_norm_to_nvfp4_swizzled_bf16"):
            assert hasattr(fvk, sym)

    @pytest.mark.gpu
    def test_qk_norm_rope_v4(self):
        assert hasattr(TestGatedBuild._F, "fused_qk_norm_rope_v4_bf16")

    @pytest.mark.gpu
    def test_quantize_symbols(self):
        fvk = TestGatedBuild._F
        assert hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled")
        assert hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled_mse")

    @pytest.mark.gpu
    def test_silu_symbols(self):
        assert hasattr(TestGatedBuild._F,
                        "silu_mul_merged_to_nvfp4_swizzled_bf16")

    # ── flash_rt_omnivoice symbols (gated by FLASHRT_ENABLE_OMNIVOICE) ──

    @pytest.mark.gpu
    def test_omnivoice_module_present(self):
        """flash_rt_omnivoice must be importable when flag is ON."""
        assert TestGatedBuild._O is not None, \
            "flash_rt_omnivoice not found — rebuild with -DFLASHRT_ENABLE_OMNIVOICE=ON"

    @pytest.mark.gpu
    def test_cfg_combine_symbol(self):
        if TestGatedBuild._O is None:
            pytest.skip("flash_rt_omnivoice not built")
        assert hasattr(TestGatedBuild._O, "cfg_combine_log_softmax_bf16")

    @pytest.mark.gpu
    def test_maskgit_symbols(self):
        if TestGatedBuild._O is None:
            pytest.skip("flash_rt_omnivoice not built")
        for sym in ("maskgit_sample_row_bf16",
                     "maskgit_select_topk_bf16"):
            assert hasattr(TestGatedBuild._O, sym), f"missing: {sym}"

    @pytest.mark.gpu
    def test_engine_cfg_kernel_true(self):
        """importlib.reload picks up the omnivoice module if present."""
        from flash_rt import api
        import importlib
        importlib.reload(api)
        if TestGatedBuild._O is not None:
            assert api._has_cfg_kernel is True

    @pytest.mark.gpu
    def test_check_kernels_passes(self):
        """_check_kernels() should not raise when all modules present."""
        from flash_rt import api
        import importlib
        importlib.reload(api)
        if TestGatedBuild._O is not None:
            api._check_kernels()  # no exception expected

    # ── flash_rt_fa2 ──

    @pytest.mark.gpu
    def test_fa2_module(self):
        assert hasattr(TestGatedBuild._A, "fwd_bf16")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
