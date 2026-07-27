"""FlashRT — OmniVoice TTS model pipeline.

The default exact-BF16 profile preserves the PyTorch token and waveform path.
Lower CFG ratios opt into a faster BF16/FP4 quality tradeoff.

Per the unified API contract:
    inject()       — patch OmniVoice model for FlashRT acceleration
    free_encoder() — release encoder weights (~600 MB VRAM saved)
    eject()        — restore original forward and generate methods

See docs/PERFORMANCE_OMNIVOICE.md for performance specifications.
"""

from flash_rt.models.omnivoice.pipeline_rtx import (
    FlashRTLlm,
    FlashRTLlmBF16,
    inject,
    free_encoder,
    eject,
    _check_kernels,
)

__all__ = [
    "FlashRTLlm",
    "FlashRTLlmBF16",
    "inject",
    "free_encoder",
    "eject",
    "_check_kernels",
]
