# OmniVoice FlashRT — Performance Specifications

RTX 5060 Ti (SM120, 448 GB/s, 36 SMs), CUDA 13.0, PyTorch 2.12.
Model: OmniVoice TTS (Qwen3-1.5B, D=1024, L=28, FFN=3072, GQA 16Q/8KV).

## Build Configuration

```sh
cmake -DFLASHRT_ENABLE_OMNIVOICE=ON -DGPU_ARCH=120 -DENABLE_NVFP4=ON ..
make -j$(nproc) flash_rt_kernels flash_rt_fa2 flash_rt_omnivoice
```

## Kernel Modules

| Module | Contents | Gate |
|--------|----------|------|
| `flash_rt_kernels` | FP4 GEMM, RMS norm, quantize, SiLU | Always built |
| `flash_rt_omnivoice` | Exact fused RMSNorm, shape-tuned BF16 GEMM, compact 32x128 packed attention, QK/RoPE/KV layout, SiLU, CFG and MaskGIT selection | `FLASHRT_ENABLE_OMNIVOICE=ON` |
| `flash_rt_fa2` | FlashAttention2 BF16 forward | Always built |

## Exact BF16 Profile

Controlled voice-clone case:

- Reference audio: `split_35.wav`
- Reference text: `Barely get by`
- Target text: `勉强糊口`
- 32 MaskGIT steps, guidance scale 2.0, seed 42
- PyTorch and FlashRT use the same checkpoint and generation settings

| Metric | Value |
|--------|-------|
| **Warm latency (median)** | 130.09 ms |
| **PyTorch warm latency (median)** | 399.83 ms |
| **RTF** | 0.1041 |
| **Speedup vs PyTorch** | 3.07x |
| **Token match** | 304 / 304 (100%) |
| **Waveform cosine** | 1.000000 |
| **Mel cosine** | 1.000000 |
| **Waveform max abs error** | 0 |

The controlled output is 1.25 seconds long. The table reports medians from
five PyTorch runs and twelve current FlashRT runs after one complete warm-up
for each path. FlashRT graph capture is therefore excluded. Measurements are
workload- and GPU-dependent.

For a single CFG request, the conditional sequence and the valid
unconditional prefix occupy 117 contiguous rows out of the padded 158-row
batch. The exact profile skips the 41 unused rows in QKV, Gate/Up, Down, and
RMSNorm. Attention packs those rows as variable-length 79- and 38-token
sequences. Its exact CUTLASS kernel retains PyTorch's 128-key tile and
reduction order but uses a 32-query tile instead of the generic 64-query
tile, avoiding most of the wasted work on the 38-token sequence. Across 30
random packed shapes up to 257+131 tokens, its BF16 output was bit-identical
to PyTorch's memory-efficient attention. On the controlled shape the
attention kernel fell from about 16.15 us to 11.85 us, reducing the complete
896-attention workload from 14.82 ms to 10.62 ms. For the O projection, it
computes only the 117 valid rows but fixes cuBLASLt to the algorithm selected
for the original 158-row shape. This preserves PyTorch's BF16 reduction order
while removing padded-tail work.
The MaskGIT loop also fuses its eight codebook embedding gathers, FP32 sum,
and two CFG input writes. Argmax, confidence max, and layer penalty share one
reduction; position Gumbel scoring, score masking, radix Top-K selection, and
token update share another kernel for the single-request path. Its FP32 CFG
subtract/scale/add sequence is fused with the same three rounding points as
PyTorch, the layer penalty is prepared once per request, and graph replay
copies only the valid input rows while reusing the constant packed-attention
layout. The audio-head GEMM and a single fused CFG/selection kernel are
captured at the tail of the model CUDA Graph. The fused kernel directly reads
the packed audio-head layout, reproduces all three FP32 PyTorch log-softmax
warp reductions and CFG rounding points, then performs mask-aware argmax and
confidence scoring without materializing permuted logits or log-probability
tensors. A fused graph prologue restores the constant prompt rows and
regenerates both CFG target embedding spans from the current tokens. The graph
epilogue draws the same PyTorch CUDA uniforms, reads K from a device schedule,
performs radix Top-K, updates tokens in place, and advances a device step
counter. Capture warm-up restores the pre-capture RNG state, so a seeded first
replay consumes exactly the random sequence used by the eager path. The
single-request MaskGIT loop therefore issues 32 complete-step graph launches,
with direct replay after the first validated step and the same shape-keyed
recapture behavior.

In a follow-up clean-process check on the controlled 1.25-second case, the
legacy graph boundary measured 138.24 ms median and the complete-step graph
measured 137.87 ms (RTF 0.11029), a 0.37 ms reduction. The generated FP32
waveform was bit-identical (30,000 / 30,000 samples, max absolute error 0).
This smaller sequential check validates the incremental optimization; the
headline table reflects the later attention-specialized engine.

After the exact 32-query attention specialization, a same-process A/B check
measured the generic PyTorch attention path at 138.44 ms median and the
specialized path at 133.60 ms. All compared waveforms were bit-identical.
The independent headline run measured 133.57 ms median.

The audio tokenizer decoder removes its redundant zero-initialized first
codebook add, then replays fixed token shapes from a CUDA Graph. Decoder graphs
are cached for at most four hot shapes; additional variable lengths use the
unchanged eager decoder. In an interleaved comparison this reduced median
end-to-end latency from 138.61 ms to 137.89 ms without changing a waveform
sample.

The exact LLM path keeps the two most recently used CUDA Graph shapes in
independent graph pools. This avoids recapture when a serving workload
alternates between two prompt/text geometries. In a same-process alternating
English/Chinese check, revisiting a hot shape fell from 150.17/149.64 ms with
the legacy one-shape behavior to 134.38/134.25 ms, a roughly 10.5% reduction.
All revisited waveforms were bit-identical to their first result. The second
hot graph added about 45.5 MiB of allocated GPU memory. A third shape evicts
the least-recently-used graph and recaptures safely; MaskGIT token, schedule,
and step buffers are cached separately so returning to a resident graph keeps
its captured pointers stable.

The BF16 allocator now removes FP4-only ping-pong buffers and aliases the
non-overlapping QKV, attention-output, and Gate/Up phases onto one contiguous
workspace. Projection, activation, Q, K, and V workspaces are sized to the
117 valid rows rather than the 158-row padded batch. In order-reversed
clean-process A/B runs, the current allocator measured 133.31 ms median versus
134.68 ms for the legacy allocator and used 1432.98 MiB versus 1437.94 MiB
with one hot graph. Both paths produced the same FP32 waveform SHA-256
(`542333e429c3e3362ced26418373aa7d554efc844cad6f3cc33071ba6b2fee0a`).

The exact fused audio-head tail directly promotes BF16 logits in registers
and reduces its six-operation sequence (BF16-to-FP32 promotion, three
log-softmax launches, CFG combine, and argmax/score) from 30.15 us to 15.88 us
on the controlled 304x1025 shape. The initial FP32-input fused version measured
132.91 ms end to end; direct BF16 consumption reduced that to 132.72 ms while
preserving the waveform SHA-256 above. Relative to the original unfused tail,
one-hot-graph allocated memory remains reduced from 1432.98 MiB to
1429.41 MiB.

The exact BF16 MLP path now uses a CUTLASS dual GEMM for profitable row
shapes. Gate and Up share each activation tile, and an epilogue reproduces
PyTorch's two BF16 rounding points around SiLU before writing only the
3072-wide activated result. On the controlled 117-row shape, Gate/Up plus
SwiGLU decreased from 43.15 us to 40.68 us per layer. Clean-process A/B
medians decreased from 132.86 ms to 130.86 ms end to end, while graph
allocation decreased from 1429.41 MiB to 1428.57 MiB. Eight cases spanning
two reference audios, Chinese and English targets, multiple lengths, and
multiple seeds produced bit-identical waveforms (cosine 1.000000 and maximum
absolute error 0 in every case).

The latest exact pass compacts the two-sequence attention launch only for the
three-block/two-block CFG bucket where it wins, eliminating the unused query
CTA without changing the 32x128 tile or its arithmetic. At the controlled
79+38 shape the attention microbenchmark decreased from 11.52 us to
11.19 us. Other buckets retain the generic packed launch when compaction does
not help. A cuBLASLt scan also found heuristic rank 4 to be bit-identical to
PyTorch rank 0 for the 112..128-row QKV bucket while reducing the controlled
117-row projection from 27.33 us to 26.74 us. The current clean-process
short-case median is 130.09 ms (RTF 0.1041), with the same exact waveform
SHA-256 shown above.

MaskGIT's fused radix Top-K and complete-step CUDA Graph now cover up to
1024 selection elements instead of 512, corresponding to 128 target tokens
with eight codebooks. Ordinary, Gumbel, and device-scheduled selection all
share the extended path. With an explicit duration, a 3.09-second output
(600 elements) measured 209.81 ms median, RTF 0.0679; a 4.63-second output
(896 elements) measured 265.28 ms, RTF 0.0573. Both outputs were
sample-for-sample identical to the old fallback. This extension improves coverage and
keeps longer requests on the captured path; Top-K is too small a share of
runtime to materially change the short-case latency.

Audio postprocessing now replaces pydub's repeated 500 ms silence-window
scan with an int64 prefix-square-sum pass and reuses cached fade ramps. It
retains pydub's int16 conversion, truncated RMS threshold, millisecond
slicing, volume-operation order, fades, and padding. Postprocessing decreased
from about 1.32 ms to 0.62 ms on the controlled output. Randomized mono and
stereo lengths, all-silence cases, and the end-to-end waveform were
sample-for-sample identical to the upstream implementation.

The user-direction check (reference text `勉强糊口`, target text
`Barely get by`) produces 0.99 seconds of audio in 133.18 ms
(RTF 0.1345). Its waveform is also bit-identical to PyTorch
(cosine 1.000000, max absolute error 0).

## Precision Notes

- The default `cfg_ratio=1.0` uses exact BF16 CFG for all MaskGIT steps.
- Exactness requires the upstream FP32 CFG operation order, the full attention
  mask, PyTorch-compatible RMS reduction, and BF16 rounding at QK/RoPE
  boundaries.
- Lower `cfg_ratio` values use FP4 without CFG for the remaining steps. This is
  an explicit speed/quality tradeoff and is not expected to match PyTorch
  tokens or satisfy the exact profile above.

### Mixed-precision feasibility check

An operator-level PTQ experiment kept CFG, QKV, attention, O projection,
normalization, and RoPE in BF16 while changing only the Gate/Up and Down
projections to dynamically-scaled FP8. On the controlled case above:

| Profile | Warm latency | Waveform cosine | Log-Mel cosine |
|---------|-------------:|----------------:|---------------:|
| Exact BF16 | 139.2 ms | 1.000000 | 1.000000 |
| All MLP layers FP8 | 125.3 ms | -0.0233 | 0.8347 |

The individual FP8 GEMMs were 1.7–2.3x faster and had cosine around 0.9993,
but MaskGIT amplified the small logit-order changes into different discrete
tokens. Quantizing only one MLP layer still changed the final token path.
Consequently, PTQ mixed precision is not enabled as a production profile:
its roughly 10% end-to-end speed ceiling does not satisfy the
`cosine >= 0.999` quality requirement. Reaching that target with low
precision requires quantization-aware training or distillation evaluated on
the final token and audio outputs, rather than a hidden-state cosine gate.

Moving low precision only to the end of MaskGIT does not provide a safe
window either:

| BF16 CFG / FP4 noCFG steps | Warm latency | Waveform cosine | Log-Mel cosine |
|----------------------------|-------------:|----------------:|---------------:|
| 32 / 0 | 139.2 ms | 1.0000 | 1.0000 |
| 31 / 1 | 137.7 ms | 0.7533 | 0.9830 |
| 30 / 2 | 136.1 ms | 0.7463 | 0.9807 |
| 28 / 4 | 132.2 ms | 0.5183 | 0.9579 |
| 30 / 1 + final BF16 bookend | 137.9 ms | 0.8577 | 0.9820 |

Real-activation error attribution shows that FP8 weight and activation
quantization contribute approximately equally: their isolated projection
cosines average 0.99982, while quantizing both averages 0.99965. Block-128
scaling and output-aware scale clipping improve individual projections only
marginally and do not remove the token-order instability. A useful next
low-precision profile therefore needs trained weight/activation adaptation;
additional PTQ scale search is not expected to cross the end-to-end bar.

### RTF 0.09 boundary on the controlled short case

An Nsight trace of the current 1.25-second, 32-step case reports about
122.4 ms of GPU kernel work. BF16 linear projections account for 99.4 ms
(81%), while all exact packed attention accounts for 10.1 ms. RTF 0.09
allows only 112.5 ms end to end, so the target is below the current GPU work
before Python, synchronization, and audio postprocessing are included.

A diagnostic cuBLASLt profile used faster reduction algorithms for the O and
Down projections and reached 123.42 ms (RTF 0.0987), but it changed MaskGIT's
discrete token path: the output shortened from 30,000 to 18,240 samples and
waveform cosine fell to -0.00083. That profile is deliberately not enabled.
On this GPU and checkpoint, the current implementation therefore reaches
RTF 0.09 for outputs around three seconds or longer, but not for the
controlled 1.25-second case while maintaining `cosine >= 0.999`. Closing the
remaining short-case gap requires a quality-trained low-precision model or
materially higher memory bandwidth, not another launch-only optimization.

## VRAM Breakdown

The controlled case reports PyTorch allocator memory after warm-up. The
normal PyTorch lifecycle keeps the voice-clone encoder available, while the
recommended FlashRT lifecycle calls `free_encoder()` after creating the
prompt:

| Runtime state | Warm allocated | Peak allocated |
|---------------|---------------:|---------------:|
| PyTorch, encoder retained | 1973.94 MiB | 2001.19 MiB |
| PyTorch, encoder released | 1359 MiB | 1386 MiB |
| FlashRT exact, encoder released | 1401.91 MiB | 1427.36 MiB |

FlashRT therefore lowers overall warm/peak allocated memory by about 29%
relative to the normal PyTorch lifecycle. On a strictly like-for-like basis
with the encoder released in both runtimes, the exact FlashRT graph costs
about 42 MiB more warm memory and 41 MiB more peak memory (about 3%).

| Component | Size |
|-----------|------|
| embed_tokens | 296 MB |
| WL_bf16 (weights, transient) | 840 MB |
| WL_fp4 (packed weights) | 230 MB (hybrid profile only) |
| Decoder (final proj) | 151 MB |
| Buffers + CUDA Graph | 500 MB |
| **Hybrid-profile total** | **~2.0 GB** |

Memory optimizations:
- The default exact profile does not construct FP4 weights or its warmup graph
- FP4 instance releases WL_bf16 after quantization (-840 MB)
- Original layer weights zeroed after calibration (-840 MB)
- Encoder weights freed via `free_encoder()` (-612 MB)

## Engine Injection

```python
from flash_rt.models.omnivoice import inject, free_encoder

model = OmniVoice.from_pretrained(...)
inject(model)  # cfg_ratio=1.0: exact BF16 quality profile
free_encoder(model)

# model._generate_iterative now uses FlashRT accelerated path
```

The engine raises `RuntimeError` before model loading if required kernel
symbols are missing, with instructions to rebuild with
`-DFLASHRT_ENABLE_OMNIVOICE=ON`.

## Regression Tests

```sh
pytest -q tests/test_omnivoice_smoke.py
```
