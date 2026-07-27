# serving/omnivoice_agent

TTS inference server for OmniVoice on FlashRT. The default profile uses the
exact BF16 engine for every CFG step; lower `--cfg-ratio` values explicitly
opt into the faster BF16 + FP4 quality tradeoff.

This directory is the policy layer above the FlashRT execution contract.
It owns the FastAPI HTTP surface, WAV framing, and request serialisation.
The model is driven through `flash_rt.models.omnivoice.inject()` which
patches OmniVoice for FlashRT acceleration.

## What runs where

```
serving/omnivoice_agent/server.py          policy: HTTP, audio framing, serialisation
  └─ flash_rt.models.omnivoice.inject()    accelerated MaskGIT loop
       ├─ FlashRTLlmBF16                    exact BF16 CFG (default: all steps)
       └─ FlashRTLlm                        optional FP4 noCFG tail
```

Single stream: TTS requests are serialised (one model instance). The engine
uses CUDA Graph and fused FlashRT kernels throughout.

## Quickstart

```bash
export OMNIVOICE_CHECKPOINT=/path/to/OmniVoice
pip install fastapi uvicorn soundfile
python -m serving.omnivoice_agent.server \
    --checkpoint "$OMNIVOICE_CHECKPOINT" --host 127.0.0.1 --port 8000
```

Clone mode (with reference audio):

```bash
curl -s http://127.0.0.1:8000/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "text": "Hello from FlashRT.",
      "mode": "clone",
      "reference_audio": "/path/to/ref.wav",
      "num_steps": 32,
      "guidance_scale": 2.0,
      "seed": 42
    }
  }' | python3 -c "import sys,json,base64; d=json.load(sys.stdin)['output']
s=d.split(',')[1]; open('out.wav','wb').write(base64.b64decode(s))"
```

Design mode (text-described voice):

```bash
curl -s http://127.0.0.1:8000/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "text": "Welcome to FlashRT.",
      "mode": "design",
      "instruct": "A calm male voice with clear pronunciation.",
      "num_steps": 32,
      "guidance_scale": 2.0
    }
  }'
```

`GET /health` reports server status.

## Measured exact profile (RTX 5060 Ti, SM120, ns=32, gs=2.0)

- Warm latency: 130.09 ms | Speedup: 3.07x | RTF: 0.1041
- Token match: 100%
- Waveform and Mel cosine: 1.000000 vs PyTorch BF16 reference

The former `cfg_ratio=0.05` hybrid measured about 100 ms and Mel cosine
0.9961. It remains available as an explicit speed/quality option, but is not
the server default because it does not meet a `cosine >= 0.999` contract.

See `docs/PERFORMANCE_OMNIVOICE.md` for full performance specifications.

## Notes

- `FLASHRT_ENABLE_OMNIVOICE=ON` must be set at cmake build time to compile
  the `flash_rt_omnivoice` module (`omnivoice_cfg_logsoftmax_bf16` and
  `omnivoice_qk_norm_rope_bf16`). Without it, OmniVoice initialization fails
  fast with a clear `RuntimeError`.
- The output is 24 kHz mono PCM16, returned as `data:audio/wav;base64`.
- `pip install fastapi uvicorn` is a server-only dependency; the engine and
  kernels do not require it.
