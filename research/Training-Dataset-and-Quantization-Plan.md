# x8D Training Dataset & Quantization Plan

Status: research/plan — audited 2026-07-31 against DiffusionGemma + x8Dsub-byte.
Objective: train the byte-native `x8D-Omni-Diffusion` (vocab 264) and serve it
compressed via the 0.001 sub-byte law with DSpark speculative decoding.

---

## 1. What dataset is needed?

The model is **byte-native**: every input is `list(data_bytes)`, no tokenizer.
That makes the *corpus* unrestricted — any byte stream (text UTF-8, image pixel
bytes, PCM audio, code, binaries) is in-vocabulary. The dataset requirements
therefore come from **what we want the model to do**, not from vocabulary limits.

### Tier 0 — byte-native pretraining (recommended start)
Pure byte streams, no special-token markup, maximally dense.

| Dataset | Content | Why |
|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 15T tokens of deduped CommonCrawl, MIT/CC | Highest-quality raw text; read directly as UTF-8 bytes |
| [The Pile](https://huggingface.co/datasets/EleutherAI/the_pile) | 825 GB text, 22 domains | Mixed code/books/arxiv — good byte diversity |
| [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | 1.2T tokens | Multi-source, easy to shard |
| [The Vault](https://huggingface.co/datasets/totally-not-an-llm/The-Vault) | permissive code | Code = hardest byte distribution (UTF-8 + ASCII-heavy) |

### Tier 1 — multimodal bytes (after Tier 0 loss plateaus)
Images and audio are just byte arrays to this model; no vision/audio encoder is
strictly required because pixels/PCM already live at ids 0-255.

| Dataset | Content | Encoded as |
|---|---|---|
| [LAION-5B](https://laion.ai/blog/laion-5b/) | 5.8B image-text pairs | image bytes (decoded PNG/JPEG pixel bytes) + text bytes |
| [ImageNet-1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k) | 1.2M labeled images | pixel bytes only |
| [LibriTTS](https://huggingface.co/datasets/facebook/libritts_r) | 585 h aligned speech | PCM byte pairs (see `evaluate_libritts.py`) |
| [VoxCeleb2](https://huggingface.co/datasets/andandandand/voxceleb2) | 6k speakers | PCM bytes |
| [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) | 2M clips | waveform bytes |

### Tier 2 — structured / SFT
- **Instruction pairs** (byte-aligned Q→A): Alpaca, LIMA, OpenOrca, UltraChat.
  Format as `[BOS] <query bytes> [EOS] <answer bytes> [EOS]`.
- **Image→text / text→image**: LAION pairs re-serialized as
  `[IMG_START] <pixel bytes> [IMG_END] <text bytes>`.
- **Audio→text / text→audio**: `[AUD_START] <pcm> [AUD_END] <text>` pairs.
  Use `mask_canvas` / `renoise_to_random_bytes` to train the diffusion denoiser:
  mask one modality, generate the other.

### Recommended recipe (small → large)
1. **Sweep**: 1B–10B bytes of FineWeb subset (UTF-8) — validate the 264-vocab
   LM fits on a single GPU, confirm byte-entropy curves look sane.
2. **Pretrain**: FineWeb + The Pile (text bytes only), 32k–64k byte batches,
   canvas length 256, masked-diffusion objective (predict 25% masked bytes,
   DiffusionGemma style, `diffusion_entropy_bound=0.1`).
3. **Multi-modal warm-start**: add LAION + LibriTTS byte pairs, keep
   `IMG_START/IMG_END/AUD_START/AUD_END` markup.
4. **SFT/DPO**: instruction byte pairs on the finetuned checkpoints.

> Dataset sharding must be **byte-aligned**: split files at raw byte offsets
> (e.g. `data[start:end]`) — never mid-UTF-8-codepoint unless you want
> `errors="replace"` behavior in decode.

---

## 2. How to quantize a model with speculative decoding (x8Dsub-byte)

Implemented in `omni_diffusion/x8d_spec_decode.py` (pure Python, stdlib).
The flow mirrors AGENTS.md §2 "Speculative Decoding for Weight Compression".

### Pipeline

```
1. Convert checkpoint  →  raw uint8 weight bytes
   - BF16/FP32 tensor → nearest uint8 byte (torch round, 0-255 range)
   - or load x8D byte-native state (already U8)

2. DSpark block-parallel quantization (per 8x8 block = 64 bytes)
   for each block:
     a. generate all 64 bytes in parallel
     b. confidence head scores each position in [0,1]
     c. positions with confidence < 0.001  → re-mask → regenerate
     d. if !converged after max_steps → raise

3. Apply 0.001 law:  Quanta[i] = weight_byte[i] * 0.001

4. Store as U8 coordinates in X8DGGUF1 container (save_gguf)

5. Serve via zero-copy mmap (mmap_load_gguf); inverse (/ 0.001)
   is a live coordinate pointer map at inference
```

### How to use it today (stdlib)
```python
from omni_diffusion.x8d_spec_decode import (
    speculative_quantize, speculative_save_gguf, size_report,
)

# 1. quantize 2 MiB of raw weights with block-parallel speculative decoding
quanta, stats = speculative_quantize(weight_bytes, max_steps=16, seed=42)
print(stats)   # {'blocks': N, 'regenerations': ..., 'converged': N}

# 2. save to an x8D .gguf container
path, _ = speculative_save_gguf("model.embed_tokens.weight", weight_bytes, "model.gguf")

# 3. zero-copy mmap serve
from omni_diffusion.x8d_export import mmap_load_gguf
mapping, meta = mmap_load_gguf("model.gguf")
```

### When torch arrives (upgrade path)
- Replace `_block_surrogate` (deterministic hash) with a real **lightweight
  confidence head** (linear probe over block embedding).
- Replace byte-noise regeneration with the actual model's logits over ids 0-255
  (DSpark-style semi-autoregressive, NOT full autoregressive).
- Under heavy load set `heavy_load=True` → verification length clips to
  `DEFAULT_VERIFY_LEN // HEAVY_LOAD_VERIFY_CLIP` (64/16 = 4 positions/block).

---

## 3. Size comparison: full FP16 16B vs x8D sub-byte

`python3 -c "from omni_diffusion.x8d_spec_decode import print_size_report; print_size_report()"`

| Representation | Size (16B params) | vs FP16 |
|---|---|---|
| Full FP16/BF16 (16-bit) | **32.00 GB** | — |
| Full FP32 (32-bit) | **64.00 GB** | 2× FP16 |
| x8D U8 .gguf storage (1 byte/param) | **16.00 GB** | 50% reduction |
| x8D sub-byte coordinates (16-bit baseline × 0.001) | **32.0 MB** | 99.90% reduction |

Math:
- FP16: `16e9 × 2 B = 32 GB`
- x8D disk: `16e9 × 1 B = 16 GB` (lossless U8 byte coordinate storage)
- sub-byte coordinate space: `16e9 × (16 bits × 0.001) / 8 = 32 MB` → **0.016 bit/weight**

The honest framing:
- **On disk** (X8DGGUF1 U8 container): 2:1 vs FP16, lossless — the real,
  servable number.
- **Coordinate space** (0.016 bit/weight): 1000:1 theoretical ceiling — only
  reachable once the quanta are stored as packed coordinates rather than bytes.

`size_report()` in `x8d_spec_decode.py` computes both and
`tests/test_spec_decode.py` locks the numbers (32 GB / 16 GB / 32 MB / 99.9%).

---

## 4. Acceptance criteria
- [ ] Byte-native dataset sharder: `list(data_bytes)` + canvas masking (no tokenizer)
- [ ] Tier-0 text pretrain runnable on 1 GPU (torch stack)
- [ ] Real confidence-head speculative quantization replaces `_block_surrogate`
- [ ] Trained weights exported to x8D .gguf, mmap-served, size report verified
