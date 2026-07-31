# Omni-Modality Stack — Model Matrix (issue #11)

Status: researched 2026-07-31 from primary sources (HF model cards, arxiv).

Goal: complete the byte-native modality set for x8D-Omni-Diffusion. Every
modality is a stream of 8-bit bytes at ids 0–255 — the SAME vocabulary the
264-wide byte embedding already speaks. No modality-specific encoder exists;
input bytes (pixel/PCM) are masked on the canvas and the denoiser fills them.

## Model Matrix

| Modality | Model (source) | Params | License | In repo? | Input bytes | Output bytes |
|---|---|---|---|---|---|---|
| ASR | Whisper large-v3 (openai) | 1.55B | Apache-2.0 | no | PCM→log-mel 128-bin | text bytes |
| ASR | SenseVoice (existing) | — | — | yes (modeling_sensevoice.py) | PCM→mel | text bytes |
| TTS | Kokoro-82M (hexgrad) | 82M | Apache-2.0 | no | text bytes + voice | PCM @ 24 kHz |
| Video | LTX-2 19B (Lightricks) | 19B | LTX-2 community | no | text/pixel bytes | pixel + PCM (joint) |
| Video tok | MagViT-v2 (existing) | — | — | yes (models/magvit) | pixel bytes | latent frames |
| LLM | Dream (existing) | — | — | yes (models/dream) | text/pixel/PCM bytes | byte ids 0-255 |

## Per-modality facts (from source)

### Whisper large-v3 — ASR (openai, Apache-2.0)
- Encoder-decoder Transformer; 30 s receptive field; 128 Mel bins (was 80 in
  v1/v2) + a Cantonese token; trained on 1M h weak + 4M h pseudo-labeled audio.
- 99 languages; sizes tiny 39M / base 74M / small 244M / medium 769M / large
  1.55B. HF lists large-v3 as "2B params" (inclusive of decoder lm heads).
- Long-form: sequential (sliding window, more accurate) or chunked (faster).
- torch.compile ≈ 4.5x; Flash-Attention-2 or SDPA (default in torch ≥2.1.1).
- **Byte contract**: input = PCM 16 kHz → STFT → log-mel (float features, NOT
  raw bytes). Decoder emits BPE text ids → UTF-8 bytes at ids 0–255.
- **x8D path**: quantize decoder-only weight tensors to U8×0.001 via
  `x8d_hf.py` (HF→gguf) + `moe_disk.py`-style mmap serving; text output already
  byte-native through the Dream byte embedding.

### Kokoro-82M — TTS (hexgrad, Apache-2.0)
- StyleTTS2 + iSTFTNet vocoder; decoder-only, NO diffusion, no encoder released.
- 82M params; output 24 kHz PCM; G2P via `misaki` (espeak-ng) + IPA phoneme
  labels; voices: 8 en + 54 multi; trained on ~a few hundred hrs permissive
  audio + synthetic (Apache/MIT/public-domain only); ~$1k A100 cost.
- **Byte contract**: input = text bytes + voice id; output = 24 kHz PCM bytes
  (ids 0–255). iSTFTNet is inverse-STFT vocoder → waveform is directly PCM.
- **x8D path**: 82M × 0.001 = 0.008 bit → ~82 KB at 8-bit; trivial single-file
  .gguf; live /0.001 reverse on the one lightweight pass.

### LTX-2 — joint audio-video foundation (Lightricks, LTX-2 community license)
- DiT; generates synchronized video + audio in ONE model. 19B params.
- Checkpoints: `ltx-2-19b-dev`, `-dev-fp8`, `-dev-fp4` (nvfp4), `-distilled`
  (8 steps, CFG=1), `-distilled-lora-384`, x2 spatial + temporal upscalers.
- Two-stage diffusers pipeline: stage 1 40 steps CFG 4.0 (121 frames, 24 fps,
  W/H div by 32, frames div by 8+1), stage 2 distilled LoRA 3 steps; VAE tiling
  avoids OOM; vocoder gives audio sample rate.
- **Byte contract**: input = text/pixel bytes; latent (video + audio) → VAE +
  vocoder → pixel byte streams + PCM bytes. Single model covers text-to-video,
  image-to-video, video-to-audio, audio-to-video, text-to-audio-video.
- **x8D path**: 19B × 0.001 = 0.008 bit → ~19 MB U8; DiT + VAE + vocoder
  tensors via `x8d_hf.py`; experts (if MoE) via `moe_disk.py` mmap.

## Byte-native encode/decode contract (uniform across all modalities)

1. **Input**: any modality reduces to a byte array `list(data_bytes)`.
   - text → UTF-8 bytes
   - image/video → pixel bytes
   - audio → PCM bytes
2. **Markup**: wrap with the 8 specials — `[IMG_START]` 260 / `[IMG_END]` 261,
   `[AUD_START]` 262 / `[AUD_END]` 263 around image/audio spans.
3. **Canvas**: tokens laid on a 256-byte diffusion canvas; `mask_canvas()`
   masks the target span, `renoise_to_random_bytes()` corrupts, denoiser fills.
4. **Output**: decode byte ids 0–255 back to the native format; /0.001-reverse
   quantized weights live at query time; compressed state IS the running state.

## x8D quantize path per model (summary)

| Model | Converter | Serving | Size at 0.008 bit |
|---|---|---|---|
| Whisper large-v3 | `x8d_hf.py` convert_shard_to_gguf | mmap | ~1.6 MB |
| Kokoro-82M | `x8d_hf.py` | mmap | ~82 KB |
| LTX-2 19B | `x8d_hf.py` (+moe_disk if MoE) | mmap + expert fetch | ~19 MB |
| Dream (byte LLM) | `x8d_export.py` | `SubByteModel` | 32 MB @ 16B |

## Integration status
- [x] Research: Whisper large-v3 (HF card), Kokoro-82M (HF card), LTX-2 (HF card)
- [ ] Torch-dependent forward integration (gated by torch install)
- [ ] Quantized .gguf conversion runs (x8d_hf.py ready, needs torch to run model)
- [ ] End-to-end: byte prompt → Dream denoiser → PCM/pixel bytes
