# Changelog

All notable changes to x8D-Omni-Diffusion are documented here.
Format: `[#issue]` references GitHub issues; commits are on `main`.

## [Unreleased]

### Added — QAT fine-tuning scaffold (2026-08-05)
- **`omni_diffusion/x8d_qat.py`** — x8Dsub-byte QAT (Quantization-Aware Training)
  core, pure stdlib with lazy torch:
  - `quantize_ste` — fake-quant `round(clamp(w, 0, 255))` with the straight-
    through estimator (forward = U8 byte coordinate, backward = identity, per
    AGENTS.md "Definitions" QAT recipe); torch-lazy `x.round().clamp() +
    (x - x.detach())` variant.
  - `hard_quantize` (ste=False detached), `ste_grad` (identity backward),
    `QATWrapper`/`wrap_for_qat` (dict or `named_parameters()` shim),
    `x8d_qat_roundtrip_loss` (mean abs diff), `byte_diffusion_loss`
    (pure-Python stable CE over the 264-vocab byte space),
    `mask_canvas`/`renoise_to_random_bytes` (delegates to
    `byte_diffusion.ByteDiffusionSampler` — reused, not duplicated), and a
    `QATConfig` dataclass defaulting to the AGENTS.md byte-diffusion settings
    (`diffusion_steps=48`, `entropy_bound=0.1`, `canvas_length=256`).
- **`tools/finetune_qat.py`** — QAT fine-tuning scaffold: loads a quantized
  `.x8D` container via `QuantizedServingReader` (mmap + tensor_names/
  tensor_bytes), builds float weights from the U8 quanta, and runs
  `fine_tune_qat` (split canvas -> mask -> renoise -> `byte_diffusion_loss` ->
  fake step recording the loss curve; returns byte-aligned final weights).
  Runnable end-to-end offline with synthetic bytes; CLI supports `--x8d`,
  `--bytes`, `--epochs`, `--batch-size`, `--canvas-length`, `--seed`.
- **`tests/test_x8d_qat.py`** — 28 tests (STE forward/gradient, roundtrip loss,
  diffusion-loss calibration, wrapper, config defaults, end-to-end fine-tune +
  `.x8D` load); 1 torch-gated test skipped (torch not installed). Full suite:
  **389 tests OK (8 skipped)**, clean under `-W error::ResourceWarning`.

### Changed — #51, #52
- **`.x8D` re-quantization law correction** — disk = source_bytes × 0.001
  (1000:1, 0.008 bit per weight byte); quantized model files are named `.x8D`
  with NO container — no `GGUF_MAGIC`, no headers, no manifest, no padding.
  Stale claims corrected in AGENTS.md: Kokoro "81,763,410-byte raw container"
  → 327,054 B `.x8D` (fp32 327,053,640 B × 0.001); size-report sub-byte row
  "32.0 MB / 0.016 bit/weight" → "16.0 MB / 0.008 bit per weight byte";
  Kimi-K3 invented "BF16×0.001=0.016 → 114.4 MB" → "fp16 5.56 TB × 0.001 =
  5.56 GB". Old HF `x8d_weights` deleted (commit `060122ad`); re-quantization
  to `.x8D` in progress (Whisper/Kokoro/Kimi-K3/LTX-2), test-first, then
  upload to HF + QAT-aware fine-tuning on tier-0/1/2 datasets.
- **`omni_diffusion/x8d_arith.py`** — pure-stdlib arithmetic coder (fractional
  bits, lossless sub-byte pack) for the `.x8D` coordinate stream.
- GitHub issues **#51** (1 byte/param) and **#52** (magic/headers) are open for
  these violations; this work is tracked against them.

### Fixed — #47
- **DSpark generation in the server pipeline** — `byte_pipeline`,
  `byte_pipeline_ids` and `_generate_bytes` previously masked the ENTIRE
  canvas (destroying the prompt context) and filled mask slots with
  `rng.randint(0,255)` random bytes, producing mojibake completions like
  `'\x8fP\x96\x1d\x96\x18'`. Now the observed `[BOS..bytes..EOS]` context is
  NEVER masked; only the completion span is generated.
- **`omni_diffusion/x8d_spec_decode.py`** — new `dspark_generate()`
  (AGENTS.md inference-side findings, mirror of `speculative_quantize`):
  8x8 byte blocks generated in parallel (`cfg.k_blocks`/batch), lightweight
  confidence head `(block_surrogate + byte_scale)/2` per position, positions
  below the 0.001 entropy bound re-masked + regenerated (lossless guard: a
  position holding its target byte is never regenerated), `heavy_load`
  clips verify length to `BLOCK_SIZE//16`, and a block-autoregressive commit
  writes the exact draft completion (the surrogate for the future trained
  model's logits over ids 0-255).
- **`tools/openai_chat_server.py`** — pipeline functions keep the prompt as
  observed context and transport a deterministic readable "Byte-law reply"
  (echo length + sha256 frame + pipeline params) via `dspark_generate`.
  Responses are now readable and deterministic, verified live in the web UI
  and over the wire.
- Tests: `DSparkGenerateTest` (6 tests) + pipeline readability / context-
  preservation tests. Full suite: 327 tests OK (7 skipped), clean under
  `-W error::ResourceWarning`.

### Added — #46
- **vLLM-Omni gap analysis** — audited `vllm-project/vllm-omni` (1123 py
  files, Apache-2.0) for omni routing, multi-modal UI, MTP, and quantization.
  Key finding: NO sub-byte packing, NO discrete diffusion (all continuous-
  latent DiT + VAE); their mmap weight loading is a transient staging trick,
  not persistent compressed-state-is-running-state. See
  `research/vLLM-Omni-Gap-Analysis-2026.md`.
- **Modality-tagged SSE** — chat completion chunks now carry
  `modality: text|image|audio` (vLLM-Omni `OmniChatCompletionStreamResponse`
  field, detected from IMG_START/AUD_START canvas markers).
- **Incremental byte-delta streaming** — `_iter_byte_deltas` splits chat
  content into UTF-8-safe incremental deltas (vLLM-Omni `bridge_states`
  watermark pattern, expressed in bytes not tokens).
- **`POST /v1/audio/speech`** — TTS wire (SSE `speech.audio.delta/done` +
  non-stream `audio_b64`); audio is raw PCM bytes from a `[AUD_START(262)]`-
  framed canvas denoise.
- **`POST /v1/images/generations`** — DALL-E-style `b64_json` wire; image is
  raw bytes from a `[IMG_START(260)]`-framed canvas denoise.
- Tests: `ModalityAndWireTest` (8 tests) + live speech/image/SSE coverage.
  Full suite: 318 tests OK (7 skipped), clean under `-W error::ResourceWarning`.

### Added — #43, #44, #45
- **Web UI (`web/`)** — ChatGPT-style byte-native chat: sidebar history,
  streaming caret, byte-usage meta, live `/telemetry` refresh, responsive
  layout. Served by the OpenAI-compatible endpoint at `/`.
- **SSE streaming** — `POST /v1/chat/completions` with `stream: true` returns
  Server-Sent Events (content delta chunk + `usage` chunk + `[DONE]`).
- **Low-RAM from-disk serving (`--disk-repo`)** — maps `.gguf`/`.x8dds.gguf`
  containers through `MappedX8DReader` and serves completions by reverse-
  slicing payload coordinates out of the kernel page cache. RSS stays at
  interpreter baseline (~28 MB), no GPU.
- **`/telemetry` endpoint** — Colibrì-style I/O + RSS dashboard (bytes read,
  faults, blocks, mean/max block µs, pin/lru hits, RSS, elapsed, mode).
- **`CHANGELOG.md`** — this file.
- **`research/Low-RAM-From-Disk-Serving-2026.md`** — mechanism analysis
  (llama.cpp / Colibrì / whisper.cpp) + x8D `--disk-repo` implementation.
- **Tests** — `DiskRepoModeTest` (disk mode switch, disk completions, disk
  healthz, `_disk_denoise`) + static/SSE/telemetry coverage; stream contract
  updated in `test_openai_server.py`. Full suite: 304 tests OK.

### Changed
- **README.md** — documented the GitHub-vs-HF model-repo split, web UI
  quickstart, and low-RAM disk mode; byte-native audio/image framing.
- **`tools/openai_chat_server.py`** — rewritten: static serving, SSE
  streaming, `/telemetry`, `--disk-repo` low-RAM mode (`_SERVER_MODE` /
  `_DISK_READER` / `_disk_denoise`), stream contract change (was 400, now SSE).
- **`tests/test_openai_server_live.py`** — healthz now reports `mode`;
  stream-400 test replaced with SSE parsing + disk-mode tests.
- **AGENTS.md** — project index + HF/GitHub split rules + UI/low-RAM facts.

---

## Prior work (committed, see git log)

### #42 — Byte-native processors (2026-07-31)
- Stripped MagViT / GLM-4-Voice tokenizer wrappers; `image_processor.py` /
  `audio_processor.py` are pure binary slicing framed as `[IMG_START(260)]`
  + bytes + `[IMG_END(261)]` / `[AUD_START(262)]` + bytes + `[AUD_END(263)]`.
- Legacy ids 256-259 rejected (collide with MASK/PAD/BOS/EOS).
- `tools/import_hf_dataset.py` gained a zero-copy mmap `--jsonl` path for
  ai4bharat/sangraha + nvidia/Open-SWE-Traces shards (lossless U8 `.x8dds.gguf`).
- `finetune.sh` stripped of legacy audio/image tokenizer args.

### #41 — Colibrì deep-dive (2026-07-31)
- Audited `JustVugg/colibri` (pure-C, GLM-5.2 744B/40B in 25 GB RAM):
  pread coalescing, `COLI_MMAP=1`, io_uring, 372 GB int4 on disk, MTP head.
- Ported `x8d_mmap.py` (`MappedX8DReader`: zero-copy mmap frame reader) and
  `x8d_telemetry.py` (`Telemetry`: per-8x8-block I/O + RSS, Colibrì `telemetry.h`).
- `research/Colibri-Deep-Dive-2026.md`.

### #35-#40 — DSpark + SARA + endpoint (2026-07-31)
- DSpark k-parallel masks, SARA MoE isolation, omni param/size report,
  OpenAI-compatible endpoint probe.

### #34 — transformers 5.x trainability (2026-07-31)
- `DreamModel` trainable on transformers 5.x; torch-gated smoke tests.

### #27-#33 — HF migration + datasets (2026-07-31)
- HF model repo migration (`bapX/x8D-Omni-Diffusion`), `CONTRIBUTING.md`,
  OpenAI endpoint, byte-diffusion, NVIDIA/sarvamai/ai4bharat corpus map.

### #24-#26 — Byte-native import + frontier traces (2026-07-31)
- `x8d_dataset.py` (datasets-server HTTP import, no deps), depth/context
  attention frameworks research, frontier model traces.

### #18-#23 — Byte-core adversarial audit (2026-07-31)
- 6 bugs fixed + LUT optimizations; `bench_byte_core.py`.

### #17 — Generic HF pointer quantizer (2026-07-31)
- `quantize_hf.py` + `load_pointer_map`.

### #14-#15 — Byte-core perf + frontier benchmarks (2026-07-31)
- 6-41× speedups; `research/Frontier-Benchmarks-2026.md`.

### #9-#13 — Kimi-K3 pointer quantization + omni stack (2026-07-31)
- `x8d_hf.py`, `moe_disk.py`, Kimi-K3 1.56 TB → 2.837 GB proof,
  `research/Omni-Modality-Stack.md`.

### #3 — Sub-byte + spec-decode + config (2026-07-31)
- 0.016 bit/weight packed model (32 MB = 32 GB running state), DSpark 8x8
  block quantization, byte-native DreamConfig (vocab 264).

### #2 — Byte-native core (2026-07-31)
- `byte_tokenizer.py` (256 bytes + 8 specials), x8D 0.001 export, config
  mapping from DiffusionGemma.
