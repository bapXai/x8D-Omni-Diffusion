# Changelog

All notable changes to x8D-Omni-Diffusion are documented here.
Format: `[#issue]` references GitHub issues; commits are on `main`.

## [Unreleased]

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
