# x8D-Omni-Diffusion — Objective

The north star of this project. Updated from `context.md` and `todo.md`.

---

## Mission
Build a **byte-native omni diffusion model** — the `x8D-Omni-Diffusion` rebrand of
the DREAM/DiffusionGemma-style masked discrete diffusion stack — where the **only
vocabulary is raw 8-bit bytes (0–255)**, every pre-trained weight that enters the
repo is compressed with the **x8Dsub-byte 0.001 threshold law**, and the
**compressed state IS the running state** (zero-copy mmap, `/0.001` reversed live
at query time only on the specific MoE expert needed).

## Principles (non-negotiable)
1. **No tokens, ever.** No BPE/SentencePiece/WordPiece, no vocab.json/merges.txt.
   Embed/lm_head = 264 (256 bytes + MASK/PAD/BOS/EOS/IMG/AUD specials).
2. **Bytes are universal.** Text (UTF-8), images (pixel bytes), audio (PCM bytes),
   code, binaries — all native at ids 0–255. 16/32/64-bit is just bytes composed.
3. **0.001 sub-byte law.** `Quanta[i] = weight_byte[i] × 0.001`. Never round at
   storage; inverse `/0.001` only at compute, per needed expert. 16-bit→0.016
   bit/weight (32 GB → 32 MB); 32-bit→0.032.
4. **No full models.** The full float checkpoint exists only as the one-time
   quantization input and is deleted immediately after the container is written.
5. **Issue-first, prove the job.** GitHub issue before code; commit references the
   issue; tests green; CI validated; HF model repo dual-synced; AGENTS.md +
   research/ updated.

## Goals
- **G1 — Quantized omni serving.** Serve text (Kimi-K3 2.78T), TTS (Kokoro 82M),
  ASR (Whisper 1.55B), image/video (LTX-2 19B) — each as an isolated SARA expert —
  entirely from `x8d_weights/*.x8dptr.gguf` + `*.x8dgguf` containers, never from a
  full checkpoint.
- **G2 — Real model output.** Replace every fake/procedural response with genuine
  forward passes (KokoroTTS real audio is the first proof; fix #47 mojibake).
- **G3 — Combined omni model.** One model over all compressed experts and datasets;
  report which MoE expert activates per query; a dense model = a single expert.
- **G4 — Disk-resident inference.** Low-RAM from-disk serving (RSS ~28 MB, no GPU)
  via `MappedX8DReader` (Colibrì COLI_MMAP port) + Colibrì-style telemetry.
- **G5 — Byte-diffusion training.** Train a byte-native masked discrete denoiser
  (entropy_bound sampler, canvas 256) from scratch on NVIDIA/sarvamai/ai4bharat
  byte-stream datasets, with QAT/STE and DSpark 8×8 block-parallel generation.
- **G6 — Frontier architecture.** Port DeepSeek-V4 / Kimi-K3 mechanics: MoE routing,
  DSpark semi-autoregressive confidence verification, KDA/MLA hybrid attention,
  AttnRes/MTP depth frameworks (research-tracked in `research/`).
- **G7 — OpenAI-compatible surface.** One endpoint (`tools/openai_chat_server.py`)
  serving chat (SSE, byte usage), `/v1/audio/speech`, `/v1/images/generations`,
  `/telemetry`, `/healthz`, web UI.
- **G8 — Clean distribution.** HF model repo `bapX/x8D-Omni-Diffusion` byte-native
  only, model-card YAML metadata fixed, `trust_remote_code=True` runtime set;
  GitHub `bapXai/x8D-Omni-Diffusion` = source + docs + tests + research.

## Current focus (from todo.md)
1. Finish real Kokoro TTS from the quantized container (fix the de2acfcc pin + stub
   verification) and wire it into the server `/v1/audio/speech`.
2. Fix `bytes_in` metric in `x8d_quanta.py`; add `test_x8d_quanta.py` + `test_x8d_expert.py`.
3. GitHub Pages at `https://bapxai.github.io/x8D-Omni-Diffusion` (move index into docs/).
4. HF model-card YAML metadata fix + upload quantized weights to the HF repo.
5. Full rebrand audit (lijiang/VITA-MLLM → bapX/bapXai/x8D).

## Definition of done
- All tests green (`unittest discover -s tests -v` and `-W error::ResourceWarning`).
- Every query type (text/image/audio/binary) produces real output from compressed
  weights — no full model, no fake generation.
- GitHub commits + CI green + HF model repo synced, verified with `gh run list`
  and `hf models list -R`.
- `AGENTS.md`, `context.md`, `todo.md`, `objective.md`, `CHANGELOG.md`,
  `research/` all updated to reflect the actual shipped state.
