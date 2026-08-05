# x8D-Omni-Diffusion — TODO

Tracked from `context.md` (prompt log) and AGENTS.md. Legend:
`[ ]` pending · `[~]` in progress · `[x]` done. Each item maps to a
GitHub issue where applicable.

---

## 🗜️ .x8D Re-quantization (2026-08-01, #51/#52)
- [x] Delete old HF `x8d_weights` (commit `060122ad`; #51/#52).
- [~] Rewrite quantizer to `.x8D` streaming output — disk = source_bytes × 0.001
      (0.008 bit per weight byte), no container/magic/headers/padding, lossless
      arithmetic coding via `omni_diffusion/x8d_arith.py`.
- [ ] Quantize Whisper, Kokoro, Kimi-K3, LTX-2 to `.x8D`.
- [ ] Run the full suite (`python3 -m unittest discover -s tests -v` +
      `-W error::ResourceWarning`) green BEFORE uploading.
- [ ] Upload `.x8D` weights to HF model repo `bapX/x8D-Omni-Diffusion`.
- [ ] QAT-aware fine-tuning on tier-0/1/2 datasets on top of the quantized `.x8D` weights.

## 🗜️ QAT Fine-Tuning Scaffold (2026-08-05)
- [x] `omni_diffusion/x8d_qat.py` — pure-stdlib QAT core: `quantize_ste` (STE fake-quant
      `round(clamp(w,0,255))`, torch-lazy `x.round().clamp()+ (x-x.detach())`), `hard_quantize`,
      `ste_grad`, `QATWrapper`/`wrap_for_qat` (dict or `named_parameters()`), `x8d_qat_roundtrip_loss`,
      264-vocab `byte_diffusion_loss`, `mask_canvas`/`renoise_to_random_bytes` delegates to
      `byte_diffusion.ByteDiffusionSampler`, `QATConfig` (steps=48, entropy_bound=0.1, canvas=256).
- [x] `tools/finetune_qat.py` — load `.x8D` via `QuantizedServingReader`, `fine_tune_qat` loop
      (split canvas -> mask -> renoise -> loss -> fake step, byte-aligned final weights), synthetic
      offline data + CLI.
- [x] `tests/test_x8d_qat.py` — 28 tests OK (1 torch-gated skip); full suite 389 OK (8 skipped).
- [ ] Wire `fine_tune_qat` into the real training path (`trainer_v4_51_3.py` Trainer hook) once torch
      is available — replace the `_pseudo_logits` surrogate with the real torch denoiser over ids 0-255.

## 🏗️ Rebranding & Repo Hygiene
- [ ] Audit all stale `lijiang` / `Omni-Diffusion` / `VITA-MLLM` references in code, docs, README, configs, HF model card — replace with `bapX` / `x8D-Omni-Diffusion` (issue needed).
- [~] Move `docs/index.html` so GitHub Pages serves at `https://bapxai.github.io/x8D-Omni-Diffusion` (currently 404). Verify Pages build + enforce HTTPS on `bapXai/x8D-Omni-Diffusion`.
- [ ] Verify HF model repo `bapX/x8D-Omni-Diffusion` has NO leftover `*.safetensors` / BPE artifacts; run the delete-files dry-run + actual delete; re-verify with `hf models list -R`.
- [x] Add CONTRIBUTING.md.

## 🔬 Model Serving (real experts, no full model)
- [~] [#50] Fix x8D quantization to apply the 0.001 law: `Quanta[i] = byte[i] × 0.001` stored as sub-byte coordinates (NOT raw U8 bytes = 1.0 row/1:1, NOT Q8 scale + manifest.json = Float Trap). Update `omni_diffusion/x8d_quanta.py` (remove scale/manifest + destructive `round(q*LAW)`; faithful port of `/tmp/x8Dsub-byte/x8Dquanta/__init__.py`) and `omni_diffusion/x8d_export.py` (store quanta coordinates, not raw bytes). x8d_quanta.py rewrite DONE (no magic, bijective 256/256); x8d_export.py no-magic alignment NOT yet done.
- [~] Verify `KokoroTTS` end-to-end from `/tmp/kokoro.x8dgguf` after the espeak/phonemizer stub + de2acfcc code pin (last run failed on Decoder `disable_complex`; fixed — retest). Note: `/tmp/kokoro.x8dgguf` is a STALE wrong-format container (Q8 + manifest) — must be rebuilt in the raw-quanta format first; source `kokoro-v1_0.pth` is deleted so a rebuild needs a fresh source.
- [~] [#50] `x8d_expert.py` updated to slice the raw quanta blob by target-architecture shapes (sorted state_dict order) + `quantize_state_dict_to_gguf()` added to `x8d_quanta.py` — needs end-to-end Kokoro test against a correctly built raw container.
- [ ] Fix `bytes_in` in `omni_diffusion/x8d_quanta.py` (`len(tensor)` = numel, not bytes; use `element_size()*numel()`).
- [ ] Write tests: `tests/test_x8d_quanta.py` (0.001-law quantize→dequantize bijective over 0-255, container structure = raw quanta coords, no manifest.json) + `tests/test_x8d_expert.py` (KokoroTTS non-silent PCM from container only).
- [ ] Wire `KokoroTTS` into `tools/openai_chat_server.py` `/v1/audio/speech` (real audio, no full model).
- [ ] Commit quantized Kokoro container + `KokoroTTS`; push; verify CI.
- [ ] Whisper ASR expert from quantized container (CPU/MPS-runnable).
- [ ] LTX-2 (image/video, GPU-gated) + Kimi-K3 (text, GPU-gated) from `x8dptr.gguf` pointer maps.
- [ ] Per-expert active-parameter report + combined omni-model size table (GLM-5.2 753B + Kimi-K3 2.78T + DeepSeek-V4-Pro 1.6T + Kokoro 82M + Whisper 1.55B + LTX-2 19B).
- [ ] Remove obsolete `omni_diffusion/x8d_media.py` + `tests/test_x8d_media.py` (procedural fake media — superseded by real-model path).

## 🗜️ x8Dsub-byte Runtime & Compression
- [ ] `SandboxComput.bin` — compress a `.venv` with the byte-native deps into an x8D container; benchmark vs a regular venv on this machine (byte-core + openai server startup + a TTS synth).
- [ ] Learned PIN hot-store from hit histograms; expose dashboard line via `/healthz` (Colibrì next steps, #41).
- [ ] Publish like-for-like benchmark vs Colibrì upstream.
- [ ] Wire `MappedX8DReader` into `moe_disk.py` SARA spans (Colibrì COLI_MMAP port completion, #41/#36).
- [ ] Formalize DSpark parallel block-mask scheduler config for multi-modal diffusion + language throughput (#7 / #47 follow-ups).

## 📚 Datasets & Training
- [ ] Import NVIDIA + sarvamai + ai4bharat corpora via `tools/import_hf_dataset.py` (byte-native `.x8dds.gguf`): Open-SWE-Traces, Nemotron agentic/tool-use/terminal/RAG/reward/pretraining, OpenCodeReasoning I+II, openmath, physical-ai omni-dreams, nemotron-personas, indic-diarbench (Assamese/Bengali/Bodo validated), samvaad-hi-v1, mmlu-indic, sangraha, samanantar, IndicVoices.
- [ ] Byte-diffusion training run (entropy_bound sampler, #5/#6): small config train-from-scratch on a byte dataset for a few hundred steps.
- [ ] DiffusionGemma-style text-diffusion validation over the 264-byte space (canvas_length=256, diffusion_entropy_bound=0.1).

## 🧪 Testing & Validation
- [ ] `python3 -m unittest discover -s tests -v` (pure stdlib) green.
- [ ] `python3 -W error::ResourceWarning -m unittest discover -s tests -v` green.
- [ ] Torch-gated tests (`HAS_TRANSFORMERS`) green in `/tmp/x8d_torch`.
- [ ] `gh run list` green after every push.

## 📦 HF Model Repo & Distribution
- [ ] Fix HF model-card YAML metadata warning (empty/missing metadata) — add proper model-card metadata + `inference` block.
- [ ] Upload quantized weight containers (`kokoro.x8dgguf`, plus `x8d_quanta.py`, `x8d_expert.py`, config) to `bapX/x8D-Omni-Diffusion`.
- [ ] Verify `trust_remote_code=True` loading works from the HF repo (byte-native runtime set per #44 split).

## 📄 Docs & Research
- [ ] Update `research/` with: DiffusionGemma train-from-scratch landscape, DeepSeek-V4 / Kimi-K3 papers (AttnRes 2603.15031, Kimi Linear 2510.26692, KDA/KBP, DSpark), Fable 5 / GPT-5.6 Sol / Opus 5 extracted traces, computer-use/tool-use/omni/world-sim/3D datasets.
- [ ] Keep `context.md`, `todo.md`, `objective.md`, `CHANGELOG.md` current with every prompt and merge.

---

## Quick-verify commands
```bash
export PATH="/Users/getwinharris/.local/bin:$PATH"
python3 -m unittest discover -s tests -v
gh run list --limit 5
hf auth whoami
hf models list bapX/x8D-Omni-Diffusion -R
```
