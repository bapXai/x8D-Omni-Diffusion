# x8D-Omni-Diffusion — TODO

Tracked from `context.md` (prompt log) and AGENTS.md. Legend:
`[ ]` pending · `[~]` in progress · `[x]` done. Each item maps to a
GitHub issue where applicable.

---

## 🏗️ Rebranding & Repo Hygiene
- [ ] Audit all stale `lijiang` / `Omni-Diffusion` / `VITA-MLLM` references in code, docs, README, configs, HF model card — replace with `bapX` / `x8D-Omni-Diffusion` (issue needed).
- [~] Move `docs/index.html` so GitHub Pages serves at `https://bapxai.github.io/x8D-Omni-Diffusion` (currently 404). Verify Pages build + enforce HTTPS on `bapXai/x8D-Omni-Diffusion`.
- [ ] Verify HF model repo `bapX/x8D-Omni-Diffusion` has NO leftover `*.safetensors` / BPE artifacts; run the delete-files dry-run + actual delete; re-verify with `hf models list -R`.
- [x] Add CONTRIBUTING.md.

## 🔬 Model Serving (real experts, no full model)
- [~] Verify `KokoroTTS` end-to-end from `/tmp/kokoro.x8dgguf` after the espeak/phonemizer stub + de2acfcc code pin (last run failed on Decoder `disable_complex`; fixed — retest).
- [ ] Fix `bytes_in` in `omni_diffusion/x8d_quanta.py` (`len(tensor)` = numel, not bytes; use `element_size()*numel()`).
- [ ] Write tests: `tests/test_x8d_quanta.py` (round-trip quantize→dequantize error bound, container structure, determinism) + `tests/test_x8d_expert.py` (KokoroTTS non-silent PCM from container only).
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
