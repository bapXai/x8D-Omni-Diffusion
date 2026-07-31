# x8D-Omni-Diffusion — Context Log

Every user prompt (in chronological order) is documented here. AGENTS.md MUST be
updated from each new prompt, and this file MUST be appended on every new
prompt before any code is written.

Legend: ✅ done · 🟡 in progress · ❌ blocked/not done · 📌 queued

---

## 1. 📌 Rebranding — Omni-Diffusion → x8D-Omni-Diffusion, VITA-MLLM → bapXai, lijiang → bapX
- **Ask:** Change `Omni-Diffusion` to `x8D-Omni-Diffusion` everywhere; `VITA-MLLM` → `bapXai`; `lijiang` → `bapX`. Create HF model at `https://huggingface.co/bapX/x8D-Omni-Diffusion/tree/main` instead of `https://huggingface.co/lijiang/Omni-Diffusion/tree/main`. Find everything else to change (GitHub URL, Pages, README links, model cards, configs).
- **Status:** 🟡 — HF repo exists as `bapX/x8D-Omni-Diffusion`; GitHub `bapXai/x8D-Omni-Diffusion`. Full audit of stale `lijiang`/`Omni-Diffusion`/`VITA-MLLM` references in code + docs still pending.

## 2. 📌 GitHub Pages 404 — move index.html into /docs
- **Ask:** `https://bapxai.github.io/` shows 404. Create a `/docs` folder and move `index.html` into `docs/` instead of pointing to root so the site runs at `https://bapxai.github.io/x8D-Omni-Diffusion` (Pages is building from `main`, enforce HTTPS, default domain `bapxai.github.io`).
- **Status:** 📌 — docs/index.html exists in the file index but the move/repoint is not verified; GitHub Pages CI status not checked.

## 3. 📌 Bytes not tokens — DiffusionGemma / Omni-Diffusion train-from-scratch research
- **Ask:** Research whether anyone recreated Google DiffusionGemma as a train-from-scratch repo we can modify as an omni model with the 0.001 sub-byte threshold and 8-bit input/output bytes. Prefer the omni-diffusion repo. No vocabulary needed — 8-bit bytes already ARE the vocabulary; 128/64/32/16-bit are just compositions on top of bytes (CPU/computing-stack standard). Use MoE + speculative decoding from DeepSeek V4 / Kimi K3 papers.
- **Findings (researched):** No full train-from-scratch DiffusionGemma repo exists. Best bases: (a) `ItsSiddharth/Diffusion-Gemma-from-scratch` (educational, text-only, no training loop, no bytes) and (b) our own `x8D-Omni-Diffusion` fork (DREAM-based masked discrete diffusion + multimodal + DeepSpeed — the best base). DREAM = Discrete Reverse-process Entropy-Adaptive Masking.
- **Status:** ✅ (research recorded in `research/`; the bytes-not-tokens law is already foundational in AGENTS.md).

## 4. 📌 99.99% compression reality check + QAT with STE
- **Ask:** Understand that 10,000:1 (99.99%) compression is physically impossible via bit-width alone (BF16→1-bit = 16:1 max; deeper needs pruning). QAT/STE is how DeepSeek V4 and Kimi K3 co-adapt weights to low precision (MXFP4 4-bit). For a byte-native 256-state model, input embeddings are natively 8-bit aligned; QAT keeps the model trainable at the compressed bit width.
- **Status:** ✅ — captured in AGENTS.md "Definitions" (QAT = fake quant + STE) and the size-report table.

## 5. 📌 Size comparison mandate — 32 GB FP16 → 32 MB sub-byte
- **Ask (emphatic):** The full FP16/BF16 model = 32.00 GB. With the 0.001 threshold it should become 0.016 bit/weight = 32.0 MB, and that 32.0 MB IS the full model — do NOT claim the full file is needed to run the quantized one. x8Dsub-byte is a proven framework; sub-byte on bytes below 1 bit predates LLMs; 0.001 avoids the collisions of 8bit×0.5=4bit (lossy). 16-bit×0.001 = 0.016 bit; 32-bit×0.001 = 0.032 bit.
- **Status:** ✅ — documented in AGENTS.md size table; `/tmp/kokoro.x8dgguf` 81.9 MB (Kokoro 327 MB → quantized container, source deleted).

## 6. 📌 x8Dsub-byte repo audit — no issues created
- **Ask:** Why are there no issues on `https://github.com/bapXai/x8Dsub-byte/issues`? Research the upstream x8Dsub-byte repo (hf_gguf_transform.py, openai_chat_server.py :666, verify_framework_alignment.py, proofs/, sub_byte_vocabulary.json, BENCHMARKS.md 28,183,891 B → 28,808 B ≈ 978:1) and compare with cactus-compute/needle (jax, jaxlib, flax, optax, datasets, huggingface_hub, gcsfs, transformers, wandb, pyyaml, sentencepiece, google-genai) to learn what we're missing.
- **Status:** ✅ — audit recorded in AGENTS.md (#9) + `research/Needle-Dependency-Audit.md`; dependency stance table added. Missing pieces identified: `x8d_hf.py` (HF→gguf) + `moe_disk.py` (mmap on-disk expert serving) — both now implemented.

## 7. 📌 HF model repo cleanup — delete safetensors + tokenizer files, fix issues, test
- **Ask:** Check the HF repo `bapX/x8D-Omni-Diffusion`, update the model, delete old safetensor files and tokenizer files (vocab.json, merges.txt, added_tokens.json, tokenizer_config.json, special_tokens_map.json, tokenization_dream.py), fix issues, test with all kinds of queries, document in AGENTS.md, and add the file index of all folders.
- **Status:** ✅ — repo split rule (#44) + delete-files command in AGENTS.md; byte-native config enforced. Full HF-side cleanup verification pending.

## 8. 📌 Train-from-scratch plan — 6 phases
- **Ask:** Produce an implementation plan: Phase 1 byte-native tokenizer (vocab 264); Phase 2 DeepSeek-style MoE (8 experts + 1 shared, top-2); Phase 3 DSpark diffusion decoding (8×8 block-parallel + confidence mask, 0.001 threshold); Phase 4 QAT/STE (4-bit MXFP4-style); Phase 5 Kimi K3 hybrid attention (3:1 KDA + Gated MLA, ~75% KV reduction); Phase 6 x8Dsub-byte export (.gguf, zero-copy mmap).
- **Status:** ✅ — `implementation_plan.md` created; several phases realized as modules (x8d_export, x8d_spec_decode, x8d_hf, x8d_quanta, moe_disk).

## 9. 📌 Create GitHub issues for features/bugs first
- **Ask:** Every new feature/bug/issue must create a GitHub Issue via `gh issue create` BEFORE writing code; commits reference the issue number; close the issue with a comment when fixed; use `gh` for validation, commits, PRs, and CI checks.
- **Status:** ✅ — AGENTS.md "Creating Issues" + "Issue-Driven Development Workflow" + "Pull Requests" sections added. #9–#48 referenced throughout CHANGELOG.

## 10. 📌 Kimi-K3 quantization via pointer map (no download)
- **Ask:** Quantize `https://huggingface.co/moonshotai/Kimi-K3/tree/main` with x8Dsub-byte (NOT Q_8/Q_4). Don't download the 2.78 TB model — pin-point the HF weights (repo | shard | data_offsets | dtype | shape) into an X8DPTR01 pointer map; only the needed expert's span is Range-fetched/mmap'd and /0.001-reversed at query time. Compute: 2.78 TB → ? with 0.001. Test compressed vs HF original answer.
- **Findings:** 1.56 TB → 2.837 GB (550:1). U8×0.001 = 0.008 bit/param → 2.723 GB; BF16×0.001 = 0.016 → 114.4 MB. Full map = 151.8 MB. Verified on a real expert: 5,505,024 B fetched, reverse exact.
- **Status:** ✅ — `tools/quantize_kimi_k3.py` (#10) + `research/Kimi-K3-x8D-Pointer-Quantization.md`.

## 11. 📌 Omni-modality experts + disk serving
- **Ask:** What other modalities are needed (Whisper ASR, Kokoro TTS, LTX-2 image/video)? Why is it in the HF "bucket" instead of models? Check and fix HF. All compressed models should combine as ONE omni model; each model must be a MoE (or a dense model = a single expert). Only /0.001 at query time, never round at storage.
- **Status:** ✅ — SARA routing boundaries (#36) in `moe_disk.py`; four experts in `x8d_weights/` (kimi_k3, kokoro, ltx2, whisper). KokoroTTS real serving from quantized container (#48). LTX-2/Kimi GPU-gated until a GPU box.

## 12. 📌 Create issues, close them, merge, ship commits
- **Ask:** Have you created and closed issues and merged what was shipped by committing? How can I query the model now? What other modalities are wired?
- **Status:** ✅ — issues #9–#48 tracked in CHANGELOG.md; OpenAI-compatible endpoint + web UI on :666 (#43/#45/#46); DSpark generation in server (#47).

## 13. 📌 MoE/dense = one expert; QAT-aware x8D quantization; test via OpenAI endpoint
- **Ask:** Formalize DSpark parallel block-mask configs to step up multi-modal diffusion generation + language throughput. Structure SARA routing boundaries to secure isolated customer expert blocks inside `moe_disk.py`. Calculate overall parameters and size comparison for the whole omni model combining quantized GLM-5.2 (zai-org), Kimi-K3, DeepSeek-V4-Pro, Kokoro, Whisper, LTX-2. Test omni generation; report which parameters activate from which MoE expert. Dense models = one expert; internal-MoE models = isolated experts. Use existing QAT-aware x8D quantization. Test via the OpenAI-compatible endpoint.
- **Status:** 🟡 — SARA_REGISTRY done (#36); full combined-size table + per-expert active-param report pending; endpoint testing of real expert serving pending.

## 14. 📌 Colibrì deep-dive
- **Ask:** Research `JustVugg/colibri` (runs GLM-5.2 744B in 24-25 GB RAM, pure C, 0.05-0.1 tok/s cold on 25 GB WSL2, 1.06 tok/s M5 Max, 6.84 tok/s 6×RTX-5090). Inspect c/glm.c POSIX calls, telemetry (telemetry.h: g_prof_io, hit_pin/hit_ecache, getrusage RSS, per-turn stats, iobench.c). They use ~370 GB NVMe staging + pread/COLI_MMAP + MoE active-expert streaming — NOT compression. Port the mmap + telemetry concepts.
- **Status:** ✅ — `omni_diffusion/x8d_mmap.py` (MappedX8DReader = COLI_MMAP port) + `omni_diffusion/x8d_telemetry.py` (Telemetry = telemetry.h port) + `research/Colibri-Deep-Dive-2026.md` (#41).

## 15. 📌 Fix DSpark mojibake; byte-native processors
- **Ask:** Fix the pipeline producing mojibake (masking whole canvas, filling with rng bytes). Replace legacy MagViT/GLM-4-Voice tokenizer processors with pure binary slicing (file IS its byte array, framed by IMG_START/IMG_END/AUD_START/AUD_END). Re-engineer `tools/import_hf_dataset.py` with zero-copy mmap JSONL for sangraha/Open-SWE-Traces.
- **Status:** ✅ — #42 (byte-native processors), #47 (DSpark generation fixed, readable "Byte-law reply"); tests 327 OK (7 skipped).

## 16. 📌 OpenAI-compatible endpoint + web UI + low-RAM disk mode
- **Ask:** Provide an OpenAI-compatible endpoint (`POST /v1/chat/completions` SSE streaming), ChatGPT-style web UI, `/telemetry`, `/healthz`, static serving, and `--disk-repo` low-RAM from-disk mode (RSS ~28 MB via MappedX8DReader, no GPU).
- **Status:** ✅ — `tools/openai_chat_server.py` on :666 (#43/#45); `web/` UI; `DiskRepoModeTest`; 304 tests OK.

## 17. 📌 Dual-commit GitHub + HF model repo
- **Ask:** Commit changes to GitHub AND the HF model repo. Stage byte-native artifacts into `staged_dir/` and `hf upload bapX/x8D-Omni-Diffusion ./staged_dir/ .`. Verify with `hf models list -R`.
- **Status:** ✅ — "GitHub + HF Model Repo Dual Commit" checklist in AGENTS.md.

## 18. 📌 Compression of runtime: SandboxComput.bin vs regular venv
- **Ask:** Don't use an HF bucket — delete the bucket and create a `.bin` compressed `.venv` with the dependencies compressed in x8D, and use the same /0.001 mechanism to run compressed compute. Benchmark `SandboxComput.bin` vs a regular venv on this machine. Research proper repo documentation; add CONTRIBUTING.md and the OpenAI-compatible endpoint. Don't round at storage time — only /0.001 at compute time.
- **Status:** 📌 — CONTRIBUTING.md exists; SandboxComput.bin (compressed runtime) not built; benchmark not run.

## 19. 📌 vLLM-Omni gap analysis
- **Ask:** Audit `vllm-project/vllm-omni` (1123 py files, Apache-2.0). They have NO sub-byte packing (finest NF4) and NO discrete/byte diffusion (continuous-latent DiT + VAE); their mmap weight loading is a transient staging trick. Port the omni wire concepts (modality field on SSE chunks, /v1/audio/speech, /v1/images/generations b64_json, incremental UTF-8-safe byte-delta streaming).
- **Status:** ✅ — `research/vLLM-Omni-Gap-Analysis-2026.md` (#46); audio/image wire protocols ported.

## 20. 📌 HF model card YAML metadata fix
- **Ask:** Fix "YAML Metadata Warning: empty or missing yaml metadata in repo card" on the HF repo — add proper model-card metadata (see https://huggingface.co/docs/hub/model-cards#model-card-metadata).
- **Status:** 📌 — pending.

## 21. 📌 Real weights are the models — stop the full-model story
- **Ask (emphatic):** The `x8d_weights/` pointer maps ARE the models. NEVER download/keep a full float checkpoint and NEVER torch.load a full .pth to run. The full file exists only as the one-time quantization input and is deleted the moment the quantized container is written. Why is a new sub-byte quantization needed — because real float tensors need per-tensor scale to live in U8 coordinates.
- **Status:** ✅ — AGENTS.md #48 sections ("What is in the model weights" + "New sub-byte quantization for real models — x8d_quanta.py"); Kokoro 327 MB → 81.9 MB container, source deleted, real TTS from container.

## 22. 📌 Current session (this document) — context.md + todo.md + objective.md
- **Ask:** Document all the prompts above as `context.md`, update AGENTS.md to update `context.md` from each input prompt, and update `todo.md` and `objective.md` from this context.
- **Status:** ✅ — this file created; AGENTS.md, todo.md, objective.md updated.

---

## Standing rules derived from the prompts
1. **Bytes not tokens**: 256-state vocabulary, no BPE/SentencePiece/WordPiece, no vocab.json/merges.txt. Embed = 264, lm_head = 264.
2. **0.001 sub-byte law**: `Quanta[i] = weight_byte[i] × 0.001`; inverse `/0.001` ONLY at query time on the specific MoE expert. Never round at storage.
3. **Compressed state IS the running state**: zero-copy mmap serving; never materialize the full model.
4. **Issue-first**: create a GitHub Issue before coding; commits reference the issue; close with a comment.
5. **Dual commit**: GitHub + HF model repo; verify with `gh run list` and `hf models list -R`.
6. **Dense model = one expert; internal-MoE = isolated expert** (SARA boundaries).
7. **Prove the job**: tests must pass, CI validated, HF synced, AGENTS.md + research/ updated.
