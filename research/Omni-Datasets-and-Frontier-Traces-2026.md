# Omni-Datasets & Frontier Traces 2026 — Byte-Native Tier Mapping (issues #25/#26)

Status: researched 2026-07-31 from Hugging Face dataset cards, arXiv, OpenAI/Anthropic/
NVIDIA/Google release pages. Cross-references `Training-Dataset-and-Quantization-Plan.md`
(Tier 0/1/2), `DiffusionGemma.md`, `Frontier-Benchmarks-2026.md`,
`Depth-Context-Attention-Frameworks-2026.md`, and `AGENTS.md` (byte law, vocab 264,
`x8d_dataset.py`).

Goal: map the 2026 frontier dataset/trace landscape (NVIDIA collections, sarvamai +
ai4bharat Indic corpora, community-extracted frontier-model traces) onto x8D's
byte-native Tier 0/1/2 training plan, and name the concrete optimization levers.

---

## 1. TL;DR

The byte law turns every 2026 "token-era" dataset into a raw byte stream. Text is UTF-8
bytes, images are pixel bytes, audio is PCM bytes, agent traces are JSONL/UTF-8 bytes —
all already inside our 256-state vocabulary. That makes corpus choice a question of
**behavior we want**, not vocabulary compatibility. Three levers this doc feeds:

1. **Data mix** — Tier 2 SFT from agentic/computer-use/tool-use traces byte-encoded with
   `[IMG_START]`/`[AUD_START]` + tool-call markup; Indic corpora for byte-level script
   diversity (Devanagari, Tamil, Telugu, Gurmukhi, …); Tier 1 Physical-AI/omni-dreams
   pixel+PCM bytes for world-sim behavior.
2. **Serving** — every corpus above is imported through `omni_diffusion/x8d_dataset.py`
   (`block_compress_dataset` → `.x8dds.gguf`, 8×8 DSpark, threshold 0.001) and lives on
   the HF model repo `bapX/x8D-Omni-Diffusion`; pointer-map/mmap serving (see `x8d_hf.py`,
   `moe_disk.py`) extends from weights to dataset shards (Range-fetch only the shard span
   a training step / eval needs).
3. **Sampler** — adopt DiffusionGemma's entropy-bound + 256-byte canvas block-autoregressive
   diffusion wholesale; DiffusionGemma is the proof that diffusion applies to **language**,
   and over a 264-vocab it is ~1600× cheaper to self-condition than over a 262K vocab.

---

## 2. Frontier-model trace extraction (Fable 5, GPT-5.6 Sol, Opus 5)

### 2.1 Claude Fable 5 agent traces (verified)

Anthropic shipped **Claude Fable 5** on 2026-06-09 ("Mythos-class": Fable 5 = Mythos 5 with
safety classifiers; Mythos 5 restricted to Project Glasswing cyber partners). Traces were
rapidly scraped by the community via Claude Code harnesses and re-published on HF:

| Dataset | Size | License | Notes |
|---|---|---|---|
| `ApertureQA/Fable-5-traces` (alias `Glint-Research/Fable-5-traces`) | 4,665 Pi trace sessions from 60 source sessions; 3,799 tool actions + 866 text actions | AGPL-3.0 | Flat `fable5_cot_merged.jsonl` + HF Agent-Traces view; provenance `local`/`hf` |
| `AlinCiocan/fable-5-claude-code-traces` | 18 sessions / 9,497 JSONL events (2,506 assistant, 1,424 user, 3,428 tool-result) | CC-BY-4.0 | Full scrubbed release, revision `v1.0-full-scrubbed`, deterministic recursive scrubber |
| `greghavens/fable-5-coding-and-debugging-traces` | 2,374 trajectories / 12,448 training rows; 14 MB parquet / 662 MB JSONL | CC-BY-4.0 | "moonshiner" harness; Codex-judged, deterministic-test verified, trajectory-disjoint train/val |
| `Poumrm/claude-fable-5-claude-code` | raw sessions | (see card) | **Explicitly warns**: Glint-Research was derived from the same underlying logs — "don't use both" |

What they encode: **real agentic coding behavior** — multi-turn `Bash`/`Edit`/`Read`/`Write`
tool loops, deferred `Agent` calls, `WebSearch`, MCP tool schemas, retries after tool errors,
headless screenshot/eval MCP calls, and genuine chain-of-thought (`claude-fable-5` always
thinks). Good for tool-call policy distillation, not just text.

**Dedup requirement (verified, acute).** Several "Fable-5" repos are re-splits or mirrors of
the same ~60 source sessions: `ApertureQA` ≈ `Glint-Research`, `Poumrm` feeds `Glint-Research`,
`Quaxicron`/`moehamid`/`ArkhAngelLifeJiggy`/`kelexine`/`Swarm-AI-Research` re-derive the same
material. `Crownelius/Complete-FABLE.5-traces-2M` documents the failure mode quantitatively:
an advertised ~2,006,487 "FABLE.5" rows collapsed to ~66K **content-verified** rows after
provenance filtering (1.1M rows were template filler, 775K were synthetic SFT). **Rule: only
use verifier-backed sources (greghavens) plus one canonical session dump (AlinCiocan or
Glint-Research), never all mirrors.** License floor is AGPL-3.0 for the CoT subset.

### 2.2 GPT-5.6 Sol traces (verified)

OpenAI GA'd GPT-5.6 (Sol/Terra/Luna) on 2026-07-09. Official eval highlights relevant to what
traces can distill: **OSWorld 2.0 (computer use) 62.6%** (surpassing Opus 4.8 with ~85% fewer
output tokens), **BrowseComp 90.4% / 92.2% Ultra**, **BenchCAD 70.6%**, **Toolathlon 58%**,
**AutomationBench 18.1%**, Terminal-Bench 2.1 88.8%/91.9% Ultra, AA Coding Agent Index 80.

| Dataset | Size | License | Notes |
|---|---|---|---|
| `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | 17,939 cumulative rows / 1,474 source trajectories, incl. 4 seed-authoring + 629 model-judge; 1.05 GB `traces.jsonl` | CC-BY-4.0 | Codex CLI, reasoning `xhigh`; domains `coding`/`security`/`harness`; `teacher_model: gpt-5.6-sol`; acceptance-test verified; live-growing |
| `Manusagents/GPT-5.6-Sol-Luna-Terra-Traces` (and `Crownelius/...`) | mirror library: 5,402 verified Sol rows | attributed | **Not crypto-certifiable** — provenance is content-verified (Codex `call_…` tool-IDs), source-asserted |

What they encode: the **full observable dev loop** — repo inspection, failure reproduction,
file edits, compile/test runs, mistake correction, final verification; plus security
(surface/classification/remediation/whole-repo static audits), harness construction, and
**model-judge trajectories** (Sol independently judging other models' code — usable as a
reward-modeling signal). No Terra data on HF as of 2026-07-31; Luna exists only as a
persona-injected synthetic distill (low value).

### 2.3 Claude Opus 5 traces — NOT FOUND (unverified)

No dedicated **Claude Opus 5** trace dataset was found on HF in this pass. Closest available:
`sammshen/wildclaw-opus-traces` (Opus 4.6, 60 tasks / 687 records, instrumented proxy),
`netpreme/coding_agent_traces` (Opus 4.7, 731 SWE-Bench Pro sessions / 32,579 turns),
`livesweagent/claude-opus-4-5_swebench_verified_traj`, `DiscoPosse/agent-llm-traces` (Opus 4.5,
1,781 OTel traces across 6 benchmarks). Mark **unverified** whether a clean Opus-5 trace
release exists — if it appears, apply the same dedup + permissive-license gate as Fable 5.

---

## 3. NVIDIA collections (verified unless noted)

### 3.1 Agentic / tool-use / SWE / terminal

| Dataset | Size | License | Notes |
|---|---|---|---|
| `nvidia/Open-SWE-Traces` (arXiv:2606.16038) | **207,489 trajectories**, 9 langs (Python 23.2%, Go 22.6%, TS 17.8%, JS 14.2%, Rust, Java, PHP, C, C++) from 20k real PRs | permissive repos only (MIT/Apache-2.0/BSD) | Dual-mode: MiniMax-M2.5 thinking + Qwen3.5-122B non-thinking; OpenHands 50.8% + SWE-agent; ~34% pass rate (v2; v1 paper said 40.6%); FT Qwen3-30B-A3B → **61.7% SWE-bench Verified, 57.1% Multilingual, 36.8% Pro** |
| `nvidia/Nemotron-Agentic-v1` | 335,122 samples = interactive_agent 19,028 + tool_calling 316,094 | CC-BY-4.0 | 3-role LLM-simulated tool-use conversations; tool sets from public datasets; user sim seeded from `Nemotron-Personas-USA`; EnvCommons pivots → ~1.2M next-action decision points |
| `nvidia/Nemotron-SFT-OpenCode-v1` | 459K samples (general 90K, bash_only_tool 97K, bash_only_tool_skills 96K, question_tool 76K, agent_skills 67K, agent_skills_question_tool 33K) | permissive | **OpenCode CLI** agent-skill subsets — closest NVIDIA analog to the greghavens traces |
| `nvidia/Nemotron-Cascade-RL-SWE` (+ `-SFT-SWE`) | RL + SFT SWE data from SWE-Bench-Train/reBench/SWE-Smith/R2E-Gym/SWE-Fixer-Train | permissive | Agentless (mini) prompt framing, 16K–32K prompt lengths; contamination-filtered against SWE-bench Verified repos; DeepSeek-R1 responses; 37.2/43.1 pass@1 for 8B/14B |
| `nvidia/SWE-Hero-openhands-trajectories` | 34,269 trajectories / 11,766 issues, Qwen3-Coder-480B-A35B | permissive (MIT/Apache/BSD) | OpenHands execution-based FT; note sibling `nebius/SWE-rebench-openhands-trajectories` (67,074 trajs) |

### 3.2 Math / reasoning / reward / pretraining / personas

| Dataset | Size | Notes |
|---|---|---|
| `nvidia/OpenCodeReasoning` (v1) | 735K Python samples / ~28K competitive-programming questions (DeepSeek-R1 distilled) | CC-BY-4.0 |
| `nvidia/OpenCodeReasoning-2` | **2.5M samples**: 1.4M Python + 1.1M C++, ~34-35K unique questions, 10 platforms | CC-BY-4.0; question-solution-**critique** triples, reasoning CoTs, execution pass rates; 49.4 GB; biggest reasoning dataset of its kind |
| `Nemotron-Cascade-*` | RL/SFT math + code "Cascade" family (14-repo collection) | permissive; multi-stage RL-to-SFT curricula |
| Nemotron collection (50-repo SFT set for Nano/Super/Ultra v3) | chat-instruction, code-swe, math-reasoning, terminal, reward-modeling, pre-training, rag, personas | permissive | Used to post-train Nemotron v3; terminal subset is our **terminal-bench-style byte traffic** |

The SFT/RL/agentic NVIDIA stack is effectively an **open re-implementation of the OpenAI
"RL with code repair reward models" recipe** — high value as Tier 2 SFT with execution-graded
labels (resolution success, test pass/fail), which we can keep byte-native in the JSONL
transcript itself.

### 3.3 Physical-AI / omni-dreams world simulation (Tier 1)

| Dataset | Size | Notes |
|---|---|---|
| `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` | per-scene **3D Gaussian Splatting** reconstructions: `usdz` scenes + OpenDRIVE `xodr` maps + surface meshes + 6-camera first frames; v26.04 → 1,607 clips; ~20 s/scene | AV-commercial/AV-research use only |
| `nvidia/omni-dreams-samples`, `nvidia/omni-dreams-scenes` | curated driving sequences (hdmap.mp4, first_frame.png, prompt.txt) + WebRTC scene USDZs | sample/scene staging for OmniDreams |

**OmniDreams** (arXiv:2606.03159, tech report) is the anchor result: an action-conditioned
**autoregressive diffusion world model** (Cosmos-Predict2.5-based, 2B) generating photoreal
multi-camera video in real time (68 FPS 720p single-view on 1× GB300; 105 FPS multi-view on
16 GB300). Training data: RDS mid-train **16,600 h / 3M 20-s clips** (7 synchronized cams, 15
countries) + RDS-HQ-1M post-train **1.14M clips / 4,944 h** (≈21k h combined). A
**World-Action Model** post-trained from it beats the ~10B VLA Alpamayo 1.5 on the NuRec
closed-loop protocol with ~1/5 the params: **collision 6.9% → 4.2%** (front 1.0→0.9, lateral
0.6→0.4, rear 5.3→3.0). Implication for x8D: byte-native diffusion over pixel/PCM byte streams
is a *principled* path to world-model behavior — the data is just bytes at ids 0-255.

---

## 4. Indic multilingual (sarvamai + ai4bharat)

### 4.1 sarvamai (verified)

| Dataset | Size | Notes |
|---|---|---|
| `sarvamai/indivibe` | **22 scheduled languages × 2 scripts (native + romanized) × 110 prompts = 4,840**; subsets chat/code/math/stem | New Indic eval, LLM-as-judge pairwise; romanized Latin = the colloquial byte distribution most tokenizers mangle |
| `sarvamai/indic-diarbench` | ~108 h, 22 languages; near-field + far-field + in-the-wild | Joint **diarization + ASR** benchmark; Sarvam pipeline lowest DER 16.0% / cpWER 38.8% |
| `sarvamai/mmlu-indic`, `arc-challenge-indic`, `boolq-indic`, `gsm8k-indic`, `trivia-qa-indic(-mcq)`, `samvaad-hi-v1`, `tatoeba-indic` | translated/transliterated benchmark mirrors | romanized configs included (e.g. `hi_roman`) |
| `sarvamai/audiollm-evals`, `tts-general-benchmark` (1,815 prompts, 11 langs, HQ+8 kHz telephony), `tts-robustness-benchmark`, `olmOCR-Bench-English`, `sarvam-dub-benchmark-set` | audio/OCR/dub evals | eval-only, but reusable as byte-format reference data |
| **Sarvam-1 / Sarvam-2T corpus** | 2B model; **custom tokenizer, vocab 68,096, fertility 1.4–2.1 (avg ~2.08) vs Llama-3.1 9.34** | ~2T-token Indic corpus (Sarvam-2T), +2T English ≈ 4T total; IndicGenBench Flores 39.83 chrF++ |
| `sarvamai/sarvam-30b`, `sarvamai/sarvam-105b` | MoE models (2.4B active / 105B); MILU 76.8 / SOTA 22-lang | post-2024 lineup; benchmark cards only, not training data |

### 4.2 ai4bharat (verified)

| Dataset | Size | Notes |
|---|---|---|
| `ai4bharat/sangraha` | **251.3B tokens across 22 languages** (Verified 64.3B + Synthetic 162.7B + Unverified 24.3B) | Largest cleaned Indic pretraining corpus; per-lang table (hin 34.5B, tam 17.4B, guj 17.2B, …); synthetic = English→14-lang translation + romanization |
| `ai4bharat/indic-align` | **IndicAlign-Instruct 74.7M prompt-response pairs** (14 langs; ShareLlama/Dolly-T/OpenAssistant-T/WikiHow/IndoWordNet/Anudesh/Wiki-Conv/Wiki-Chat) + **IndicAlign-Toxic 123K** toxic-prompt/non-toxic-response pairs | IFT + safety alignment |
| `IndicLLMSuite` + **Setu** (Spark pipeline) | cleaning/filtering/toxicity/dedup code | blueprint to reuse for any byte-corpus pipeline |

**Why this is uniquely our advantage.** Sarvam-1 had to *build a custom 68K-token tokenizer*
and still reports fertility ~2 for Indic scripts; Llama-3.1 needs 8–17 tokens per Indic word
(Bengali 8.02, Tamil 12.39, Kannada 14.95, Oriya 16.84). Our byte law skips tokenization
entirely: every Indic script is already native UTF-8 bytes at ids 0–255, romanized and native
script are both in-vocabulary, and byte-fertility is definitionally 1 byte/codepoint. The
Indic corpora give us maximal **byte-level script diversity** (UTF-8 multi-byte sequences,
Devanagari conjuncts, abugida shaping) for free, with no vocab engineering — the counterpoint
to Sarvam's tokenizer work is our `byte_tokenizer.py`.

---

## 5. DiffusionGemma & diffusion-over-bytes (verified, model card + vLLM/SGLang/NVIDIA NIM)

**`google/diffusiongemma-26B-A4B-it` — Apache 2.0, Google DeepMind, released 2026-06-10.** The
definitive proof that **language is also diffusion**. Key spec (official model card):

| Item | Value |
|---|---|
| Params | 25.2B total / 3.8B active (MoE) — initial reports said "26B/4B"; model card is authoritative |
| Layers / experts | 30 layers; 128 experts + 1 shared, 8 active; sliding window 1024; ctx up to 256K |
| Canvas | **256 tokens/block**, block-autoregressive (commit denoised canvas → KV cache → next canvas) |
| Sampler | **entropy_bound**, `diffusion_entropy_bound=0.1`; max 48 denoising steps; temp linear 0.8→0.4; adaptive stop (avg entropy < 0.005 **and** argmax stable 2 consecutive steps) |
| Noise | **uniform-state** (random tokens) + full re-noising of non-selected positions |
| Speed | >1,100 tok/s low-batch on H100 FP8; ~700+ on RTX 5090; vLLM (≥0.24.0) + SGLang `Gemma4Renoise`; self-conditioning buffer = `max_seqs × canvas × vocab` |
| Modalities | text, image, video → text; vision encoder ~550M |

Mapping to x8D (re-affirms `AGENTS.md` issues #2/#5/#6 and `DiffusionGemma.md` §7/§9):

- **Vocab 264 vs 262K.** DiffusionGemma's self-conditioning tensor is
  `max_seqs × 256 × 262144`. With vocab 264 ours is ~**1600× smaller** — self-conditioning is
  uniquely cheap for a byte model.
- **Uniform-state byte diffusion.** Re-noise with a random byte 0–255; MASK=256 stays only as
  interface protocol. Trivially natural over 256 states (see `DiffusionGemma.md` §7.1).
- **Entropy-bound acceptance over bytes** = the DSpark confidence-head idea at inference time:
  accept low-entropy positions within budget, re-mask + regenerate the rest
  (`x8d_spec_decode.py` `_block_surrogate` → real confidence head when torch lands).
- **Canvas = 256 bytes** (our `canvas_length=256`, `diffusion_entropy_bound=0.1` already in the
  repo `generation_config.json`). Block-autoregressive canvas commit is exactly our
  `mask_canvas`/`renoise_to_random_bytes` loop.
- **MTP linkage** (`Depth-Context-Attention-Frameworks-2026.md` §4): DiffusionGemma's "15-20
  tokens per forward pass" is a *whole-canvas* draft; our MTP-style confidence head drafts and
  verifies 8×8 blocks. Same family.

---

## 6. What x8D-Omni-Diffusion optimizes (decision)

**(a) Training data mix (Tier mapping per `Training-Dataset-and-Quantization-Plan.md`):**
- **Tier 2 SFT** ← agentic/computer-use/tool-use traces byte-encoded as
  `[BOS] <issue/prompt bytes> <thinking> <tool-call bytes> [IMG_START] <screen pixels> [IMG_END] <result> <next action> [EOS]`.
  Highest value: `greghavens/gpt-5.6-sol-*` + `nvidia/Nemotron-SFT-OpenCode-v1` + a single
  verifier-backed Fable-5 source. Reserve `nvidia/Open-SWE-Traces` thinking/non-thinking
  dual-mode and OpenCodeReasoning-2 for code + critique + reward-modeling SFT.
- **Tier 2 multilingual** ← Sangraha (Verified subset first), IndicAlign-Instruct/Toxic for
  script diversity + safety; Sarvam corpora are license-shared but not cleanly downloadable
  for pretraining (evals usable as byte-format reference).
- **Tier 1** ← NuRec 3DGS scenes + omni-dreams samples as **pixel/PCM byte streams** at ids
  0-255 (no vision/audio encoder), with `[IMG_START]`/`[AUD_START]` markup; world-model
  (OmniDreams-style) behavior via masked byte diffusion.
- **Tier 0** unchanged: FineWeb/Pile/RedPajama UTF-8 bytes.

**(b) Serving (byte-native, no tokenizer):**
- Import every corpus through `omni_diffusion/x8d_dataset.py` (#25): datasets-server API →
  flat bytes → `X8DDS` stream → 8×8 DSpark block compression → `<name>.x8dds.gguf` +
  `manifest.json`, staged to the HF model repo (`hf upload bapX/x8D-Omni-Diffusion ./staged_dir/ .`).
- **Pointer-map mmap extends to data**: with shards stored as mmap'd `.x8dds.gguf`, a training
  step or eval loads only the needed byte span — the same Range-fetch + `/0.001` reversal law
  that serves K3 weights (`x8d_hf.py`, `moe_disk.py`, `Kimi-K3-x8D-Pointer-Quantization.md`).
- Dedup + permissive-license filter runs **inside** the import pipeline (Fable-5 mirrors,
  Crownelius-style contamination).

**(c) Sampler (diffusion, not AR):**
- Adopt DiffusionGemma's `entropy_bound` sampler verbatim at byte scale: canvas 256, budget
  0.1, temp 0.8→0.4, adaptive stop (entropy<0.005 + 2-step stability), uniform-state byte
  re-noise, self-conditioning. Wire into `generation_utils.py` `_sample()` (#5/#6).
- Keep block-autoregressive canvas commit + KV-cache reuse for long byte streams; use the
  8×8 DSpark confidence head as the byte-domain MTP drafter (#4/#7).

**(d) Guardrails:** dedup across Fable-5 mirrors and any future Opus-5 releases; verify
licenses (AGPL-3.0 CoT floor, CC-BY-4.0 for AlinCiocan/greghavens, permissive for NVIDIA);
skip Terra (no data) and Luna (persona-polluted distill).

---

## 7. Decision / priority table

Priority: P0 = train on it now, P1 = strong value / next, P2 = evaluate-only or license-gated.

| Dataset | Tier | Modality bytes | Priority | License / gate |
|---|---|---|---|---|
| `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | 2 (SFT) | UTF-8 JSONL | **P0** | CC-BY-4.0 |
| `nvidia/Open-SWE-Traces` | 2 (SFT) | UTF-8 JSONL | **P0** | permissive repos |
| `nvidia/Nemotron-SFT-OpenCode-v1` | 2 (SFT) | UTF-8 JSONL | **P0** | permissive |
| `nvidia/OpenCodeReasoning-2` | 2 (SFT) | UTF-8 code bytes | **P0** | CC-BY-4.0 |
| `greghavens/fable-5-coding-and-debugging-traces` (only) | 2 (SFT) | UTF-8 JSONL | P1 | CC-BY-4.0; dedup against mirrors |
| `AlinCiocan/fable-5-claude-code-traces` | 2 (SFT) | UTF-8 JSONL | P1 | CC-BY-4.0; revision `v1.0-full-scrubbed` |
| `ApertureQA/Glint-Research/Fable-5-traces` | 2 (SFT) | UTF-8 JSONL | P1 | AGPL-3.0 (CoT); use in place of Poumrm/Quaxicron |
| `nvidia/Nemotron-Agentic-v1` | 2 (SFT) | UTF-8 JSONL | P1 | CC-BY-4.0 |
| `nvidia/Nemotron-Cascade-{SFT,RL}-SWE` | 2 (RL/SFT) | UTF-8 JSONL | P1 | permissive |
| `nvidia/OpenCodeReasoning` (v1) | 2 (SFT) | UTF-8 code bytes | P1 | CC-BY-4.0 |
| `ai4bharat/sangraha` (Verified first) | 0 (pretrain) | UTF-8 bytes | **P0** | permissive (IndicLLMSuite) |
| `ai4bharat/indic-align` (Instruct + Toxic) | 2 (SFT/safety) | UTF-8 bytes | P1 | permissive |
| `sarvamai/indivibe` + eval mirrors | 2 (eval) | UTF-8 native+romanized | P2 | eval-only, byte-format reference |
| `sarvamai/indic-diarbench` + `audiollm-evals` + TTS benches | 1/2 (eval) | PCM + text | P2 | eval-only |
| `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` | 1 (world-sim) | 3DGS/pixel bytes | P1 | AV-use license |
| `nvidia/omni-dreams-{samples,scenes}` | 1 (world-sim) | video/pixel bytes | P1 | gated HF |
| FineWeb / The Pile / RedPajama (from Tier plan) | 0 (pretrain) | UTF-8 bytes | P0 | MIT/CC |

Unverified / not found: **Claude Opus 5 trace dataset** (only Opus 4.5/4.6/4.7 traces exist);
**Sarvam-2T direct HF download** (corpus described in blog; no public card found). Fable-5
"Claude Mythos" naming resolved: Fable 5 = Mythos 5 with classifiers (Anthropic 2026-06-09).
DiffusionGemma parameter counts corrected from the earlier 26B/4B estimate to the official
25.2B/3.8B.

---

## 8. Newly-requested corpora (2026-07-31 expansion, issue #33)

Status: verified 2026-07-31 against the HF datasets-server API (`/splits` + `/size`
endpoints) and the NVIDIA collection API (`/api/collections/...`). Row counts are
from the live datasets-server, which supersedes any older card numbers in §2–§4.
Every corpus below imports through `omni_diffusion/x8d_dataset.py` — datasets-server
API → flat raw bytes (UTF-8 / pixel / PCM / little-endian numerics) → `X8DDS` stream
→ 8×8 DSpark block compression → `<name>.x8dds.gguf` + `manifest.json`, **lossless
roundtrip, threshold 0.001** — then stages to the HF model repo
(`hf upload bapX/x8D-Omni-Diffusion ./staged_dir/ .`).

### 8.1 NVIDIA Nemotron collections (official, verified)

NVIDIA's post-training data is now organized as one collection per capability
(collection URLs below; some repos are gated and need HF login before datasets-server
serves them — mark those `gated`).

| Collection (URL) | Key datasets (verified rows) | Modality | Notes |
|---|---|---|---|
| [nemotron-agentic-and-tool-use](https://huggingface.co/collections/nvidia/nemotron-agentic-and-tool-use) | `Nemotron-SFT-Agentic-v2`, `Nemotron-Agentic-v1` (335,122), `Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` (4,120), `Nemotron-RL-Agentic-Function-Calling-Pivot-v1` (1,190), `Nemotron-RL-Agentic-SWE-Pivot-v1` (3,660), `Nemotron-RL-agent-calendar_scheduling` (4,010), `Nemotron-RL-agent-workplace_assistant` (1,840), `Nemotron-SFT-ARC-AGI-v1` (2,785), `Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` (964) | UTF-8 JSONL tool calls | function-calling + multi-step agent pivots; the RL pivots are our tool-call policy SFT byte traffic |
| [nemotron-chat-and-instruction-following](https://huggingface.co/collections/nvidia/nemotron-chat-and-instruction-following) | `Nemotron-SFT-Instruction-Following-Chat-v3` (682,491), `Nemotron-SFT-Multilingual-v2` (370,081), `Nemotron-RL-instruction_following-structured_outputs` (9,949), `Nemotron-RL-Identity-Following-v1` (21,660), `Nemotron-RL-Instruction-Following-MultiTurnChat-v1` (2,011), `Nemotron-RL-Instruction-Following-Calendar-v2` (9,915), `Nemotron-RL-CFBench-v1` (1,121), `Nemotron-RL-Multichallenge-v1` (2,118), `Nemotron-RL-InverseIFEval-v1` (1,000), HelpSteer3 (132,937) | UTF-8 chat bytes | v3 = the current SFT tier; HelpSteer3 also appears under reward-modeling |
| [nemotron-code-and-swe](https://huggingface.co/collections/nvidia/nemotron-code-and-swe) | `Nemotron-SFT-OpenCode-v1` (459K), `Nemotron-SFT-SWE-v3` (3,526), `Nemotron-SFT-SWE-v2`, `Nemotron-SWE-v1` (1,391), `Nemotron-SFT-Competitive-Programming-v2`, `Nemotron-RL-coding-competitive_coding` (1,463), `Nemotron-Cascade-RL-SWE` / `-SFT-SWE`, `Nemotron-SFT-CUDA-v1` (566), pretraining: `Nemotron-Pretraining-Code-v1` (935M), `-v2` (836M), `Nemotron-CC-Code-v1` (216M, gated) | UTF-8 code bytes | the whole code/SWE ladder, SFT→RL→pretraining |
| [nemotron-math-and-reasoning](https://huggingface.co/collections/nvidia/nemotron-math-and-reasoning) | `Nemotron-Math-v2` (7,085,839), `Nemotron-Math-Proofs-v1` (924,942) / `-v2` (55,577), `Nemotron-PrismMath` (1,002,595), `Nemotron-Research-GooseReason-0.7M` (673,125), `AceReason-1.1-SFT` (3,958,018), `AceReason-Math` (49,585), `OpenMathReasoning` (5,678,317), `OpenMathInstruct-1` (6,078,712), `OpenMathInstruct-2` (21,972,791), `Nemotron-CC-Math-v1` (189M, gated-auto), `Nemotron-Cascade-RL-Math` (14,476), `Nemotron-RL-Math-v2` (7,732) | UTF-8 math bytes | OpenMathReasoning = AIMO-2 winning-recipe SFT; Nemotron-CC-Math paper arXiv:2508.15096 |
| [nemotron-terminal](https://huggingface.co/collections/nvidia/nemotron-terminal) | `Nemotron-Terminal-Corpus` (366,154 across 4 configs: `dataset_adapters` 226,313, `skill_based_easy` 44,809, `skill_based_medium` 89,343, `skill_based_mixed` 5,689), `Nemotron-Terminal-Synthetic-Tasks`, models `Nemotron-Terminal-8B/14B/32B` | UTF-8 terminal sessions | **terminal-bench-style byte traffic** — command output + error + retry loops; imports with `--config dataset_adapters` etc. |
| [nemotron-reward-modeling](https://huggingface.co/collections/nvidia/nemotron-reward-modeling) | HelpSteer (37,120), HelpSteer2 (21,362), HelpSteer3 (132,937), `Nemotron-RLHF-GenRM-v1` (199,267), `Nemotron-Cascade-RM-Training` (81,808), `Nemotron-Cascade-RL-RLHF` (45,882) | UTF-8 preference bytes | human + generative reward-modeling signal; GenRM = Sol-style model-judge signal in NVIDIA form |
| [nemotron-pre-training-datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) | `Nemotron-CC-v2` (8,793,738,251), `Nemotron-CC-v2.1` (3,800,016,491, gated-manual), `Nemotron-Pretraining-Specialized-v1.2` (599,514,257), `Nemotron-Pretraining-Code-v3` (146,323,609), `Nemotron-Pretraining-SFT-v1` (299,245,017), `Nemotron-Pretraining-Legal-v1` (9,616,568), `Nemotron-Pretraining-Dataset-sample` (27,706) | UTF-8 bytes | Tier-0-scale common-crawl dedup; most `gated` (HF login) |
| [nemotron-rag](https://huggingface.co/collections/nvidia/nemotron-rag) | models: `llama-nemotron-embed-vl-1b-v2`, `llama-nemotron-rerank-vl-1b-v2`, `omni-embed-nemotron-3b`, `NVIDIA-Nemotron-Parse-v1.2` | embedding/rerank models, not SFT data | reference for RAG evals; no training-bytes value on its own |
| [nemotron-personas](https://huggingface.co/collections/nvidia/nemotron-personas) | `Nemotron-Personas-USA` (1,000,000 rows / 6M personas), `-India` (3,000,000 / 21M; Hindi Devanagari + Latin + Indian English), `-Japan` (1,000,000), `-Korea` (1,000,000 / 7M), `-Brazil` (1,000,000), `-France` (1,000,000), `-Singapore` (148,000), `-El-Salvador` (148,000), `-Vietnam` (100,000), `-Belgium` (1,200,000) | UTF-8 persona bytes | synthetic persona *seeds* for the tool-use/user-sim datasets; `-India` is byte-script-diverse (see §4.2's byte-advantage argument) |
| [opencodereasoning](https://huggingface.co/collections/nvidia/opencodereasoning) + [opencodereasoning-ii](https://huggingface.co/collections/nvidia/opencodereasoning-ii) | `nvidia/OpenCodeReasoning` (752,713; paper arXiv:2504.01943), `nvidia/OpenCodeReasoning-2` (2,164,812 total: `python` ≈1,422,489 + `cpp` ≈742,323; 4.38 GB parquet) | UTF-8 code bytes | question–solution–**critique** triples + execution pass rates; v2 is the biggest reasoning dataset of its kind |
| [openmath](https://huggingface.co/collections/nvidia/openmath) | `OpenMathInstruct-1` (6,078,712), `OpenMathInstruct-2` (21,972,791), `OpenMathReasoning` (5,678,317), `OpenMath-GSM8K-masked` (7,473), `OpenMath-MATH-masked` (7,500) | UTF-8 math bytes | the OpenMath family of instruction-tuning + RL-ready math |
| [openmathreasoning](https://huggingface.co/collections/nvidia/openmathreasoning) | `OpenMathReasoning` (5,678,317) + fine-tuned `OpenMath-Nemotron-*` models | UTF-8 math bytes | AIMO-2 winning solution dataset (paper arXiv:2504.16891) |
| [physical-ai](https://huggingface.co/collections/nvidia/physical-ai) | `PhysicalAI-Autonomous-Vehicles-NuRec`, `PhysicalAI-Autonomous-Vehicles` (gated-auto), `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` (1,121,947), `PhysicalAI-Robotics-GR00T-Teleop-Sim` (5,820,277), `PhysicalAI-Robotics-GR00T-Teleop-GR1` (7,553,609), `PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes` (156,461,194), `PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes`, `-Autonomous-Driving-Scenarios`, `-Warehouse-Operations-Scenes`, `PhysicalAI-Robotics-Locomanipulation-GRAIL` (2,002), `PhysicalAI-VANTAGE-Bench` (3,276), `PhysicalAI-GR00T-Tuned-Tasks` (532,206), `LIBERO_LeRobot_v3` (848,441), `GR00T-N1.7-AppleToPlate` (171,625) | pixel/PCM/3DGS/action bytes | Tier-1 world-sim/3D; NuRec is 3DGS (`usdz`+`xodr`); GR00T teleop = video+action streams, all bytes at ids 0–255 |
| [nvidia-omnidreams](https://huggingface.co/collections/nvidia/nvidia-omnidreams) | `nvidia/omni-dreams-samples` (66 rows, gated-auto), `nvidia/omni-dreams-scenes` (gated-auto), `nvidia/omni-dreams-models` (gated-auto, image-to-video) | video/pixel bytes | OmniDreams staging data (arXiv:2606.03159, see §3.3) |

`nvidia/Open-SWE-Traces` (already §3.1) confirmed live: **4 splits** =
`{openhands, sweagent} × {minimax_m25, qwen35_122b}` — import one agent family at a
time to control dedup/contamination; the 207,489-trajectory count stands.

### 8.2 sarvamai (expanded, all verified via datasets-server)

| Dataset | Verified rows / configs | Notes |
|---|---|---|
| `sarvamai/indic-diarbench` | **22 language configs** (`Assamese`, `Bengali`, `Bodo`, `Dogri`, `Gujarati`, `Hindi`, `Kannada`, `Kashmiri`, `Konkani`, `Maithili`, `Malayalam`, `Manipuri`, `Marathi`, `Nepali`, `Odia`, `Punjabi`, `Sanskrit`, `Santali`, `Sindhi`, `Tamil`, `Telugu`, `Urdu`), each a single `test` split; 1,164 rows / 12.06 GB total | **config = language**: `--dataset sarvamai/indic-diarbench --config Assamese` etc.; Assamese 27 / Bengali 104 / Bodo 29 rows; fields `audio`, `annotated_transcript` (speaker_id/transcript/start_time/end_time), `num_speakers`, `duration_seconds` — audio→PCM bytes + transcript→UTF-8 bytes (§4.1 diarization eval) |
| `sarvamai/samvaad-hi-v1` | 101,476 rows, default config, `train` split, 202 MB | Hindi conversation/QA dataset |
| `sarvamai/mmlu-indic` | 296,318 rows across 22 configs (`bn, en, gu, hi, kn, ml, mr, or, pa, ta, te` + `_roman`), `test`+`validation` | §4.1 mirror; native + romanized configs, 14,042 rows/language test |
| `sarvamai/boolq-indic` | default config, `train` + `validation` | §4.1 mirror |
| `sarvamai/arc-challenge-indic` | 22 configs (`bn, en, gu, hi, kn, ml, mr, or, pa, ta, te` + `_roman`), `test`+`validation` | §4.1 mirror |
| `sarvamai/gsm8k-indic` | 21 configs (`bn, en, gu, hi, kn, ml, mr, or, pa, ta, te` + `_roman`), `test` | §4.1 mirror |
| `sarvamai/trivia-qa-indic-mcq` | 11 configs (`bn, en, gu, hi, kn, ml, mr, or, pa, ta, te`), `validation` | §4.1 mirror |
| `sarvamai/tatoeba-indic` | 32 configs (`asm, awa, ben, bho, brx, guj, hin, kan, kha, kok, lah, mai, mal, mar, mni, nep, ori, pan, pli, san, sat, snd, tam, tel, urd` + dev splits), `test`+`dev` | translation-sentence pairs |
| `sarvamai/indivibe` | 4,840 rows (`chat` 2,200, `code` 880, `math` 880, `stem` 880), `test` | §4.1 new-Indic eval; LLM-as-judge pairwise |
| `sarvamai/audiollm-evals` | 980 rows, default config, `train`, 213 MB | audio-LLM eval set (PCM bytes) |
| `sarvamai/tts-general-benchmark` | 1,815 rows, default config, `train` | TTS eval, 11 langs, HQ + 8 kHz telephony |
| `sarvamai/tts-robustness-benchmark` | 959 rows, default config, `train` | TTS robustness eval |
| `sarvamai/olmOCR-Bench-English` | 1,258 rows, default config, `train`, 307 MB | OCR bench (image→text bytes) |
| `sarvamai/sarvam-dub-benchmark-set` | 704 rows, default config, `train`, 83 MB | dubbing eval (audio+text) |

All are `sarvamai/*` on HF — datasets-server `/rows` + `/parquet` endpoints serve
them without the `datasets`/torch dependency, so `x8d_dataset.py` flattens every
field (audio refs `audio[0].src/type`, UTF-8 transcripts, float timing) to raw bytes.

### 8.3 ai4bharat (expanded, verified via HF API)

| Dataset | Verified | Notes |
|---|---|---|
| `ai4bharat/sangraha` | 251B tokens / 22 langs (CC-BY-4.0) | §4.2 pretraining tier; per-lang table (hin 34.5B, tam 17.4B, guj 17.2B, …) |
| `ai4bharat/indic-align` | Instruct 74.7M + Toxic 123K (14 langs, CC-BY-4.0) | §4.2 IFT + safety alignment |
| `ai4bharat/IndicCorpV2` | monolingual pretraining corpora (ACL 2023) | IndicCorp v2 per-language corpora |
| `ai4bharat/samanantar` | 49.6M En→Indic sentence pairs | §4.2 parallel corpus |
| `ai4bharat/Aksharantar` | 26M transliteration pairs, 20 langs | script diversity (native ↔ romanized) |
| `ai4bharat/IndicVoices` | 23.7K h audio / 51K speakers / 22 langs; 11,200 h transcribed | ASR bytes |
| `ai4bharat/indicvoices_r` | 1,704 h TTS / 10,496 speakers / 22 langs | TTS bytes |
| `ai4bharat/Rasa` | expressive TTS, ≥20 h/speaker | TTS bytes |
| `ai4bharat/Kathbath` | 1,684 h labelled ASR, 12 langs | ASR bytes |
| `ai4bharat/Shrutilipi` | 6,400 h ASR from AIR news, 12 langs | ASR bytes |
| `ai4bharat/SeamlessAlign` + `NPTEL` | BhasaAnuvaad subsets (44,400 h speech translation, 13 langs) | AST bytes |
| `ai4bharat/IndicContextEval` | 16,884 utterances / 55.93 h / 8 langs (arXiv:2606.19157) | audio-LLM context eval |
| `ai4bharat/MILU` | 11 langs, 8 domains, 41 subjects (arXiv:2411.02538) | text eval (usable as byte-format reference) |
| `ai4bharat/Svarah` | 9.6 h Indic-accented English ASR | §eval |
| `ai4bharat/naamapadam` | NER, 11 langs (CC0) | §token eval |
| `ai4bharat/IndicQuestionGeneration` (98K/lang), `IndicSentenceSummarization` (431K), `IndicHeadlineGeneration` (1.43M), `IndicParaphrase` (5.57M), `IndicWikiBio` (57K) | IndicNLG suite (arXiv:2203.05437) | NLG tasks, most CC-BY-NC (license-gated for training) |
| `ai4bharat/IndicCOPA`, `ai4bharat/IndicQA`, `ai4bharat/IN22-Gen`, `ai4bharat/Bhasha-Abhijnaanam`, `ai4bharat/Rural_Women_ASR_v2`, `ai4bharat/ncert-bench-lite` | benchmark/test sets | eval-only byte-format references |

### 8.4 Frontier trace corpora — computer use / tool use / omni & 3D

Fable 5, GPT-5.6 Sol, and Opus 5 text/tool traces are covered in §2 (dedup rule:
use one verifier-backed source + one canonical session dump; AGPL-3.0 CoT floor,
CC-BY-4.0 for AlinCiocan/greghavens, permissive for NVIDIA). This subsection adds
the adjacent trace families:

- **Computer-use traces.** GPT-5.6 Sol's OSWorld 2.0 62.6% and AutomationBench
  18.1% runs (§2.2) are the signal; the closest extractable bytes are the
  greghavens/Manusagents Sol trajectories (screenshots + `call_…` tool-IDs +
  `[IMG_START]`-wrappable pixel frames). No dedicated verified "computer-use-only"
  frontier trace repo was found in this pass — apply the §2.1 dedup + permissive-
  license gate if one appears (mirrors of mirrors are the norm).
- **Tool-use / agentic traces.** NVIDIA's `Nemotron-Agentic-v1` + the
  `nemotron-agentic-and-tool-use` RL pivots (§8.1) are the open, verified
  equivalent; `Nemotron-SFT-OpenCode-v1` is the closest to greghavens' harness.
- **Omni / world-sim / 3D.** The `physical-ai` + `nvidia-omnidreams` collections
  (§8.1) carry NuRec 3DGS (`usdz`/`xodr`), GR00T teleop video+action, and
  world-model synthetic scenes — all pixel/PCM/3DGS bytes at ids 0–255 for
  Tier-1 world-sim SFT; `PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes`
  alone is 156M rows.

Unverified in this pass (flag before use): a clean **Claude Opus 5** trace release
(still only Opus 4.5/4.6/4.7 on HF), a dedicated frontier **computer-use** trace
repo, and any permissive-licensed **Fable-5/Sol video** corpus.

### 8.5 Import + byte-native notes (uniform)

- Every repo above flows through `tools/import_hf_dataset.py --dataset <id> [--config <cfg>]`
  → `omni_diffusion/x8d_dataset.py` → `.x8dds.gguf`. `--config` is forwarded as the
  datasets-server config name — for `sarvamai/indic-diarbench` it is the language
  (`Assamese`, `Bengali`, `Bodo`, …), for `nvidia/Nemotron-Terminal-Corpus` it is
  the corpus split (`dataset_adapters`, `skill_based_easy`, …).
- Gated repos (most `Nemotron-*` pretraining, `omni-dreams-*`, `PhysicalAI-*` auto-
  gated) need an HF token for datasets-server; offline tests in
  `tests/test_x8d_dataset.py` cover identical code paths with synthetic data.
- Lossless roundtrip is asserted in-repo (MAGIC `X8DDS`, threshold 0.001, 8×8
  DSpark); the pointer-map serving law (`x8d_hf.py`/`moe_disk.py`) extends from
  weights to these shards — a training step loads only the byte span it needs.

---

## 9. Sources

- Anthropic: "Claude Fable 5 and Claude Mythos 5" (2026-06-09); Claude platform docs.
- OpenAI: "GPT-5.6: Frontier intelligence…" + "Previewing GPT-5.6 Sol" (2026-07-09) — eval tables.
- HF datasets: `greghavens/gpt-5.6-sol-coding-and-debugging-traces`,
  `greghavens/fable-5-coding-and-debugging-traces`,
  `ApertureQA/Fable-5-traces`, `AlinCiocan/fable-5-claude-code-traces`,
  `Crownelius/Complete-FABLE.5-traces-2M`, `Manusagents/GPT-5.6-Sol-Luna-Terra-Traces`,
  `nvidia/Open-SWE-Traces` (+ arXiv:2606.16038), `nvidia/Nemotron-Agentic-v1`,
  `nvidia/Nemotron-SFT-OpenCode-v1`, `nvidia/Nemotron-Cascade-{SFT,RL}-SWE`,
  `nvidia/OpenCodeReasoning`, `nvidia/OpenCodeReasoning-2` (+ arXiv:2507.09075),
  `nvidia/SWE-Hero-openhands-trajectories`, `nebius/SWE-rebench-openhands-trajectories`,
  `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, `nvidia/omni-dreams-{samples,scenes}`,
  `sammshen/wildclaw-opus-traces`, `netpreme/coding_agent_traces`.
- NVIDIA OmniDreams: arXiv:2606.03159; github.com/nv-tlabs/omni-dreams; flashdreams docs.
- sarvamai: `indivibe`, `indic-diarbench` (arXiv:2607.23808), `mmlu-indic`, `gsm8k-indic`,
  `arc-challenge-indic`, `trivia-qa-indic`, `audiollm-evals`, `tts-general-benchmark`,
  `sarvam-1` blog (custom 68K tokenizer, fertility 1.4–2.1, Sarvam-2T), `sarvam-30b/105b` cards.
- ai4bharat: `sangraha` (251B tokens), `indic-align`, IndicLLMSuite + Setu (ACL 2024).
- Google DeepMind: DiffusionGemma model card (ai.google.dev), HF `google/diffusiongemma-26B-A4B-it`,
  vLLM blog (2026-06-10), SGLang `Gemma4Renoise` docs, NVIDIA NIM page.
- Repo cross-refs: `research/Training-Dataset-and-Quantization-Plan.md`,
  `research/DiffusionGemma.md`, `research/Frontier-Benchmarks-2026.md`,
  `research/Depth-Context-Attention-Frameworks-2026.md`, `AGENTS.md`,
  `omni_diffusion/x8d_dataset.py` + `tools/import_hf_dataset.py` (#25).
