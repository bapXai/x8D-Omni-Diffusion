# x8D-Omni-Diffusion — Status & Optimization Audit 2026 (issue #33)

Status: audited 2026-07-31 against the working tree, the HF model repo
`bapX/x8D-Omni-Diffusion`, `tools/quantize_kimi_k3.py`,
`research/Depth-Context-Attention-Frameworks-2026.md`, `research/DiffusionGemma.md`,
and the upstream `bapXai/x8Dsub-byte` repo. This doc is the single source of truth
for "what actually exists" vs "what is planned", so downstream agents stop
over-claiming.

---

## 1. TL;DR

x8D has a **complete model architecture + a complete byte-native serving
pipeline, and ZERO trained weights**. Everything in the HF model repo under
`x8d_weights/*.x8dptr.gguf` is a **pointer map** (bytes stay on the upstream
model's disk), not a trained x8D weight file. The model has never been trained;
`torch` is not installed on this Mac; the only `.safetensors` files in the tree
are ~296–500-byte synthetic test fixtures plus one 5.5 MB regenerated reference
shard. Training is blocked on GPU + torch, not on any missing code.

| Claim | Status |
|---|---|
| Architecture (28L Qwen2-style DreamModel, vocab 264) | ✅ complete, in repo |
| Byte tokenizer / serving pipeline / quantizers | ✅ complete, tested |
| SandboxComput.bin venv (lossless on disk) | ✅ complete, benchmarked (#28/#32) |
| Trained x8D weights | ❌ **none exist** |
| `x8d_weights/kimi_k3.x8dptr.gguf` (163.4 MB) | ⚠️ pointer map, **not** weights |
| `x8d_weights/kokoro/ltx2/whisper.x8dptr.gguf` | ⚠️ pointer maps, **not** weights |
| torch on dev machine | ❌ not installed |

---

## 2. What IS trained vs NOT trained

### 2.1 Not trained (everything real)

- **The DreamModel has never been trained.** `omni_diffusion/models/dream/modeling_dream.py`
  (1,136 lines, Qwen2-style) defines the forward pass but holds no learned weights.
  The checkpoints it references (`_CHECKPOINT_FOR_DOC = "Dream-7B"`) are doc strings
  inherited from the upstream HKUNLP DREAM project, not artifacts in this repo.
- **No `.safetensors` weights exist in the repo.** The only `.safetensors` files are
  synthetic test fixtures generated at test time:
  - `tests/_tmp_hfq/model.safetensors` (318 B)
  - `tests/_tmp_ptr/model-00013-of-000096.safetensors` (500 B)
  - `tests/_tmp_x8dhf/shard.safetensors` (296 B)
  - `tests/_tmp_ptr/` `cmp/model-00013-of-000096.safetensors` (5,505,182 B) and
    `model-00013-cmp.safetensors` (5,505,182 B) — a regenerated 3072×1792 U8 expert
    used by `test_pointer_quantize.py::test_hf_vs_compressed_forward_identical`.
- **torch is not installed** on the dev machine, so the training tools
  (`tools/finetune_dream_v4_51_3.py`, `tools/trainer_v4_51_3.py`) are dormant.
  The byte-native core and all 70+ tests are pure stdlib.
- **The `README.md` is a ported upstream readme.** Its "Experimental Results"
  section (visual/speech/qualitative PNGs), the `asset/` media, the evaluation
  commands, and the GLM-4-Voice / image-tokenizer paths describe the HKUNLP DREAM
  base, **not** results produced by this repo. The byte-native framework has
  removed MagViT-v2 (images are raw pixel bytes) and the byte tokenizer replaces
  the BPE-era Qwen2 tokenizer — but no byte-native model has been trained to
  reproduce those benchmark numbers.

### 2.2 Trained / built (the actually-working artifacts)

- **Architecture**: `DreamConfig` (vocab 264, ids 256–263, canvas_length 256,
  `diffusion_sampler="entropy_bound"`, `diffusion_entropy_bound=0.1`,
  `max_denoising_steps=48`, `self_conditioning=True`, `final_logit_softcap=30.0`)
  + `modeling_dream.py` + `generation_utils.py` `_sample()` (entropy-bound hook
  at line 404) + `modeling_sensevoice.py` + `resampler_projector.py`.
- **Byte pipeline**: `byte_tokenizer.py` (vocab 264), `byte_diffusion.py`
  (`ByteDiffusionSampler`, pure stdlib mirror of the DiffusionGemma contract),
  `x8d_export.py` (X8DGGUF1 U8 container, 0.001 law), `x8d_spec_decode.py`
  (DSpark 8×8 spec-decode quantizer + confidence-head stand-in),
  `x8d_subbyte.py` (0.016 bit/weight packed model), `x8d_hf.py` (HF→gguf +
  pointer loader), `moe_disk.py` (mmap on-disk expert serving),
  `x8d_dataset.py` (datasets-server import → `.x8dds.gguf`, #25),
  `x8d_venv.py` (SandboxComput.bin compressed venv, #28).
- **Live tools**: `tools/quantize_kimi_k3.py` (#10), `tools/quantize_hf.py` (#17),
  `tools/import_hf_dataset.py` (#25), `tools/bench_byte_core.py` (#14).
- **Tests**: 70+ stdlib `unittest` tests across 15 files (byte tokenizer, config,
  queries, spec-decode, subbyte, export, x8d_hf, pointer quantize, quantize_hf,
  x8d_dataset, x8d_dataset_canvas, x8d_venv, byte_diffusion, openai_server).

**Bottom line for anyone reading claims**: there is a model *that can run its
forward pass* and a serving stack *that can serve any quantized bytes*, but the
0.001-law "x8D weights" everyone talks about are compressed *pointer* coordinates,
not trained parameters. Nothing has learned.

---

## 3. What IS documented vs NOT documented

### 3.1 Documented (research/ + AGENTS.md)

| Topic | Where |
|---|---|
| Byte law, vocab 264, model-repo rules, git/gh workflow, dependency stance | `AGENTS.md` |
| DiffusionGemma notes + config breakdown + x8D mapping | `research/DiffusionGemma.md`, `research/Config-Mapping-DiffusionGemma-to-x8D.md` |
| Tier 0/1/2 training plan + dataset mix | `research/Training-Dataset-and-Quantization-Plan.md` |
| 2026 dataset/trace landscape (NVIDIA, sarvamai, ai4bharat, Fable-5/Sol/Opus-5) | `research/Omni-Datasets-and-Frontier-Traces-2026.md` |
| Depth/context arch wave (AttnRes, KDA, mHC, CLVR, MTP, Engram) + x8D decision | `research/Depth-Context-Attention-Frameworks-2026.md` |
| Kimi-K3 pointer quantization proof (#10) | `research/Kimi-K3-x8D-Pointer-Quantization.md` |
| Whisper/Kokoro/LTX-2 modality matrix (#11) | `research/Omni-Modality-Stack.md` |
| Byte-core optimization benchmarks (#14, #18-#23) | `research/Byte-Core-Optimizations.md` |
| Frontier benchmark/arch deep-dive (#15) | `research/Frontier-Benchmarks-2026.md` |
| Dep-by-dep audit vs cactus-compute/needle | `research/Needle-Dependency-Audit.md` |
| OpenAI-compatible endpoint probe + active-expert report (#39) | `research/Omni-Endpoint-and-Experts-2026.md` |
| MoE routing + omni any-to-any + diffusion-LM + byte-law justification (#40) | `research/MoE-Omni-Diffusion-Language-Modeling-2026.md` |

### 3.2 NOT documented (gaps this audit surfaced)

- **No "state of training" doc** — until now, nothing stated plainly that the
  model is untrained and the `.x8dptr.gguf` files are pointer maps. This doc fixes it.
- **No runtime training plan with GPU targets/budget.** The Tier plan names the
  data; it does not state the DreamModel forward/backward cost at 28L×3584h, the
  expected memory, or a step schedule. (Prerequisite: torch + GPU.)
- **No release/version plan for the HF model repo** beyond the byte-native rules;
  the `x8d_weights/` directory semantics (pointer maps vs trained weights) were
  implied by `x8d_hf.py` but never written down — see §6.
- **`generation_config.json` in `omni_diffusion/models/dream/` is empty on disk.**
  The byte-native values live in `config_dream_resume.json` + `configuration_dream.py`;
  the HF repo's `generation_config.json` (154 B, `alg="entropy_bound"`, steps 48,
  bound 0.1, canvas 256) is authoritative for distribution.
- **OpenAI-compat server** (`tools/` + `tests/test_openai_server.py`, the
  `:666` endpoint from upstream `bapXai/x8Dsub-byte`) — now documented in
  `research/Omni-Endpoint-and-Experts-2026.md` (#39): probe output,
  SARA/MoEOnDisk active-expert report, 26-test live suite.
- **`x8d_dataset_canvas.py`** (HF repo + `staged_dir/`) has a test but no research
  write-up; it is the canvas-level dataset container sibling of `x8d_dataset.py`.

---

## 4. Kimi-3 compressed-or-not status (the 550:1 claim, precisely)

### 4.1 What `x8d_weights/kimi_k3.x8dptr.gguf` actually is

It is an **X8DPTR01 pointer map**, built by `tools/quantize_kimi_k3.py` with **no
download**: each of Kimi-K3's 497,220 tensors is pin-pointed as
`repo | shard | data_offsets | dtype | shape` inside the upstream
`moonshotai/Kimi-K3` safetensors index (96 shards, 2.78 T params). The weight
**bytes stay on the upstream HF disk**. At query time `moe_disk.py` /
`serve_expert_from_pointer()` Range-fetches (or mmaps, for local shards) only the
specific expert span and applies the live `/0.001` reverse. Verified on a real
expert (layer-12/expert-895 w1, U8 3072×1792): 5,505,024 B fetched, reverse exact,
forward matmul **bit-identical** (`maxdiff=0.0`).

### 4.2 The 550:1 number is a pointer-map compression, not trained weights

`research/Kimi-K3-x8D-Pointer-Quantization.md` computes: under the 0.001 sub-byte
law the full 2.78 T params map to **2.837 GB** (U8 0.008 bit → 2.723 GB, BF16
0.016 bit → 114.4 MB, F32 ~0) — from a 1.56 TB FP16-era reference, that is
**≈550:1 / 99.82%**. This is the theoretical size of the *sub-byte coordinate
space* for every upstream parameter, **not** a 2.837 GB trained x8D weight file
that has ever been materialized. The pointer map itself is 151.8 MB on disk. The
distinction matters: serving a pointer map does **not** require the compressed
weights to exist — it requires the upstream model to keep existing on HF.

### 4.3 What IS trained-adjacent

`nvidia/...` aside, x8D's own "weights" story is still: quantized coordinates of
*other* models (Kimi-K3 #10, any HF model via `quantize_hf.py` #17, Whisper/Kokoro/
LTX-2 via `Omni-Modality-Stack.md` #11). None are x8D-trained checkpoints.

---

## 5. What we can optimize in the model (map onto the existing DreamModel)

`research/Depth-Context-Attention-Frameworks-2026.md` (issue #24) is the map. The
DreamModel is a dense 28-layer Qwen2-style stack (hidden 3584, intermediate 18944,
28 heads, 4 KV heads, 2 global KV heads, sliding window 1024, rope_theta 1e6,
max_position 131072). Every optimization below is a **drop-in residual/sampler
replacement**, byte-law neutral, and targets issue #7's planned
`kda_attention.py`/`dspark_diffusion.py` work.

### 5.1 Depth look-back (residuals) — Block AttnRes

DreamModel suffers the same PreNorm dilution as any 28-layer stack. **Block
AttnRes** (arXiv:2603.15031) replaces fixed residual accumulation with softmax
attention over block summaries: 28 layers → N≈8 blocks, per-layer I/O 5.5d
(vs mHC 34d), ≈ baseline quality with 1.25× compute. Cost: 1 RMSNorm + 1
zero-initialized pseudo-query `w_l` per layer (< 0.01% params — nothing to
quantize). **Phase 2 of issue #7**: keep `h_l = h_{l-1} + f` during phase 1,
swap in Block AttnRes as the residual operator once torch lands.

### 5.2 Context look-back (sequence) — KDA + Gated DeltaNet-2

For 1M-token contexts the dense KV cache explodes. **KDA** (arXiv:2510.26692) is
a recurrent linear memory with a *channel-wise* forget gate `W = W·D_α + β·r⊗κ`
(O(1)/token inference, ~75% KV-cache cut at K3's 3:1 KDA:MLA interleave).
**Gated DeltaNet-2** decouples the scalar write strength into channel-wise erase
`b` + write `w` gates (its most granular form). For the byte denoiser this is the
**context** half of issue #7 — replace or interleave the dense attention layers
with a KDA layer and reuse the existing RoPE/KV plumbing.

### 5.3 Cross-layer routing — CLVR (cheapest test)

**CLVR** routes a lower delta-rule layer's internal **write value** (not its
error — CLER fails) into the shared residual stream via a zero-init projection
`h += P_l·v_l`. Modest gain on DeltaNet/Gated DeltaNet, preserves linear-time
structure, zero-init = safe to drop in on top of a KDA variant. Do this *after*
KDA lands; it is the cheap depth-pathway add-on.

### 5.4 mHC — evaluated, not adopted

DeepSeek's **mHC** (m=4 parallel streams, learned mixing matrices) reaches a
comparable 16L loss (1.747) to Full AttnRes (1.737) at **6× the per-layer I/O**
(34d vs 5.5d for Block AttnRes). It is the fallback if Block AttnRes underperforms
in the byte regime, not the default.

### 5.5 Speculative decoding — MTP-style confidence head + DiffusionGemma sampler

**MTP** (DeepSeek V4) turns the model into its own drafter: independent prediction
modules at the stack tail draft K future positions for one-pass verification.
`x8d_spec_decode.py`'s `_block_surrogate` (sha256-of-8×8-block) is the placeholder
confidence head; replace it with a real MTP-style head when torch lands. The
**entropy-bound byte sampler** from DiffusionGemma is already mirrored in
`byte_diffusion.py` + `generation_utils.py` `_sample()` — the optimization is to
wire the learned confidence head in and commit canvases block-autoregressively
with KV reuse (`canvas_length=256`, budget 0.1, temp 0.8→0.4, adaptive stop at
avg-entropy < 0.005 + 2-step argmax stability).

### 5.6 Optimizer — Muon

ETH linear-attention sweep (arXiv:2607.07953): **Muon beats AdamW for every
recurrent/linear family** at matched scale (final loss 2.273 KDA+Muon vs 2.433
fastest-pure). Default the Dream training plan (per
`Training-Dataset-and-Quantization-Plan.md`) to Muon, especially for any
KDA-style variant.

### 5.7 Summary table

| Optimization | Axis | Add to DreamModel | I/O / cost | Priority |
|---|---|---|---|---|
| Block AttnRes | depth | residual op (N=8 blocks) | 5.5d/layer | **P0 (#7 p2)** |
| KDA (3:1) + GDN-2 gates | context | attention replacement | O(1)/token | **P0 (#7 p1)** |
| Entropy-bound byte diffusion + MTP head | sampler | `_sample()` + confidence head | — | **P0 (#5/#6)** |
| CLVR | depth | zero-init write-value injection | ~3d | P1 |
| Muon optimizer | training | trainer default | — | P1 |
| mHC | depth | fallback only | 34d | P2 |

---

## 6. HF model repo expertise: pointer maps vs trained weights

The HF model repo `bapX/x8D-Omni-Diffusion` is the **sole distribution channel**
(the old `bapX/x8D-Omni-Diffusion` bucket was deleted). Its byte-native-only rules
are enforced in AGENTS.md. Inside `x8d_weights/` there are **two very different
classes of file** — do not conflate them:

| File | Size | Class | Serves what |
|---|---|---|---|
| `kimi_k3.x8dptr.gguf` | 163,374,871 B | **pointer map** (`X8DPTR01`) | Range-fetch + `/0.001` of the *upstream* Kimi-K3 span; weights stay on `moonshotai/Kimi-K3` |
| `kokoro.x8dptr.gguf` | 171 B | **pointer map** (empty/near-empty) | Kokoro-82M is small enough that a pointer map is trivial; no x8D-trained Kokoro weights exist |
| `ltx2.x8dptr.gguf` | 2,319,390 B | **pointer map** | Pin-points LTX-2 19B tensors on Lightricks' disk for on-demand `/0.001` reversal |
| `whisper.x8dptr.gguf` | 343,642 B | **pointer map** | Pin-points Whisper large-v3 tensors on OpenAI's disk for on-demand `/0.001` reversal |

Key serving law (unchanged): **the compressed state IS the running state** —
weights never load into RAM wholesale; `/0.001` reverses only the specific expert/
tensor span a query needs, live, at query time (`moe_disk.py`, `x8d_hf.py`,
`serve_expert_from_pointer()`).

**When a real trained x8D weight file appears**, it will be an `X8DGGUF1` U8
container (from `x8d_export.py` / `x8d_hf.py convert_shard_to_gguf`) whose payload
bytes are the raw 0–255 quanta and whose size follows the 0.001 law — clearly
distinct from an `X8DPTR01` pointer map, which contains only indices and offsets.
`kimi_k3.x8dptr.gguf` is a pointer map and will stay one.

---

## 7. Language is also diffusion — x8D vs DiffusionGemma (exact deltas)

`google/diffusiongemma-26B-A4B-it` (Apache 2.0, 2026-06-10) is the proof that
diffusion applies to **language**, not just pixels/audio: it denoises a 256-token
canvas with uniform-state noise, an `entropy_bound` sampler, and block-
autoregressive canvas commit, at >1,100 tok/s on H100. x8D mirrors the *entire
contract* over the 264-id byte space — `byte_diffusion.py` `ByteDiffusionSampler`
is the pure-stdlib reference and `config_dream_resume.json` carries
`canvas_length=256`, `diffusion_entropy_bound=0.1`, `max_denoising_steps=48`.

| Axis | DiffusionGemma | x8D (byte-native) | x8D edge |
|---|---|---|---|
| Vocab | 262,144 subword (Gemma 4 tokenizer) | **264** (256 bytes + 8 specials) | Self-conditioning buffer `max_seqs × 256 × vocab` is **~1600× smaller**; no tokenizer, no `vocab.json`/`merges.txt` |
| Text input | tokenize → ids | `list(data_bytes)` → ids 0–255 | No encoding step; UTF-8 bytes are in-vocabulary by construction |
| Noise | uniform random tokens | uniform random **bytes** (`renoise_to_random_bytes`, ids 0–255) | Trivially natural over 256 states; MASK=256 stays only as interface protocol |
| Canvas | 256 tokens, commit → KV cache → next canvas | 256 bytes, block-autoregressive commit (`sample_canvas` / `mask_canvas`) | Identical block loop; KV reuse for long byte streams |
| Sampler | entropy_bound, bound 0.1, temp 0.8→0.4, adaptive stop | entropy-bound acceptance, budget 0.1 (`generation_utils._sample`) | Byte-domain confidence head doubles as the DSpark 8×8 spec-decode verifier |
| Parallel decode | 256 positions per forward | 8×8 DSpark block draft + verify (`x8d_spec_decode.py`) | MTP-style whole-canvas draft; spec-decode quantization and sampling share the confidence-head idea |
| Specials | `boi/eoi/image` near end of 262144 | IMG_START/IMG_END=260/261, AUD_START/AUD_END=262/263 | +audio modality DiffusionGemma lacks; image bytes are the raw 256 states |
| Attention | sliding(6)+full(1) 30L, 8 active experts | dense 28L Qwen2-style (→ KDA + Block AttnRes plan, §5) | Depth/context upgrades map cleanly (§5.1–5.2) |

**What x8D optimizes vs DiffusionGemma**: (1) vocab 264 vs 262K makes
self-conditioning, logit tensors, and the embedding/lm_head uniquely cheap
(embed + `lm_head` are `Linear(hidden, 264)`); (2) removing the tokenizer
eliminates vocabulary-mismatch and makes Indic/multilingual byte diversity free
(see `Omni-Datasets-and-Frontier-Traces-2026.md` §4); (3) the DSpark 8×8
speculative decode quantizes *and* serves from the same confidence-head
abstraction; (4) audio (AUD_START/AUD_END) is native, which DiffusionGemma lacks.

---

## 8. Sources

- Working tree: `omni_diffusion/models/dream/*`, `omni_diffusion/x8d_*.py`,
  `tests/`, `tools/`, `research/`.
- HF model repo `bapX/x8D-Omni-Diffusion` (listed via `hf models list -R`):
  `x8d_weights/{kimi_k3,kokoro,ltx2,whisper}.x8dptr.gguf`, byte-native configs.
- `research/Kimi-K3-x8D-Pointer-Quantization.md` (#10), `tools/quantize_kimi_k3.py`.
- `research/Depth-Context-Attention-Frameworks-2026.md` (#24) + its sources
  (arXiv:2603.15031 AttnRes, arXiv:2510.26692 KDA, arXiv:2607.07953 ETH,
  DeepSeek mHC/Engram/V4).
- `research/DiffusionGemma.md` + `Config-Mapping-DiffusionGemma-to-x8D.md`.
- `research/Byte-Core-Optimizations.md` (#14/#18-#23), `omni_diffusion/x8d_venv.py`
  (#28) + `research/...` benchmark section for #32.
- Upstream `bapXai/x8Dsub-byte` `x8Dquanta/__init__.py` (0.001 law definition).
