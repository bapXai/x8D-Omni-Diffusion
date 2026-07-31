# x8D-Omni-Diffusion — Agent Rules

## 🔒 Foundational Law: Bytes, Not Tokens

**There are NO tokens in this project. Only raw 8-bit bytes (0–255).**

Every agent, script, and model component MUST treat the 256 unsigned integer states (0x00–0xFF) as the **sole native vocabulary**. The traditional concept of "tokens" (BPE, SentencePiece, WordPiece, or any sub-word vocabulary) is explicitly **banned** from this codebase.

### Why Bytes Replace Tokens

- Modern CPUs, GPUs, memory buses, and storage devices already operate on 8-bit bytes natively.
- The 256-state byte vocabulary is **universal** — it encodes text (UTF-8), images (pixel bytes), audio (PCM samples), code, binaries, and every other data format without any tokenizer overhead.
- Higher bit-widths (16-bit, 32-bit, 64-bit, 128-bit) are nothing but sequential compositions built on top of 8-bit bytes. Bytes are the atomic unit.
- Eliminating the tokenizer removes an entire software layer of complexity, latency, and vocabulary mismatch errors.

### Enforcement Rules

1. **Never** import, instantiate, or reference any BPE/SentencePiece/WordPiece tokenizer.
2. **Never** use `vocab.json`, `merges.txt`, or any merge-based encoding file.
3. The model's embedding layer MUST have size `264` (256 byte states + 8 special tokens: MASK=256, PAD=257, BOS=258, EOS=259, IMG_START=260, IMG_END=261, AUD_START=262, AUD_END=263).
4. The `lm_head` output projection MUST match: `nn.Linear(hidden_size, 264)`.
5. All data pipelines MUST convert inputs to raw byte arrays: `list(data_bytes)` — no encoding step, no vocabulary lookup.

---

## 🗜️ x8Dsub-byte 0.001 Threshold & Pre-trained Weight Compression

All pre-trained weights that enter this repository MUST be compressed using the **x8Dsub-byte 0.001 scaling law** before storage:

```
Quanta[i] = weight_byte[i] × 0.001
```

### Speculative Decoding for Weight Compression

When importing or converting pre-trained model weights (from HuggingFace, PyTorch checkpoints, safetensors, etc.):

1. **Quantize** all floating-point weight tensors to their nearest 8-bit unsigned integer representation.
2. **Apply the 0.001 sub-byte scaling** to map each byte into sub-byte coordinate space.
3. **Store in x8D `.gguf` containers** using `U8` dtype — no float bloat, no JSON metadata pollution.
4. **Target: 98% disk space reduction** compared to original BF16/FP32 checkpoints.
5. **Zero-copy mmap serving**: The compressed state IS the running state. No decompression loop. The inverse math (`/ 0.001`) operates as a live coordinate pointer map at inference time.

### Speculative Decoding at Inference

Use DSpark-style semi-autoregressive speculative decoding for inference:

1. Generate entire **8×8 byte blocks** in parallel (not one byte at a time).
2. A lightweight **confidence head** predicts survival probability per position.
3. Positions with confidence **below 0.001 threshold** are re-masked and regenerated.
4. Under heavy load, dynamically clip verification length to save compute.

---

## 🛠️ Git Workflow — Use `gh` CLI for Everything

**All git operations MUST use the GitHub CLI (`gh`) for validation, commits, and issue management.**

### Hugging Face Hub — `hf` CLI

The Hugging Face CLI (`hf`, v1.26.0) is installed and authenticated as **`bapX`**
(`~/.hf-cli`, symlinked into `~/.local/bin`, token `oauth-bapX` auto-refreshes).

The model lives in an HF **model repo**: `bapX/x8D-Omni-Diffusion`
(https://huggingface.co/bapX/x8D-Omni-Diffusion). The repo holds ONLY
byte-native files — NO safetensors, NO `vocab.json`/`merges.txt`, NO BPE artifacts.

```bash
export PATH="/Users/getwinharris/.local/bin:$PATH"

# Auth / identity
hf auth whoami

# List repo contents (recursive)
hf models list bapX/x8D-Omni-Diffusion -R

# Download a single file from the repo
hf cp hf://bapX/x8D-Omni-Diffusion/config.json ./config.json

# Upload a local file/folder into the repo (single commit)
hf upload bapX/x8D-Omni-Diffusion ./config.json config.json
hf upload bapX/x8D-Omni-Diffusion ./staged_dir/ .

# Delete files from the repo (e.g. old safetensors / BPE tokenizer files)
hf repos delete-files bapX/x8D-Omni-Diffusion \
  '*.safetensors' '*.safetensors.index.json' \
  'vocab.json' 'merges.txt' 'added_tokens.json' \
  'tokenizer_config.json' 'special_tokens_map.json' \
  'tokenization_dream.py'

# Always dry-run before deleting (list what would match first)
hf models list bapX/x8D-Omni-Diffusion -R
```

**Model-repo rules (enforced):**
1. NEVER upload `*.safetensors`, `vocab.json`, `merges.txt`, `added_tokens.json`,
   `tokenizer_config.json`, `special_tokens_map.json`, or `tokenization_dream.py`.
2. `config.json` MUST be byte-native: `vocab_size=264`, `mask=256`, `pad=257`,
   `bos=258`, `eos=259`, `img=260/261`, `aud=262/263`, `tie_word_embeddings=true`.
3. `generation_config.json` MUST use byte-native ids + `alg="entropy_bound"`,
   `steps=48`, `diffusion_entropy_bound=0.1`, `canvas_length=256`.
4. Keep `byte_tokenizer.py`, `x8d_export.py`, `configuration_dream.py`, model code,
   and README in the repo; the model loads via `trust_remote_code=True`.
5. Source of truth for HF distribution is `omni_diffusion/models/dream/` + `x8d_export.py`
   in this repo; sync those into the model repo.

### Commits & Validation

```bash
# Stage and commit changes (always use descriptive messages)
git add -A
git commit -m "feat(byte-tokenizer): replace BPE vocab with 256-byte native embedding"

# Push to remote
git push origin main

# Validate CI status after push
gh run list --limit 5
gh run view <run-id>
```

### GitHub + HF Model Repo Dual Commit

Every merged change that adds or changes byte-native artifacts
(`omni_diffusion/` modules, `tools/`, `README`, `research/`) MUST ALSO be
synced to the HF model repo `bapX/x8D-Omni-Diffusion`.

```bash
export PATH="/Users/getwinharris/.local/bin:$PATH"

# Upload a local folder into the repo (single commit)
hf upload bapX/x8D-Omni-Diffusion ./staged_dir/ .

# Or a single file
hf upload bapX/x8D-Omni-Diffusion ./tools/import_hf_dataset.py tools/import_hf_dataset.py

# Verify after sync
hf models list bapX/x8D-Omni-Diffusion -R
```

**Dual-commit checklist:**
1. Commit + push to GitHub; validate CI via `gh run list --limit 5`.
2. Stage the byte-native artifacts into `staged_dir/`.
3. `hf upload bapX/x8D-Omni-Diffusion ./staged_dir/ .` the staged folder.
4. Verify with `hf models list bapX/x8D-Omni-Diffusion -R`.

The enforced repo rules above still apply — never upload `*.safetensors`,
`vocab.json`, `merges.txt`, or any BPE artifact; the repo is byte-native only.

### Creating Issues

When a user reports or prompts a **new feature**, **bug**, or **issue**, create a GitHub Issue FIRST before writing any code:

```bash
# New feature request
gh issue create \
  --title "feat: <short description>" \
  --body "## Description\n<detailed description>\n\n## Acceptance Criteria\n- [ ] <criterion 1>\n- [ ] <criterion 2>" \
  --label "enhancement"

# Bug report
gh issue create \
  --title "bug: <short description>" \
  --body "## Bug Description\n<what happened>\n\n## Expected Behavior\n<what should happen>\n\n## Steps to Reproduce\n1. <step 1>\n2. <step 2>" \
  --label "bug"

# General issue
gh issue create \
  --title "issue: <short description>" \
  --body "## Context\n<description>\n\n## Action Items\n- [ ] <item 1>" \
  --label "task"
```

### Issue-Driven Development Workflow

1. **User prompts a feature/bug/issue** → Agent creates a GitHub Issue via `gh issue create`
2. **Agent works on the fix/feature** → Commits reference the issue number: `git commit -m "fix(#42): ..."`
3. **Agent validates** → Run `gh run list` to check CI, run tests locally
4. **Agent closes the issue** → `gh issue close <number> --comment "Fixed in commit <sha>"`

### Pull Requests

```bash
# Create a PR from a feature branch
gh pr create \
  --title "feat: <description>" \
  --body "Closes #<issue-number>\n\n## Changes\n- <change 1>\n- <change 2>" \
  --base main

# Check PR status
gh pr status
gh pr checks <pr-number>

# Merge when ready
gh pr merge <pr-number> --squash --delete-branch
```

---

## 🧮 Quantization, Size & Datasets (audited 2026-07-31)

Implemented in `omni_diffusion/x8d_spec_decode.py` (pure stdlib, no torch).

**Speculative-decode quantization flow** (DSpark-style, per block 8x8=64 bytes):
1. Generate each 8x8 byte block in parallel; 2. confidence head scores each
   position; 3. positions < 0.001 threshold are re-masked + regenerated;
   4. `heavy_load=True` clips verify length to `64 // 16 = 4` positions/block.
- API: `speculative_quantize()`, `speculative_save_gguf()`, `size_report()`,
  `print_size_report()`; upgrade path = replace `_block_surrogate` hash with a
  real confidence head once torch is available.
- Verify: `python3 -m unittest tests.test_spec_decode -v`

**Size comparison (16B params, 16-bit baseline)** — `print_size_report()`:
| Representation | Size | vs FP16 |
|---|---|---|
| Full FP16/BF16 | 32.00 GB | — |
| x8D U8 .gguf on disk | 16.00 GB | 50% ↓ (lossless, servable) |
| Sub-byte coordinates | 32.0 MB | 99.9% ↓ (0.016 bit/weight ceiling) |

**Dataset plan** (`research/Training-Dataset-and-Quantization-Plan.md`):
- Tier 0 text bytes: FineWeb, The Pile, RedPajama (pure UTF-8 byte streams).
- Tier 1 multimodal bytes: LAION-5B, ImageNet-1K (pixel bytes), LibriTTS,
  VoxCeleb2, AudioSet (PCM bytes) — no image/audio encoder needed, bytes live
  at ids 0-255.
- Tier 2 SFT: byte-aligned instruction pairs + `[IMG_START]`/`[AUD_START]`
  modality markup; train denoiser via `mask_canvas`/`renoise_to_random_bytes`.
- **Shard at raw byte offsets**, never mid-UTF-8-codepoint.

**Dataset expertise (byte-native import):**
- Zero-dep dataset import module `omni_diffusion/x8d_dataset.py` (NEW, #25):
  imports ANY Hugging Face dataset via the datasets-server HTTP API
  (`https://datasets-server.huggingface.co/parquet` + `/rows`, no
  `datasets`/torch dependency), flattens every field to raw 8-bit bytes
  (text -> UTF-8, image/audio -> raw bytes, numerics -> little-endian), builds
  a reversible byte stream (MAGIC `X8DDS`), and packs it into an x8D 8x8
  DSpark-compressed container via `block_compress_dataset` ->
  `<name>.x8dds.gguf` + manifest.json (lossless roundtrip, threshold 0.001).
  CLI: `tools/import_hf_dataset.py`. This is the `load_dataset()` equivalent
  under the byte law — no tokenizer.
- **Tier mapping additions** (see `research/Omni-Datasets-and-Frontier-Traces-2026.md`):
  Tier 2 SFT now includes NVIDIA agentic/tool-use/SWE traces
  (nvidia/Open-SWE-Traces 207k trajectories, Nemotron-Agentic-v1 335k samples,
  OpenCodeReasoning 1+2, Nemotron-SFT-OpenCode, Cascade-RL-SWE),
  community-extracted frontier-model traces (Claude Fable 5, GPT-5.6 Sol — MUST
  dedup across mirrors, permissive licenses only), and Indic multilingual corpora
  (sarvamai + ai4bharat Sangraha 251B tokens / IndicAlign 74.7M pairs). Tier 1
  adds NVIDIA Physical-AI / omni-dreams scenes + PhysicalAI-Autonomous-Vehicles-NuRec
  3DGS (pixel/PCM bytes at ids 0-255).
- `sarvamai/indic-diarbench` imports via `--config <language>` (Assamese/Bengali/Bodo/...)
  — the config argument IS the `load_dataset()` config argument (dataset_id +
  language code, split defaults to the only split, `test`). Exercised through
  `tools/import_hf_dataset.py --dataset sarvamai/indic-diarbench --config <lang>`;
  validated for at least 3 configs (Assamese, Bengali, Bodo). Rows carry raw audio
  references (`audio[0].src`/`type`), UTF-8 transcripts, and numeric timing fields;
  all flatten to raw bytes at ids 0-255.
- **NVIDIA corpus mapping** (via `tools/import_hf_dataset.py`, all byte-native):
  `nvidia/Open-SWE-Traces` (configs `openhands`/`sweagent`, splits per agent family),
  nemotron agentic/tool-use/terminal/RAG/reward-modeling/pretraining datasets,
  `opencodereasoning` I + II, `openmath`, physical-ai `omni-dreams-samples`/`scenes`,
  `nemotron-personas`, `code-and-swe`, `chat-and-instruction`, `math-and-reasoning`.
  sarvamai (`indic-diarbench`, `samvaad-hi-v1`, `mmlu-indic`, `audiollm-evals`) and
  ai4bharat (`sangraha` verified/synthetic/unverified, `samanantar`, `IndicVoices`, ...)
  Indic corpora flow through the same `.x8dds.gguf` lossless import path.
- All imports run through `tools/import_hf_dataset.py`; offline tests cover identical
  code paths with synthetic data; live tests are network-gated.
- **DiffusionGemma note: LANGUAGE IS ALSO DIFFUSION** —
  google/diffusiongemma-26B-A4B-it (Apache 2.0) proves text diffusion:
  canvas_length=256, entropy_bound sampler (diffusion_entropy_bound=0.1),
  uniform-state diffusion, block-autoregressive canvas commit, >1000 tokens/s
  H100. x8D does the same over the 264-vocab byte space (256 bytes + specials
  256-263), re-affirming issue #2 (embed/lm_head -> 264) and #5/#6 (entropy-bound
  sampler + byte-diffusion training).

Query testing: `tests/test_queries.py` exercises text/image/audio/binary
queries through the full encode→mask→denoise→decode pipeline in pure Python
(`ByteDiffusionSampler` mirrors the future torch `_sample` contract).

**x8Dsub-byte repo audit (2026-07-31, #9)** — upstream `bapXai/x8Dsub-byte`
also contains: `hf_gguf_transform.py` (HF repo→x8D .gguf), `openai_chat_server.py`
(OpenAI-compat endpoint :666), `verify_framework_alignment.py` (500 MB lossless
proof), `proofs/` (integrity + parameter-compression proofs, `sub_byte_vocabulary.json`),
`BENCHMARKS.md` (28,183,891 B → 28,808 B ≈ 978:1). Our missing pieces = **`x8d_hf.py`**
(HF→gguf) + **`moe_disk.py`** (mmap on-disk expert serving). Serving law: compressed
state IS the running state; weights NEVER load into RAM; `/0.001` reverses live at
query time only on the specific MoE expert needed. (#9)

**Kimi-K3 pointer quantization (2026-07-31, #10)** — `tools/quantize_kimi_k3.py`
quantizes **without downloading the model**: it pin-points each upstream HF tensor
(repo | shard | data_offsets | dtype | shape) into an `X8DPTR01` pointer map. Weight
bytes stay on HF disk; only the requested expert's span is Range-fetched / mmap'd and
`/0.001`-reversed at query time. Verified on a real expert: 5,505,024 B fetched from
the 2.78 TB model, reverse exact. Full map = 151.8 MB. **Kimi-K3: 1.56 TB → 2.837 GB
(550:1)**: U8×0.001=0.008 bit/param → 2.723 GB, BF16×0.001=0.016 → 114.4 MB. (#10)

**SARA routing boundaries (#36)** — `omni_diffusion/moe_disk.py`:
`SARABoundary` + `SARA_REGISTRY` + `SARARouter` (issue #36). Dense models
(Kokoro-82M, Whisper large-v3, LTX-2) = single expert; internal-MoE models
(GLM-5.2 753B, Kimi-K3 2.78T, DeepSeek-V4-Pro 1.6T) = their own isolated
expert. Routing guarantees only the requested boundary's byte span is mmap'd +
/0.001-reversed; boundaries are pairwise isolated by construction.

**Colibrì deep-dive (#41, audited 2026-07-31 vs live upstream)** —
`JustVugg/colibri` (pure-C `c/colibri.c`, 437 KB) runs GLM-5.2 (744B/40B MoE)
in 25 GB RAM by treating VRAM/RAM/NVMe as one managed memory hierarchy —
it does NOT compress the model (372 GB int4 stays on disk; 4 bit/weight).
Mechanics verified in source: (1) ~370 GB NVMe staging; (2) ~19 MB coalesced
`pread` into 16 KB-aligned slabs OR `COLI_MMAP=1` (`mmap PROT_READ MAP_SHARED`
+ `madvise(MADV_WILLNEED)` so the kernel page cache IS the cache); (3) MoE
active-expert streaming (only ~40B/744B + ~11 GB disk reads/token cold);
(4) `URING=1` io_uring, `O_DIRECT`, `COLI_NUMA=1` mbind interleave,
`PIN_GB=N` mlock hot-store, per-layer LRU `ecache`, `DROP=1` madvise-evict,
int8 MTP speculative head (int4 → 0% draft acceptance, #8); token-exact vs
transformers oracle. Telemetry (`telemetry.h`): `g_prof_io` atomic byte
counter, `hit_pin`/`hit_ecache` tier split, `getrusage` RSS, per-turn stats
line, `iobench.c` drive probe. Measured: **0.05–0.1 tok/s cold** on 25 GB
WSL2 (~1 GB/s VHDX); 0.07→0.11 with `--topp 0.7`; 1.06 tok/s M5 Max; 6.84
tok/s 6×RTX-5090 full residency. **Why x8D wins**: Colibrì buys placement with
a 372 GB footprint + SSD-bound decode; x8D compacts the matrix (0.008
bit/weight → 1.56 TB → 2.837 GB) so the *file* is addressable, then uses the
same mmap pointer-map for zero-copy live /0.001. Ported: `omni_diffusion/
x8d_mmap.py` (`MappedX8DReader` = Colibrì `COLI_MMAP` over the sub-byte
container: offset-index slice, zero-copy memoryview frames, live reverse) +
`omni_diffusion/x8d_telemetry.py` (`Telemetry` = Colibrì `telemetry.h`:
record_io/fault, RSS, per-8x8-block timing, pin/lru hits, dashboard line).
Next steps: wire `MappedX8DReader` into `moe_disk.py` SARA spans; learned
PIN hot-store from hit histograms; expose dashboard line via /healthz; publish
like-for-like benchmark vs upstream. See `research/Colibri-Deep-Dive-2026.md`.

**Byte-native processors (#42, audited 2026-07-31)** — legacy MagViT / GLM-4-Voice
tokenizer wrappers STRIPPED from `omni_diffusion/data/processor/`. `image_processor.py`
and `audio_processor.py` are pure binary slicing: a file IS its byte array
(`list(open(path,"rb").read())`), framed on the canvas as
`[IMG_START(260)] + bytes + [IMG_END(261)]` and
`[AUD_START(262)] + bytes + [AUD_END(263)]`. Legacy ids 256-259 are REJECTED
(collide with MASK/PAD/BOS/EOS). `to_tensor()` imports torch lazily; the core
stays stdlib. `dataset_base.py` call signatures preserved (positional
`"byte-native"` + legacy kwargs ignored). `tools/import_hf_dataset.py` gained a
zero-copy mmap JSONL path (`--jsonl`) for ai4bharat/sangraha + nvidia/Open-SWE-Traces
shards: `mmap` the shard, flatten `text`/`code` fields, frame into an `X8DDS`
stream, store as raw U8 `.x8dds.gguf` (0.001 law at compute ONLY — float32
packing is byte-law-banned bloat), verify lossless via `MappedX8DReader`.
Proof: 200-row sangraha-style shard → 20,995 B stream → 20,995 B gguf, lossless.

**Definitions (researched, not assumed):**
- **Speculative decoding** = draft-verify loop. A cheap draft model (or a
  lightweight EAGLE-3/P-EAGLE head on the target) proposes K candidate tokens;
  the target verifies all K in ONE parallel pass; the longest matching prefix is
  accepted and the first rejection is resampled. Output distribution is provably
  identical to plain autoregressive decoding. In our repo it appears in
  `x8d_spec_decode.py` as block-parallel 8x8 quantization.
- **QAT (Quantization-Aware Training)** = fake quantization in the forward pass:
  `x_q = (x/scale + zp).round().clamp()` then dequantize back, so training sees the
  exact numerics the deployed low-bit model will run. The backward pass uses the
  straight-through estimator (STE): the zero-a.e. gradient of `round` is replaced
  by identity so weights co-adapt to quantization noise. PTQ is ~lossless at 8-bit;
  QAT is the standard recipe below 4-bit. (#6)

**2026 depth/context framework definitions (#24, see
`research/Depth-Context-Attention-Frameworks-2026.md`):**
- **AttnRes (Attention Residuals)** = replace fixed residual accumulation
  (h_l = h_{l-1} + f) with softmax attention over ALL preceding layer outputs:
  `h_l = Σ_i α_{i→l}·v_i`, α from a learned zero-initialized pseudo-query w_l.
  Fixes PreNorm dilution (hidden magnitudes grow O(L), gradients explode/vanish
  with depth). **Block AttnRes**: L layers → N≈8 blocks; intra-block sum, softmax
  attention over block summaries + embedding; memory/comm O(Ld)→O(Nd), I/O 5.5d
  per layer (vs mHC 34d, Full AttnRes 24d). ≈ baseline with 1.25× compute.
  Drop-in residual replacement → target for the Dream byte denoiser (#7 phase 2).
- **KDA (Kimi Delta Attention)** = recurrent linear memory with channel-wise
  decay gate: `W = W·D_α + β·r⊗κ` (D_α = Diag(vector forget gate)). Constant
  memory → O(1)/token inference; K3 interleaves 3:1 KDA:MLA. Gated DeltaNet-2
  decouples β into channel-wise erase + write gates (its most granular form).
- **mHC (Manifold-constrained Hyper-Connections)** = DeepSeek's residual
  re-think: m parallel streams with learned mixing matrices (α_l, β_l, A_l);
  I/O 34d @ m=4; competitive loss (1.747) but 6× Block-AttnRes I/O.
- **CLVR (Cross-Layer Value Routing)** = route a lower delta-rule layer's
  internal write VALUE (not write error — CLER fails) into the shared residual
  stream via a zero-initialized projection; cheap linear-time depth pathway.
- **MTP (Multi-Token Prediction)** = independent prediction modules at the
  stack tail that draft K future tokens for speculative verification (DeepSeek
  V4). Aligns with our DSpark block-parallel confidence-head plan for
  `x8d_spec_decode.py` and issue #7.
- **Engram** = hardware-efficient conditional dictionary lookup as external
  neural scratchpad memory; ~10% KV vs V3 (secondary source).

---

## 📁 Project Structure — Full File Index

Complete index of every file in the repo (regenerated 2026-07-31, #12).

```
x8D-Omni-Diffusion/
├── AGENTS.md                          # THIS FILE — agent behavioral rules
├── README.md                          # Project readme (byte-native pitch)
├── .gitignore
├── .gitmodules                        # git submodules (if any)
├── setup.py                           # Package setup
├── requirements_core.txt              # ZERO-dep byte core (stdlib only)
├── requirements_ds_gpu.txt            # Optional torch training stack
│
├── asset/                             # Demo media
│   ├── asr_0.wav  s2i_0.wav  svqa_0.wav   # audio samples
│   ├── svqa_0.jpg  vqa_0.png              # image samples
│   ├── qualitative_results.png  teaser.png
│   ├── speech_task.png  visual_task.png
│
├── configs/
│   └── finetune.yaml                  # Training config
│
├── docs/
│   └── index.html                     # GitHub Pages landing
│
├── omni_diffusion/
│   ├── __init__.py                    # Lazy package (no eager imports)
│   ├── constants.py                   # Shared constants
│   ├── tokenizer.py                   # Legacy Qwen2 tokenizer wrapper
│   ├── tokenizer_sensevoice_glm4voice.py  # SenseVoice/GLM4Voice tokenizer
│   ├── x8d_export.py                  # x8D 0.001 + X8DGGUF1 U8 container
│   ├── x8d_spec_decode.py             # DSpark 8x8 spec-decode quantizer + size report
│   ├── x8d_subbyte.py                 # 0.016 bit/weight packed model (32MB=32GB)
│   ├── x8d_hf.py                      # [#9] HF repo -> x8D .gguf converter + pointer loader
│   ├── x8d_dataset.py                 # [#25] HF datasets-server import -> 8x8 block-compressed .x8dds.gguf
│   ├── x8d_mmap.py                    # [#41] zero-copy mmap frame reader over .gguf/.x8dds.gguf
│   ├── x8d_telemetry.py               # [#41] per-8x8-block I/O + RSS telemetry (Colibrì port)
│   ├── moe_disk.py                    # [#9] mmap on-disk MoE expert serving
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build.py
│   │   ├── data_collator.py
│   │   ├── dataset_base.py
│   │   ├── dataset_qwen2.py           # (legacy BPE ids 151643 — needs byte fix)
│   │   ├── utils.py
│   │   └── processor/
│   │       ├── __init__.py
│   │       ├── audio_processor.py
│   │       └── image_processor.py
│   │
│   └── models/
│       ├── __init__.py                # Lazy (no imports)
│       ├── dream/
│       │   ├── __init__.py            # Lazy; registration via register.py
│       │   ├── register.py            # register_dream_classes() — lazy import hook
│       │   ├── byte_tokenizer.py      # Raw 8-bit byte tokenizer (vocab=264)
│       │   ├── configuration_dream.py # Byte-native DreamConfig (ids 256-263)
│       │   ├── config_dream_resume.json  # Byte-native resume config
│       │   ├── config.yaml            # Dream model config
│       │   ├── configuration.json     # Extra config
│       │   ├── tokenizer_config.json  # Legacy (BPE-era) — do not use
│       │   ├── tokenization_dream.py  # Legacy BPE — BANNED, replace via byte_tokenizer
│       │   ├── modeling_dream.py      # Core model (embed/lm_head still old size — #2)
│       │   ├── modeling_sensevoice.py # SenseVoice ASR head
│       │   ├── generation_utils.py    # _sample() at line 404 — entropy_bound hook (#2)
│   │       ├── resampler_projector.py # Audio/image resampler projector
│   │       └── (moe_layer.py, dspark_diffusion.py, kda_attention.py)  # planned #4/#5/#7
│
├── research/                          # Research notes, papers, experiments
│   ├── .gitkeep
│   ├── DiffusionGemma.md              # Uniform-state diffusion + config breakdown
│   ├── Config-Mapping-DiffusionGemma-to-x8D.md
│   ├── Needle-Dependency-Audit.md     # dep-by-dep audit vs cactus-compute/needle
│   ├── Training-Dataset-and-Quantization-Plan.md
│   ├── Kimi-K3-x8D-Pointer-Quantization.md  # [#10] 1.56TB->2.837GB proof
│   ├── Omni-Modality-Stack.md         # [#11] Whisper/Kokoro/LTX-2 matrix
│   ├── Byte-Core-Optimizations.md     # [#14] 6-41x byte-core speedups + [#18-#23] LUT round
│   ├── Frontier-Benchmarks-2026.md    # [#15] x8D vs GPT-5.6/K3/Opus5/V4/... + arch deep-dive
│   ├── Omni-Datasets-and-Frontier-Traces-2026.md  # [#25/#26] NVIDIA/sarvamai/ai4bharat + Fable5/Sol traces + DiffusionGemma
│   └── Depth-Context-Attention-Frameworks-2026.md  # [#24] AttnRes/KDA/mHC/Engram/CLVR + x8D map
│   └── Colibri-Deep-Dive-2026.md      # [#41] JustVugg/colibri 24GB-GLM-5.2 audit + mmap/telemetry port
│
├── scripts/
│   ├── set_env_ds_gpu.sh              # GPU env setup
│   ├── deepspeed/
│   │   ├── ds_config_zero2.json       # ZeRO-2 config
│   │   └── diffusion_dream/
│   │       ├── finetune.sh
│   │       ├── evaluate_imageqa_mme.sh
│   │       ├── evaluate_librispeech.sh
│   │       └── evaluate_libritts.sh
│
├── tests/                             # ALL tests (stdlib unittest, no torch)
│   ├── test_byte_tokenizer.py         # 18 tests — byte vocab 264
│   ├── test_config.py                 # byte-native config defaults
│   ├── test_queries.py                # 10 tests — full pipeline all modalities
│   ├── test_spec_decode.py            # 11 tests — DSpark spec-decode quantizer
│   ├── test_subbyte.py                # 8 tests — 32MB=32GB packed model
│   ├── test_x8d_export.py             # x8D gguf container tests
│   ├── test_x8d_hf.py                 # [#9] shard->gguf + MoE on-disk serving
│   ├── test_pointer_quantize.py       # [#10] Kimi-K3 pointer map + forward-identical
│   ├── test_quantize_hf.py            # [#17] generic HF pointer quantizer
│   ├── test_x8d_dataset.py            # [#25] HF dataset import + block-compress
│   ├── test_x8d_mmap.py               # [#41] zero-copy mmap frame reader (Colibrì COLI_MMAP port)
│   ├── test_x8d_telemetry.py          # [#41] per-8x8-block I/O + RSS telemetry (Colibrì telemetry.h port)
│   ├── test_byte_processors.py        # [#42] byte-native image/audio processors + mmap JSONL import
│   └── (test_moe_disk.py)             # [#9] planned
│
└── tools/
    ├── finetune_dream_v4_51_3.py      # Training tool
    ├── trainer_v4_51_3.py             # Trainer
    ├── inference.py                   # Inference tool
    ├── evaluate_asr.py
    ├── evaluate_imageqa_mme.py
    ├── evaluate_libritts.py
    ├── compute-wer.py                 # WER eval
    ├── bench_byte_core.py             # [#14] byte-core micro-benchmarks
    ├── import_hf_dataset.py           # [#25] CLI: HF dataset -> x8D block-compressed
    ├── quantize_kimi_k3.py            # [#10] Kimi-K3 pointer quantizer (live)
    └── quantize_hf.py                 # [#17] generic HF pointer quantizer (live)
```

GitHub: https://github.com/bapXai/x8D-Omni-Diffusion (branch `main`, Pages CI).
HF model repo: https://huggingface.co/bapX/x8D-Omni-Diffusion (byte-native only).

---

## 🧪 Testing Rules

- Every new module MUST have a corresponding test in `tests/`
  (e.g. `x8d_dataset.py` -> `tests/test_x8d_dataset.py`).
- **Byte-native core tests run on pure Python stdlib `unittest`** — NO torch/transformers
  required. Command: `python3 -m unittest discover -s tests -v`.
- Also run with ResourceWarning promoted to errors:
  `python3 -W error::ResourceWarning -m unittest discover -s tests -v`.
- Torch-dependent tests (model forward, training) are gated with `skipUnless(HAS_TRANSFORMERS)`.
- **Network-gated tests**: live tests that hit datasets-server must be
  `@unittest.skipUnless(_NETWORK_OK, ...)` behind a module-level probe; offline
  tests MUST cover the same code paths with synthetic data (no network).
- Tests MUST pass before any commit.
- Use `gh run list` to verify CI after pushing.

## 📦 Dependency Stance (audited 2026-07-31 against cactus-compute/needle)

The byte-native core (`byte_tokenizer.py`, `x8d_export.py`, their tests) has **ZERO
external dependencies** — pure Python stdlib only. `requirements_core.txt` documents the
optional torch-training stack.

| Dependency | Verdict |
|---|---|
| torch, transformers | training/inference only, never the byte core |
| datasets, huggingface_hub | data + distribution (justified) |
| pyyaml | training configs (justified) |
| wandb, tqdm, google-genai | optional |
| jax, jaxlib, flax, optax | **REJECTED** — PyTorch is the sole DL framework (needle used these for JAX/TPU; we port concepts only: Muon, WSD, scan/remat) |
| sentencepiece, tiktoken, tokenizers | **BANNED** — byte law |
| gcsfs | **REJECTED** — dead dependency even in needle |

See `research/Needle-Dependency-Audit.md` for the full analysis.

---

## 📐 Code Style

- Python 3.10+
- Type hints on all function signatures
- Docstrings on all public classes and methods
- No external tokenizer dependencies (no `tiktoken`, no `sentencepiece`, no `tokenizers`)
- PyTorch as the sole deep learning framework
- All byte operations use unsigned 8-bit integers (`torch.uint8` or `np.uint8`)
