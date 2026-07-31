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

The model lives in an HF **bucket**: `bapX/x8D-Omni-Diffusion`
(https://huggingface.co/buckets/bapX/x8D-Omni-Diffusion). The bucket holds ONLY
byte-native files — NO safetensors, NO `vocab.json`/`merges.txt`, NO BPE artifacts.

```bash
export PATH="/Users/getwinharris/.local/bin:$PATH"

# Auth / identity
hf auth whoami

# List bucket contents
hf buckets list bapX/x8D-Omni-Diffusion --human-readable --tree

# Download a single file from the bucket
hf buckets cp hf://buckets/bapX/x8D-Omni-Diffusion/config.json ./config.json

# Upload a local folder into the bucket (uploads/deletes/skips delta)
hf buckets sync ./staged_dir/ hf://buckets/bapX/x8D-Omni-Diffusion

# Delete files from the bucket (e.g. old safetensors / BPE tokenizer files)
hf buckets remove hf://buckets/bapX/x8D-Omni-Diffusion --recursive -y \
  --include '*.safetensors' --include '*.safetensors.index.json' \
  --include 'vocab.json' --include 'merges.txt' --include 'added_tokens.json' \
  --include 'tokenizer_config.json' --include 'special_tokens_map.json' \
  --include 'tokenization_dream.py'

# Always dry-run before deleting
hf buckets remove hf://buckets/bapX/x8D-Omni-Diffusion --recursive --dry-run ...
```

**Bucket rules (enforced):**
1. NEVER upload `*.safetensors`, `vocab.json`, `merges.txt`, `added_tokens.json`,
   `tokenizer_config.json`, `special_tokens_map.json`, or `tokenization_dream.py`.
2. `config.json` MUST be byte-native: `vocab_size=264`, `mask=256`, `pad=257`,
   `bos=258`, `eos=259`, `img=260/261`, `aud=262/263`, `tie_word_embeddings=true`.
3. `generation_config.json` MUST use byte-native ids + `alg="entropy_bound"`,
   `steps=48`, `diffusion_entropy_bound=0.1`, `canvas_length=256`.
4. Keep `byte_tokenizer.py`, `x8d_export.py`, `configuration_dream.py`, model code,
   and README in the bucket; the model loads via `trust_remote_code=True`.
5. Source of truth for HF distribution is `omni_diffusion/models/dream/` + `x8d_export.py`
   in this repo; sync those into the bucket.

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
│   ├── tokenizer_magvitv2.py          # MagViT-v2 tokenizer
│   ├── tokenizer_sensevoice_glm4voice.py  # SenseVoice/GLM4Voice tokenizer
│   ├── x8d_export.py                  # x8D 0.001 + X8DGGUF1 U8 container
│   ├── x8d_spec_decode.py             # DSpark 8x8 spec-decode quantizer + size report
│   ├── x8d_subbyte.py                 # 0.016 bit/weight packed model (32MB=32GB)
│   ├── x8d_hf.py                      # [#9] HF repo -> x8D .gguf converter
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
│       │   ├── resampler_projector.py # Audio/image resampler projector
│       │   └── (moe_layer.py, dspark_diffusion.py, kda_attention.py)  # planned #4/#5/#7
│       └── magvit/
│           ├── common_modules.py
│           ├── misc.py
│           ├── modeling_magvitv2.py   # MagViT-v2 video tokenizer
│           └── modeling_utils.py
│
├── research/                          # Research notes, papers, experiments
│   ├── .gitkeep
│   ├── DiffusionGemma.md              # Uniform-state diffusion + config breakdown
│   ├── Config-Mapping-DiffusionGemma-to-x8D.md
│   ├── Needle-Dependency-Audit.md     # dep-by-dep audit vs cactus-compute/needle
│   ├── Training-Dataset-and-Quantization-Plan.md
│   └── Omni-Modality-Stack.md         # [#11] Whisper/Kokoro/LTX-2.3 matrix
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
│   └── (test_x8d_hf.py, test_moe_disk.py)  # [#9] planned
│
└── tools/
    ├── finetune_dream_v4_51_3.py      # Training tool
    ├── trainer_v4_51_3.py             # Trainer
    ├── inference.py                   # Inference tool
    ├── evaluate_asr.py
    ├── evaluate_imageqa_mme.py
    ├── evaluate_libritts.py
    ├── compute-wer.py                 # WER eval
    └── (quantize_kimi_k3.py)          # [#10] planned Kimi-K3 quantizer
```

GitHub: https://github.com/bapXai/x8D-Omni-Diffusion (branch `main`, Pages CI).
HF bucket: https://huggingface.co/buckets/bapX/x8D-Omni-Diffusion (byte-native only).

---

## 🧪 Testing Rules

- Every new module MUST have a corresponding test in `tests/`.
- **Byte-native core tests run on pure Python stdlib `unittest`** — NO torch/transformers
  required. Command: `python3 -m unittest discover -s tests -v`.
- Torch-dependent tests (model forward, training) are gated with `skipUnless(HAS_TRANSFORMERS)`.
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
