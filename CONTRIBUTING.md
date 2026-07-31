# Contributing to x8D-Omni-Diffusion

Thanks for contributing! This guide tells you how to land changes that fit the
byte-native architecture. Please read it fully before opening an issue or a PR.

- [The Byte Law](#the-byte-law)
- [Getting Started](#getting-started)
- [Issue-Driven Workflow](#issue-driven-workflow)
- [Git Workflow (gh CLI)](#git-workflow-gh-cli)
- [Dual Commit: GitHub + HF Model Repo](#dual-commit-github--hf-model-repo)
- [Testing Rules](#testing-rules)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Never Commit These](#never-commit-these)
- [Code Style](#code-style)
- [Running the Benchmark](#running-the-benchmark)
- [Getting Help](#getting-help)

## The Byte Law

**There are NO tokens in this project — only raw 8-bit bytes (0–255).** Every
contribution MUST treat the 256 unsigned byte states as the sole native
vocabulary. The embedding layer and `lm_head` are sized **264**:
bytes 0–255 plus MASK=256, PAD=257, BOS=258, EOS=259, IMG_START=260,
IMG_END=261, AUD_START=262, AUD_END=263.

Enforced in every contribution:

1. **Never** import or reference BPE/SentencePiece/WordPiece tokenizers.
2. **Never** add `vocab.json`, `merges.txt`, or any merge-based encoding file.
3. All data pipelines convert inputs to raw byte arrays: `list(data_bytes)`.
   No encoding step, no vocabulary lookup.
4. Text is UTF-8 bytes; images are raw pixel bytes; audio is raw PCM bytes.
   They all live at ids 0-255 on the same diffusion canvas.
5. `config.json` / `generation_config.json` MUST be byte-native (`vocab_size=264`,
   special ids 256-263, `tie_word_embeddings=true`).

> [!WARNING]
> A PR that reintroduces a tokenizer or a sub-word vocabulary is rejected.

## Getting Started

```bash
export PATH="/Users/getwinharris/.local/bin:$PATH"   # hf + gh CLIs
git clone https://github.com/bapXai/x8D-Omni-Diffusion.git
cd x8D-Omni-Diffusion
```

The byte-native core (`byte_tokenizer.py`, `x8d_export.py`, `x8d_dataset.py`,
their tests) is **pure Python stdlib** — it runs without torch/transformers.
The optional GPU training stack lives in `requirements_ds_gpu.txt`.

## Issue-Driven Workflow

1. **Create a GitHub issue FIRST** before writing any code:

   ```bash
   gh issue create --title "feat: <short description>" \
     --body "## Description\n<details>\n\n## Acceptance Criteria\n- [ ] <criterion>" \
     --label "enhancement"
   ```

   Bug reports use the `bug` label and include steps to reproduce + expected
   behavior.
2. **Work on the fix/feature** and reference the issue in every commit:
   `git commit -m "fix(#42): ..."`.
3. **Validate**: run the full test suite (see below), push, and confirm CI is
   green with `gh run list`.
4. **Close the issue** with `gh issue close <number> --comment "Fixed in <sha>"`.

## Git Workflow (gh CLI)

All git operations use the GitHub CLI (`gh`).

```bash
git checkout main && git pull
# make changes on a topic branch (do NOT work directly on main)
git checkout -b feat/my-change
git add -A
git commit -m "feat(#42): byte-native widget"
git push origin main
gh run list --limit 5          # confirm CI is green
gh run view <run-id>           # inspect a failing run
```

Open a PR with `gh pr create --title "feat: <desc>" --body "Closes #42"` and
merge with `gh pr merge <number> --squash --delete-branch`. PRs must keep
changes focused: one logical unit per PR.

## Dual Commit: GitHub + HF Model Repo

Changes that add or touch byte-native artifacts (`omni_diffusion/`, `tools/`,
`README.md`, `research/`, `CONTRIBUTING.md`) MUST also be synced to the Hugging
Face **model repo** `bapX/x8D-Omni-Diffusion` (a model repo, NOT a bucket —
issue #27 migrated away from the bucket):

```bash
export PATH="/Users/getwinharris/.local/bin:$PATH"
hf auth whoami                                  # must be bapX

# Stage only byte-native artifacts
mkdir -p staged_dir && cp -r omni_diffusion staged_dir/

# Sync (uploads/deletes/skips delta)
hf repositories sync ./staged_dir/ bapX/x8D-Omni-Diffusion

# Verify
hf repositories list bapX/x8D-Omni-Diffusion
```

The model repo is byte-native only: NO `*.safetensors`, NO `vocab.json`,
NO `merges.txt`, NO BPE tokenizer files, NO `safetensors.index.json`.

## Testing Rules

- Every new module MUST ship with a test in `tests/`
  (e.g. `x8d_dataset.py` -> `tests/test_x8d_dataset.py`).
- Byte-native core tests run on **pure Python stdlib `unittest`** — no torch,
  no network. Run the whole suite:

  ```bash
  python3 -m unittest discover -s tests -v
  ```

- Also run with `ResourceWarning` promoted to errors:

  ```bash
  python3 -W error::ResourceWarning -m unittest discover -s tests -v
  ```

- **Network-gated tests** (live HF datasets-server hits) are guarded with
  `@unittest.skipUnless(_NETWORK_OK, ...)` behind a module-level probe; the same
  code paths MUST be covered offline with synthetic data.
- Torch-dependent tests are gated with `skipUnless(HAS_TRANSFORMERS)`.
- Tests MUST pass before any commit. After pushing, verify CI with
  `gh run list --limit 5`.

## Adding a New Dataset

Datasets are imported byte-native through `omni_diffusion/x8d_dataset.py` +
`tools/import_hf_dataset.py` — the `load_dataset()` equivalent with no
tokenizer and no `datasets`/torch dependency.

```bash
python3 tools/import_hf_dataset.py --dataset sarvamai/indic-diarbench \
    --config Assamese --split train --length 50 --out ./datasets/
```

The flow: `resolve_hf_dataset` (datasets-server HTTP API) -> `rows_to_byte_stream`
(reversible `X8DDS` framing) -> `block_compress_dataset` (DSpark 8x8
speculative-decode quantizer) -> `<name>.x8dds.gguf` + `manifest.json`
(lossless roundtrip at the 0.001 sub-byte threshold). Every field lands as raw
8-bit bytes: text -> UTF-8, image/audio -> raw bytes, numerics -> little-endian.

Guidelines:

- Prefer **permissive-license** corpora (Tier 0/1/2 map lives in
  `research/Omni-Datasets-and-Frontier-Traces-2026.md`).
- Shard at **raw byte offsets**, never mid-UTF-8-codepoint.
- Add synthetic offline coverage in `tests/test_x8d_dataset.py` for any new
  path; keep live tests behind `_NETWORK_OK`.

## Never Commit These

- `*.safetensors`, `*.safetensors.index.json`, `*.ggml` / raw float checkpoints.
- `vocab.json`, `merges.txt`, `added_tokens.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `tokenization_dream.py` (legacy BPE artifacts).
- Any secret: HF tokens, `~/.hf-cli` credentials, API keys, `.env` files.
- Bulk `datasets/` uploads; import via `tools/import_hf_dataset.py` instead.

## Code Style

- Python 3.10+; type hints on every function signature.
- Docstrings on all public classes and methods (the repo follows Google-style).
- No external tokenizer dependencies (`tiktoken`, `sentencepiece`,
  `tokenizers` are BANNED).
- PyTorch is the sole DL framework; the byte core stays stdlib-only.
- All byte operations use unsigned 8-bit integers (`torch.uint8`/`np.uint8`
  for torch paths, `bytes`/`bytearray` elsewhere).
- No comments unless they earn their place; match the surrounding style.

## Running the Benchmark

There is no `tools/benchmark_sandbox.py` in this repo — run the byte-core
micro-benchmarks instead:

```bash
python3 tools/bench_byte_core.py
```

This exercises `pack_subbyte`/`unpack_subbyte`, the DSpark speculative
quantizer, mmap on-disk `SubByteModel` serving, and prints the 16B-model
scaling table (FP16 vs x8D U8 vs sub-byte coordinates).

## Getting Help

- Project: https://github.com/bapXai/x8D-Omni-Diffusion
- HF model repo: https://huggingface.co/bapx/x8D-Omni-Diffusion
- For questions that are not bugs or concrete changes, open a Discussion
  instead of an issue. For bugs and features, open an issue first (see above).
