# Deep Research: Cactus-Compute/needle — Dependencies & Architecture for x8D-Omni-Diffusion

**Date:** 2026-07-31
**Sources:** github.com/cactus-compute/needle (cloned, 5,250 LOC), Cactus-Compute/needle HF repo, README, config.json.

## 1. What Needle is

A **26M-param encoder-decoder "Simple Attention Network"** for single-shot function calling on
tiny devices (14 MB, 1–6k tok/s on mobile/edge). MIT license, 3.3k stars.

- d=512, 8 heads / 4 KV, SentencePiece **BPE vocab 8192** (NOT byte-native).
- 12 encoder layers (self-attn + RoPE + gated residual, **no FFN**) → 8 decoder layers
  (masked self-attn + **cross-attn** + gated residual).
- **ZCRMSNorm** (zero-centred RMSNorm, scale init 0, applied as `(1+γ)·x/RMS(x)`).
- Tied embeddings, tied output `logits = h @ embedding.T`.
- Contrastive retriever head (`encode_contrastive`) for tool retrieval, plus CLIP-style loss.
- **Matryoshka FFN**: FFN intermediate sliced for export (`export_submodel`, shrink factor).
- **INT4/INT8 fake-quantization with Straight-Through Estimator (STE)** during training
  (`_fake_quantize_int4/int8`, group_size=32) — QAT, exactly issue #6 in our repo.
- Trained on **16 TPU v6e** for 200B tokens (27h) + 2B function-call tokens (45 min).

**Needle architecture diagram** (README):
```
Text query → Embedding → Encoder×12 (self-attn + RoPE, no FFN)
                       → (KV) → Decoder×8 (self-attn + cross-attn + gated residual)
                              → ZCRMSNorm → Linear(T) [tied] → Softmax → Tool Call
```

## 2. Dependency-by-dependency usage (what we're missing vs. using)

Needle's `pyproject.toml` deps: `jax, jaxlib, flax, optax, datasets, huggingface_hub,
transformers, wandb, sentencepiece, scipy, gcsfs, tqdm, google-genai`. Plus `[tpu]`
=`jax[tpu]`, `[gpu]`=`jax[cuda12]`.

| Dep | Needle uses it for | Do we need it? |
|---|---|---|
| **jax + jaxlib** | Entire model (architecture.py), jit/pmap, sharding, `jax.random`, `jax.tree` | **Optional/NO** — we are PyTorch-per-AGENTS.md. JAX gives TPU support + remat/scan memory tricks. Not required for byte-native core. |
| **flax** | `nn.Module` model defs, `nn.scan` (layer unrolling), `nn.remat` (gradient checkpointing), `train_state.TrainState` | **NO** — torch `nn.Module` replaces this. Steal the ideas: `nn.scan` ≡ layer weight-tied loop, `nn.remat` ≡ `torch.utils.checkpoint`. |
| **optax** | Muon optimizer, AdamW, WSD schedule, multi-transform (muon for 2D/3D kernels, adam rest), clip_by_global_norm, softmax CE | **NO (optax)**, but **YES (concepts)** — Muon + WSD + grad-clip should port to torch. optax `multi_transform` ≡ separate param groups in torch. |
| **datasets** | `load_dataset("PleIAs/SYNTH", split="train", streaming=True)` for pretrain; tool-call dataset for SFT | **YES** — we already import it in `omni_diffusion/data/build.py`. Needed for our dataset pipeline. |
| **huggingface_hub** | Auto-download tokenizer + checkpoints (`hf_hub_download`), auto-upload checkpoints (`HfApi.upload_file` in a background thread), `create_repo` | **YES** — needed for `bapx/x8D-Omni-Diffusion` weights + tokenizer download/upload. |
| **transformers** | Only sets `logging.getLogger("transformers").setLevel(logging.ERROR)` — that's it! No model code depends on it | **NO (for needle)** / **YES (for us)** — we use transformers `PreTrainedTokenizer`, `PretrainedConfig`, Trainer infra. |
| **wandb** | Training logging (`--wandb` flag) | **Optional** — only if we want experiment tracking. |
| **sentencepiece** | BPE tokenizer train + encode (`spm.SentencePieceTrainer.Train`, `byte_fallback=True`) | **NO — BANNED.** Byte law forbids SentencePiece. This is the exact layer we are *removing*. |
| **scipy** | (declared; minimal direct use) | **NO** |
| **gcsfs** | Declared in requirements/pyproject but **zero direct imports** found | **NO** — dead dependency in needle. |
| **pyyaml** | Declared but **zero direct imports** found | **NO** — dead dependency. |
| **tqdm** | Progress bars everywhere | **Optional** (nice for training loops). |
| **google-genai** | `generate.py` synthesizes function-call training data via `genai.Client(...).models.generate_content` (Gemini) | **Optional** — only for synthetic data generation (we could use it for SFT data gen). |

**Key finding:** needle lists `gcsfs` and `pyyaml` but never imports them — they are
**dead dependencies**. Its `transformers` usage is just log-suppression. The *real* stack is
`jax/flax/optax` (compute) + `datasets`/`huggingface_hub` (data) + `sentencepiece` (tokenizer).

## 3. What needle has that we should ADOPT (with byte-native twist)

1. **Muon optimizer** (`scale_by_muon`): orthogonalizes 2D/3D grads via Newton–Schulz
   (5 steps), then Nesterov momentum. Matmul-heavy byte-model layers converge much faster.
   Port: torch `@torch.compile`/manual polar decompo. **High value for our from-scratch byte model.**
2. **WSD schedule** (`_wsd_schedule`): warmup → stable → cosine-decay 15%. Cheap, standard.
3. **ZCRMSNorm** (zero-centred): scale initialized to 0 → stable gated residual training.
   DREAM already has RMSNorm; adding zero-init scale is a small, safe upgrade.
4. **INT4/INT8 STE QAT** (`_fake_quantize_*`): matches our issue #6 exactly. Their
   `_quantize_params` applies it per Dense kernel in the training loop via `jax.lax.cond`.
   **For us:** combine with x8Dsub-byte 0.001 export — train with STE fake-quant, export
   as U8 quanta. Confirms the x8Dsub-byte export path must read *fake-quantized* weights.
5. **Matryoshka FFN export** (`export_submodel`): slice FFN by factor → one checkpoint
   yields many model sizes. Pairs beautifully with x8D .gguf containers (one export, many
   sizes, all in U8).
6. **Contrastive tool/byte retriever** (`encode_contrastive` + CLIP loss + `retrieve_tools`):
   a cheap dense retriever for tool schemas — byte-native equivalent: retrieve over raw
   byte-prefix embeddings. Replaces the LLM needing all tools in-context.
7. **Constrained decoding** (`constrained.py`, `constrain_logits`): grammar-constrained
   generation for tool JSON. Byte-native version: constrain to `{` `"` etc. byte-level —
   **important**, since byte-vocab models need explicit structural guards.
8. **`nn.scan` + `nn.remat`**: weight-tied layer scan + gradient checkpointing → tiny model
   footprint. Torch equivalent: loop over shared layer weights + `torch.utils.checkpoint`.
9. **TPU-ready setup script** (`setup`): detects gpu/tpu/cpu, installs the right backend,
   enables THP, caches XLA. We don't need XLA, but the venv-first pattern is right.

## 4. What we REJECT (explicitly)

- `sentencepiece` — BPE banned by the byte law (AGENTS.md). Needle's tokenizer is exactly
  the BPE layer we delete.
- `jax/jaxlib/flax/optax` — AGENTS.md pins PyTorch as sole DL framework. We adopt Muon/WSD
  *concepts*, not the libs.
- `gcsfs`, `pyyaml` — dead deps, will not be added.
- `wandb` — only optional.

## 5. Concrete "missing dependencies" audit for OUR repo

From the user's list (`jax, jaxlib, flax, optax, datasets, huggingface_hub, gcsfs,
transformers, wandb, pyyaml, sentencepiece, google-genai`):

| Dep in user list | Verdict for x8D-Omni-Diffusion |
|---|---|
| jax / jaxlib / flax / optax | ❌ Not adopted (PyTorch-only). Concepts ported (Muon, WSD, scan/remat). |
| datasets | ✅ Already a real dependency (`omni_diffusion/data/build.py`). |
| huggingface_hub | ✅ Needed — add to requirements; use for `bapx/x8D-Omni-Diffusion` weight/tokenizer I/O. |
| gcsfs | ❌ Dead in needle, dead for us. |
| transformers | ✅ Already a dependency (config/tokenizer/trainer). |
| wandb | ⚠️ Optional, training logging only. |
| pyyaml | ⚠️ Our `configs/` uses yaml for training configs — **this one IS justified for us**, unlike needle. |
| sentencepiece | ❌ BANNED (byte law). |
| google-genai | ⚠️ Optional — synthetic SFT data generation. |

## 6. Verdict

We are **not missing** JAX/Flax/Optax/sentencepiece — those are the wrong layer for a
byte-native, PyTorch, dependency-free model. We are missing:

1. `huggingface_hub` (declared dependency) for model/tokenizer distribution.
2. `pyyaml` (justified — training configs).
3. **Algorithm ports**: Muon optimizer, WSD schedule, ZCRMSNorm, INT4/INT8 STE QAT,
   matryoshka FFN export, constrained byte decoding, contrastive byte retriever.

The single most important insight: needle does **QAT + export-to-tiny-format** exactly the
way our issues #3/#6 want, and its 14 MB edge footprint is the target our x8Dsub-byte 0.001
compression is designed to beat (98% reduction vs. BF16).

## 7. References

- GitHub: https://github.com/cactus-compute/needle (cloned, commit on main)
- HF: https://huggingface.co/Cactus-Compute/needle (config.json: NeedleForCausalLM,
  d_model 512, 8H/4KV, vocab 8192, tie_word_embeddings, is_encoder_decoder)
- README: Simple Attention Networks (arXiv 2607.18363), Cactus Compute
- Files analyzed: needle/model/{architecture,run,quantize,export,constrained}.py,
  needle/training/{train,optim,pretrain,finetune,eval}.py,
  needle/dataset/{tokenizer,dataset,generate,tokenize}.py, needle/utils/{distributed}.py,
  needle/ui/server.py, needle/cli.py, setup, pyproject.toml, requirements.txt
