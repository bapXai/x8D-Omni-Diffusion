# MoE, Omni any-to-any Models, and Diffusion Language Modeling (issue #40)

Status: researched 2026-07-31 from primary sources (papers, model cards,
official blogs; URLs in §6). Goal: ground the byte-law stack
(vocab 264, `/0.001` QAT quantization, DSpark block speculation,
mmap'd on-disk experts) in the 2024–2026 MoE + omni + diffusion-LM wave, and
explain precisely WHY raw bytes replace torch/transformers tokenizer+float
paths for **byte manipulation** while the transformer forward/backward stays
in torch.

---

## 1. MoE routing theory → SARA / MoEOnDisk

### 1.1 The two questions every router answers

A Mixture-of-Experts layer answers two questions per token:

1. **Which experts?** — top-k selection over a learned gate.
   Mixtral (arXiv 2401.04088) routes each token through the top-2 of 8
   SwiGLU experts: `y = Σ_i Softmax(Top2(x·W_g))ᵢ · SwiGLUᵢ(x)`, giving 47B
   total params with only ~13B active per token. Increasing expert count N
   while holding k fixed grows total params with (almost) constant compute
   per token — the sparse-vs-active parameter distinction.
2. **How are the selected outputs combined?** — Mixtral renormalizes the
   top-k gate weights to sum to 1 and sums weighted expert outputs.

Variants along the same axis: Switch Transformers (arXiv 2101.03961)
collapse routing to **top-1** (k=1) and show it trains stably, and
Expert-Choice routing (Zhou et al., 2022) inverts the direction — each
expert picks its fixed-capacity batch of tokens, guaranteeing load balance
by construction at the cost of a token possibly being selected 0 or many
times.

### 1.2 Expert isolation & specialization

GShard/Mixtral-style routing lets experts overlap in the knowledge they
capture. **DeepSeekMoE** (arXiv 2401.06066, ACL 2024) fixes this with two
principles, both of which x8D's on-disk layout mirrors:

- **Fine-grained expert segmentation**: split N experts into mN smaller
  ones and activate mK of them; smaller experts stay more specialized and
  combine more flexibly (mN total, mK non-zero gates).
- **Shared expert isolation**: K_s shared experts are ALWAYS activated —
  they absorb the common knowledge so routed experts stay specialized.
  DeepSeekMoE keeps a 1:3 shared:routed ratio and shows that disabling the
  shared expert and adding one more routed expert instead RAISES Pile loss
  from 1.808 to 2.414 (the shared expert is irreplaceable).

DeepSeek-V3 follows up with **256+1 experts (1 shared), top-8 routing**, and
**auxiliary-loss-free load balancing**: each expert keeps a bias `b_i` used
ONLY for routing decisions — `i* = TopK(s(x)ᵢ + bᵢ)` — while the actual gate
weight is the renormalized softmax of the chosen top-k. Overloaded experts
have `b_i` decreased, underloaded ones increased, so the load stays balanced
without contaminating the training loss.

GLM-4.5 (arXiv 2508.06471) adds "loss-free balance routing" + sigmoid gates
on a 160+1-shared, top-8, 355B/32B MoE, and stacks an MoE layer as the MTP
(multi-token prediction) speculative-decoding head.

### 1.3 How SARA / MoEOnDisk map this onto byte-native on-disk experts

`omni_diffusion/moe_disk.py` implements the serving side of the same theory:

- **`SARABoundary` = one customer = one isolated expert block.** Dense
  models (Kokoro-82M, Whisper large-v3, LTX-2) register `mode="dense"` and
  are served as a single expert; internal-MoE models (GLM-5.2 753B,
  Kimi-K3 2.78T, DeepSeek-V4-Pro 1.6T) register `mode="moe"` and are their
  own expert pool. `SARA_REGISTRY` carries the researched active/total param
  counts (kimi-k3: 104.2B active / 2.78T total).
- **`SARARouter.route(modality)`** maps a query hint to exactly one boundary
  (text/language -> kimi-k3, image/video -> ltx2, audio/asr/speech ->
  whisper-large-v3, tts/voice -> kokoro-82m). `is_isolated(a, b)` is
  unconditionally True: routing to one boundary never mmaps or
  `/0.001`-reverses another customer's bytes.
- **`MoEOnDisk`** is the mechanical router: at query time it memory-maps
  the x8D `.gguf` container and calls `load_expert(layer, expert, proj)`,
  which slices ONLY that one tensor's byte span and applies the live
  `/0.001` reverse (a 256-entry LUT — no per-element float math). This is
  DeepSeek-style routing where the "experts" are byte spans on disk and
  "routing" is a key -> (repo|shard|offsets|shape) pointer lookup
  (`X8DPTR01`, `x8d_hf.py`). Only the routed span ever becomes the running
  state.

Issue #39's probe (`tools/omni_chat_probe.py`) ties the theory to the
endpoint: a text chat request routes to the `kimi-k3` boundary (104.2B
active params) while `MoEOnDisk` demonstrates on a synthetic container that
exactly 384 B (192 routed + 192 shared) are `/0.001`-reversed, byte-exactly.

---

## 2. Omni any-to-any models → x8D's byte space

### 2.1 The two architectural camps

**Camp A — bolt-on encoders (understanding-only).** GLM-4.5V (arXiv
2507.01006) is a ViT encoder + MLP projector + LLM decoder: images/videos
become patch features projected into token space, the LLM consumes them and
emits text (+ bounding-box tokens). LTX-2 (arXiv 2601.03233) is the
generative mirror image: an asymmetric dual-stream DiT (14B video + 5B
audio streams) with bidirectional cross-attention and modality-aware CFG —
but everything happens in **latent space** (separate video/audio VAEs), so
there is no shared discrete vocabulary at all.

**Camp B — one shared token space (any-to-any).** GPT-4o (OpenAI, May 2024)
was the first production omni model: one network trained end-to-end across
text/vision/audio that accepts any combination of text/audio/image/video
and generates any combination of text/audio/image. Google's Gemini Omni
(May 2026) frames the mechanism explicitly: every modality is converted at
the door into the **same kind of token** (text via tokenization, images via
patch->codebook quantization, audio/video likewise) so a single transformer
reads and emits all of them. NExT-OMNI (arXiv 2510.13721) pushes this to a
fully open, unified DFM (discrete-flow-matching) omnimodal model with one
encoder for understanding+generation and lightweight per-modality decode
heads.

### 2.2 What x8D does differently — one vocab, 264 ids

Every Camp B model still needs *per-modality codecs*: a text tokenizer
(50k–262k ids), a VQ-VAE codebook for images, another for audio — three
vocabularies that must be kept aligned. The byte law collapses all of them:

| Modality | Camp B (GPT-4o/Gemini Omni/NExT-OMNI) | x8D |
|---|---|---|
| text | BPE/SP/WordPiece ids (50k–262k) | UTF-8 bytes → ids 0–255 |
| image | patch → VQ codebook ids | raw pixel bytes → ids 0–255, wrapped `IMG_START/IMG_END` (260/261) |
| audio | audio codec ids | raw PCM bytes → ids 0–255, wrapped `AUD_START/AUD_END` (262/263) |
| video | frame → codebook ids | frame bytes → ids 0–255 |
| vocab | 3+ disjoint vocabularies + alignment glue | **one** vocab: 256 bytes + 8 specials = 264 |

Text, image, audio, video, code, and binaries all reduce to the same
`list(data_bytes)`; the "conversion at the door" is the byte boundary, not a
learned codebook. DiffusionGemma still needs a 262,144-id tokenizer for
text and patch codebook ids for its vision tower; x8D needs 264 ids for
everything. This is exactly the unification (§4e) that Camp B wants, minus
the tokenizers.

---

## 3. Diffusion language modeling — DiffusionGemma as the proof

DiffusionGemma (Google DeepMind, 2026-06-10, Apache 2.0,
`google/diffusiongemma-26B-A4B-it`) is the existence proof that **diffusion
applies to language**, not just pixels/audio. Verified facts (model card +
vLLM/Google docs):

- **Uniform-state discrete diffusion**: a 256-token canvas starts as random
  tokens (noise) and is denoised with bidirectional attention; every
  position is available in every step (vs left-to-right AR).
- **Entropy-bounded sampler**: at each step the sampler accepts the
  lowest-entropy tokens such that `Σᵢ entropyᵢ − max(entropy₁..entropyₖ)
  ≤ entropy_bound = 0.1`; the rest are fully **re-noised** with random
  tokens. 48 max denoising steps; temperature decays 0.8 → 0.4.
- **Adaptive stopping**: halt only when (a) average canvas entropy < 0.005
  AND (b) the top predictions are identical across two consecutive steps.
- **Block-autoregressive multi-canvas commit**: once a 256-token canvas is
  denoised it is appended to the KV cache, the encoder re-runs, and the
  next canvas diffuses — diffusion on the inside, autoregression across
  canvases.
- **Self-conditioning**: `softmax(logits) × embedding → FFNN` is added to
  the next step's input embeddings (stability across steps).
- **Speed**: parallel 256-token denoising shifts generation from
  memory-bound to compute-bound — >1,000 tok/s on H100 (15–20 tokens per
  forward pass), 700+ tok/s on RTX 5090, ~4× faster than AR Gemma-4.
- **MoE backbone**: Gemma 4, 25.2B total / 3.8B active, 128 routed + 1
  shared expert, top-8, canvas_length=256, vocab 262,144, context 256K,
  text + image.

### Why language-as-diffusion fits x8D's byte space perfectly

- **Uniform noise is trivial over bytes**: "re-noise with a random token"
  is `renoise_to_random_bytes`: sample 0–255. MASK=256 stays only as the
  interface protocol, exactly as DiffusionGemma's uniform-state scheme.
- **The 264-vocab is the cheapest canvas possible**: the self-conditioning
  buffer is `max_seqs × canvas × vocab`; with vocab 264 that is ~1,000×
  smaller than DiffusionGemma's 262K (already computed in
  `research/DiffusionGemma.md` §7b).
- **The sampler contract is already mirrored**: `generation_utils._sample`
  entropy-bound hook (budget 0.1), `canvas_length=256`, `max_denoising_steps
  =48`, `config_dream_resume.json` — all byte-native.
- **DSpark block speculation extends the canvas idea**: instead of a single
  full 256-canvas forward, `x8d_spec_decode.py` drafts/verifies 8×8 byte
  blocks with a confidence head and re-masks positions below the 0.001
  threshold — the same accept/re-noise loop at block granularity, and the
  same confidence-head abstraction DiffusionGemma's entropy sampler uses for
  token acceptance.

---

## 4. The byte-law justification — why bytes REPLACE torch/transformers for byte manipulation

This is the load-bearing section. It is about **byte manipulation**, not
model math: encoding/decoding, quantization, storage, routing, LUT reversal,
block speculation. torch/transformers are still the training/inference
engine (§4.6). The claim is that every byte-plane operation below is
strictly better in raw integers than in torch-float-with-a-tokenizer.

### 4.1 (a) No tokenizer vocabulary to store or align

Every tokenizer ships artifacts: `vocab.json`, `merges.txt`,
`added_tokens.json`, special-token maps. They must be versioned, aligned
with the model (a vocab mismatch breaks loading), and extended per
modality/domain (Gemma adds `boi/eoi/image` ids; DiffusionGemma reserves 8
ids near the end of 262,144 for multimodal markers). Byte-native has
**nothing to store or align**: the vocabulary is the 256 unsigned byte
states of every CPU/GPU/memory bus, defined by hardware, not by a training
run. `ByteTokenizer.encode` is literally `list(data_bytes)` — no lookup, no
merge rules, no OOV, no id>255. All 264 ids (256 bytes + 8 specials) are
fixed by the byte law in `AGENTS.md`.

### 4.2 (b) Byte ops are integer LUT lookups, not float matmuls

- Encode/decode: `bytes → ints` and `ints → bytes` — pure integer
  identity, zero floating point.
- The `/0.001` QAT reverse: because quanta are exactly
  `b * 0.001` for integer `b` in 0–255, the inverse `round(q / 0.001)` is
  a 256-entry **LUT** (`_REVERSE_LUT` in `moe_disk.py`; `_dequantized`
  memoization in `x8d_export.py`), not per-element float division.
- DSpark block speculation: `sha256`-seeded pseudo-logits and byte mixing
  are integer/hash operations (a real confidence head will be torch, but
  the byte-packing/verification stays integer).
- Container I/O: `X8DGgufReader` slices bytes out of an `mmap`; the tensor
  index is walked with `struct` — no torch tensors are constructed for
  addressing.
These are the operations measured in `research/Byte-Core-Optimizations.md`
(6–41× speedups) and `Frontier-Benchmarks-2026.md` (56 ms to load_expert a
5.5 MB expert; 96 ms to spec-quantize 1 MB). Running them in torch would
mean: constructing tensors, materializing float ops for what is a table
lookup, and paying kernel-launch overhead — strictly slower and it drags in
a GPU dependency for non-GPU work.

### 4.3 (c) The 0.001 reverse is EXACT — no FP precision drift

QAT's usual objection is round-trip error: `round(float(x))` can drift.
Here it cannot:

```
stored quanta:   q = b * 0.001          (b ∈ {0..255} integer)
reverse:         b' = round(q / 0.001) & 0xFF
=> b' == b for every b ∈ 0..255         (verified: _REVERSE_LUT, byte-exact)
```

Because `0.001` and `0.001` are the SAME float constants in both directions
and `b` is a small integer, `round((b*0.001)/0.001)` returns exactly `b`
for the full 0–255 range — no float noise, no accumulation, no drift. The
probe verifies it live (`reverse_exact: True`) and
`test_pointer_quantize.py::test_hf_vs_compressed_forward_identical` proves a
real 5.5 MB expert forward is bit-identical (`maxdiff=0.0`). A float path
has no such guarantee; it also multiplies storage (BF16/FP32) with no
information gain over the 256 states the hardware already natively handles.

### 4.4 (d) Compressed state IS the running state — zero RAM residency

torch serving loads weights into RAM/VRAM once and keeps them resident;
tensor memory *is* the model. Byte-native serving inverts this: the
`X8DGGUF1` U8 container (or `X8DPTR01` pointer map) is `mmap`'d, and only
the specific routed expert's span is sliced and `/0.001`-reversed at query
time (`MoEOnDisk.load_expert`, `serve_expert_from_pointer()`). No
decompression loop, no residency. Consequences:

- A 1.56 TB FP16 Kimi-K3 becomes a 2.837 GB sub-byte coordinate map
  (~550:1, issue #10) or a 151.8 MB pointer map whose weight bytes stay on
  the upstream HF disk — served from a single commodity host.
- Per-token cost = only the routed experts' spans: ~16 active experts ×
  0.008 bit/param ≈ 50 MB/token of actual compute bytes (the Frontier
  Benchmarks math), i.e. the memory-bound cost of a mid-size dense model.
- `matmul_fp32` on the reversed slice is a CPU-only demonstration that a
  real query-time forward can run without the model ever being resident.

### 4.5 (e) Multimodal unification: one vocab, one pipeline

Byte-native collapses GPT-4o/Gemini-Omni/NExT-OMNI's multiple codecs into a
single vocab (§2.2): text (UTF-8), image (pixel bytes), audio (PCM bytes),
video (frame bytes), code, binaries — all ids 0–255 with
`IMG_START/IMG_END` and `AUD_START/AUD_END` markers. The byte-level
literature backs the modeling side: MambaByte (arXiv 2401.13660) shows
token-free SSMs are competitive with subword Transformers while being
robust to typos/corruption and generalizing across orthographic variants,
and its speculative decoding (subword draft + **byte-level verification**)
is the exact draft-verify shape x8D's DSpark blocks use; MegaByte (arXiv
2305.07185) shows byte-level transformers match subword models on
long-context LM, ImageNet density estimation, AND raw audio — all three
from the same 256-id byte vocab, with tokenization itself described as
"complicating pre-processing, multi-modal modelling, and transfer to new
domains while hiding useful structure."

### 4.6 What torch IS still needed for (the honest boundary)

The byte core (tokenizer, containers, LUT quantizers, spec-decode packing,
on-disk routing, dataset import) is pure stdlib and must STAY stdlib — that
is the point of `requirements_core.txt`. But the **DreamModel transformer
forward/backward is a float neural net and torch is its engine**: embedding
matmul, attention (QKV + RoPE + softmax + KV cache), MLP/SwiGLU, the
`lm_head` projection to 264, the cross-entropy over the byte vocab, and the
entire backward pass. Those are dense float matmuls and tensor ops — not
byte LUT lookups — so they stay in torch (training stack in
`requirements_ds_gpu.txt`). Issue #34 (trainable checkpoint + first forward)
and #7 (`kda_attention.py`/`dspark_diffusion.py`, plus Block AttnRes from
#24) are the torch-side work items.

The division of labor, in one line: **bytes handle what data IS
(identity/coordinates/storage); torch handles what the model LEARNS
(weighted combinations of bytes).**

---

## 5. x8D implementation map

| Concept | Theory source | x8D implementation |
|---|---|---|
| Top-k routing / active params | Mixtral, Switch, DeepSeekMoE | `SARARouter.route()` → boundary active/total params |
| Expert isolation | DeepSeekMoE shared/routed split | `SARABoundary` per customer; `is_isolated()` always True |
| Load balancing | Switch aux loss; DeepSeek-V3 bias routing | n/a at serving; routing biases live in upstream weights (pointer map) |
| Shared experts | DeepSeekMoE K_s, DeepSeek-V3 1 shared | `SHARED_EXPERT` block, always active (#39 probe) |
| On-disk expert serving | — | `MoEOnDisk` mmap + `/0.001` LUT reverse, `X8DPTR01` |
| Any-to-any omni | GPT-4o, Gemini Omni, NExT-OMNI | one 264-id vocab; IMG/AUD markers |
| Diffusion LM | DiffusionGemma | `generation_utils._sample` entropy-bound, canvas 256, steps 48 |
| Block speculation | DiffusionGemma multi-canvas; MambaByte draft/verify; MTP | DSpark 8×8 block draft+verify (`x8d_spec_decode.py`) |

---

## 6. Sources

- Mixtral of Experts — arXiv:2401.04088 — https://arxiv.org/pdf/2401.04088
- Switch Transformers — arXiv:2101.03961 — https://arxiv.org/abs/2101.03961
- Mixture-of-Experts with Expert Choice Routing (Zhou et al., 2022) —
  NeurIPS 35.
- DeepSeekMoE — arXiv:2401.06066 (ACL 2024) —
  https://arxiv.org/html/2401.06066 ; https://aclanthology.org/2024.acl-long.70/
- DeepSeek-V2/V3 fine-grained experts + auxiliary-loss-free balance —
  https://arxiv.org/abs/2405.04434 ; https://arxiv.org/abs/2412.19437
- GLM-4.5 / GLM-4.5-Air — arXiv:2508.06471 — https://arxiv.org/pdf/2508.06471
- GLM-4.5V / GLM-4.1V-Thinking — arXiv:2507.01006 —
  https://arxiv.org/html/2507.01006v5 ; https://huggingface.co/zai-org/GLM-4.5V
- NExT-OMNI: Any-to-Any Omnimodal Foundation Models — arXiv:2510.13721 —
  https://arxiv.org/html/2510.13721v1
- GPT-4o announcement + System Card —
  https://openai.com/index/hello-gpt-4o/ ; https://arxiv.org/html/2410.21276
- Gemini Omni / shared token space explainer —
  https://learnaivisually.com/ai-explained/gemini-omni-shared-token-space
- DiffusionGemma — model card: https://ai.google.dev/gemma/docs/diffusiongemma/model_card ;
  overview: https://ai.google.dev/gemma/docs/diffusiongemma ;
  explained: https://ai.google.dev/gemma/docs/diffusiongemma/explained ;
  HF: https://huggingface.co/google/diffusiongemma-26B-A4B-it ;
  transformers docs: https://huggingface.co/docs/transformers/en/model_doc/diffusion_gemma ;
  NVIDIA blog: https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/ ;
  vLLM blog: https://vllm-project.github.io/2026/06/10/diffusion-gemma.html
- LTX-2 — arXiv:2601.03233 — https://arxiv.org/pdf/2601.03233 ;
  https://github.com/Lightricks/LTX-2
- MambaByte — arXiv:2401.13660 — https://arxiv.org/html/2401.13660v2
- MEGABYTE — arXiv:2305.07185 — https://arxiv.org/abs/2305.07185
- MoE deep-dive (routing strategy comparison table) —
  https://www.youngju.dev/blog/ai-papers/2026-03-10-mixture-of-experts-moe-architecture-routing.en
- HuggingFace transformers Mixtral `load_balancing_loss_func` (Switch eq.
  4–6) — https://github.com/huggingface/transformers/blob/v4.37.1/src/transformers/models/mixtral/modeling_mixtral.py
- Repo-internal: `omni_diffusion/moe_disk.py` (SARABoundary/SARA_REGISTRY/
  SARARouter/MoEOnDisk), `omni_diffusion/x8d_export.py`, `omni_diffusion/x8d_hf.py`,
  `omni_diffusion/x8d_spec_decode.py`, `tools/omni_chat_probe.py`,
  `research/{DiffusionGemma,Depth-Context-Attention-Frameworks-2026,Frontier-Benchmarks-2026,Omni-Endpoint-and-Experts-2026}.md`.
