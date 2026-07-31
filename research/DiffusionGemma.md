# DiffusionGemma — Research Notes for x8D-Omni-Diffusion

**Date:** 2026-07-31
**Status:** Initial research pass (web sources only; no official paper found yet)

## 1. TL;DR

DiffusionGemma is Google DeepMind's **non-autoregressive diffusion LM** for text, released
June 10 2026 under Apache 2.0. It abandons token-by-token autoregression and instead
denoises a full **256-token "canvas" in parallel**, producing ~4x faster generation than an
equivalent autoregressive model. It is built on **Gemma 4** (12B backbone scaled to a
26B MoE) and the "Gemini Diffusion" line of research.

- **HF model:** `google/diffusiongemma-26B-A4B-it` (26B total, 4B active — MoE)
- **Release window:** ~Jun 10 2026, after Gemma 4 12B
- **License:** Apache 2.0
- **MoE details:** Gemma 4 backbone — 128 fine-grained experts, **top-8 routing**,
  26B total / 4B active
- **Canvas:** 256 tokens per denoising block; up to **48 denoising steps per block**
- **Context length:** 262,144
- **Hardware:** 1000+ tok/s on H100; 700+ tok/s on RTX 5090; fits ~18GB VRAM when quantized
- **Vision encoder:** Gemma 4 vision tower, 27 layers, 1152 hidden, 16×16 patches,
  `use_bidirectional_attention: "vision"` (text stays causal)

## 2. Core Mechanism (uniform-state discrete diffusion)

Unlike our current DREAM-style **absorbing/masked** diffusion (MASK token = absorbing state),
DiffusionGemma uses **uniform-state diffusion**:

1. Start with a canvas of N tokens (default 256) filled with **random tokens** (noise).
2. Repeatedly feed the canvas through the model with **bidirectional attention**.
3. Each pass predicts better token probabilities for every position **in parallel**.
4. **Entropy-bounded sampling** decides which positions to commit vs. re-noise:
   - positions with low entropy (high confidence) are accepted;
   - positions with high entropy are re-noised with random tokens and retried.
5. After a fixed or adaptively-stopped number of denoising steps, the canvas is final.

Key consequence: **all token positions are available in every denoising step**, not
generated left-to-right. Information flows freely between positions across steps.

## 3. Architecture Innovations

### 3.1 Encoder–Denoiser patch (no cross-attention)
Single decoder-only Gemma 4 (26B-A4B MoE) switches between two modes:
- **Denoiser mode:** attends to the canvas with bidirectional attention; every token's
  logits are used (training objective is the full joint).
- **Encoder mode:** processes the user prompt/query with causal attention, then
  **shares its KV cache** with the denoiser via a small patch. No cross-attention layer
  is added.

### 3.2 Self-conditioning
At each denoising step, the previous step's probability output is carried forward:
- Take `softmax(logits) × embedding_matrix` (token-probability-weighted embedding),
- pass through a small FFNN,
- add it to the current step's input embeddings.

This gives the model memory of its own previous guesses, dramatically improving
stability across denoising iterations.

### 3.3 Multi-canvas (block) sampling
Generation is not limited to 256 tokens. Successive 256-token canvases are stitched
together:
- The encoder-mode KV cache is **reused/updated** between canvases.
- Canvas i+1 conditions on the finalized tokens of canvas i.
This is effectively a **semi-autoregressive block generation** at the canvas level.

### 3.4 Scheduler
- **Steps:** number of max denoising steps.
- **Logits scheduler:** temperature decreases over denoising steps (exploration early,
  exploitation late).
- **Adaptive stopping:** halt when (a) top predictions are identical for N consecutive
  steps (**stability**), and (b) confidence exceeds a threshold (**entropy < 0.005**).

### 3.5 Entropy-bounded sampler
- Canvas initialized with uniform random tokens.
- Positions sorted by entropy (lowest first = most confident).
- Accept tokens greedily while the running **sum of entropies** stays under a budget.
- Rejected positions are re-noised with random tokens for the next step.

### 3.6 vLLM configuration surface (from vLLM blog, 2026-06-10)
- `--diffusion-config '{"canvas_length": 256}'` — canvas block size.
- `--hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'`
  — sampler + entropy budget.
- `--max-num-seqs 4` — the diffusion state buffers
  (`self_conditioning_probs`) pre-allocate `max_seqs × canvas_length × vocab_size`,
  so with a 262K vocab, larger concurrency OOMs. Confirms the **self-conditioning**
  tensor layout (see §3.2).
- `--generation-config vllm` — checkpoint `generation_config.json` sets
  `max_tokens: 256`; flag ignores it so per-request limits work.
- Min vLLM version `0.24.0`, docker image `vllm/vllm-openai:gemma`.

## 4. Why it's faster

Autoregressive models are **memory-bandwidth-bound** (one token forward per step, weights
read per token). Diffusion parallelizes: **256 tokens per forward pass** with a single
weight read, making generation **compute-bound** instead. Result: 4x wall-clock speedup.

### 4.1 Measured vs autoregressive Gemma 4 26B-A4B (SPEED-Bench, H100, concurrency=1)

| Metric | Gemma 4 26B-A4B (AR baseline) | DiffusionGemma 26B-A4B |
|---|---|---|
| Output TPS | 199 tok/s | 375 tok/s (**1.9×**) |
| E2E request time (mean) | 2.87s | 0.88s (**3.3× faster**) |
| TTFT (mean) | 53ms | 489ms (higher — denoises whole canvas first) |
| Per-request gen TPS (mean) | 205 tok/s | 1,282 tok/s (**6.2×**) |

Trade-off: ~10× higher TTFT, but per-request generation throughput and E2E latency win
by wide margins.

## 5. How it relates to DREAM (our current base)

| Axis | DREAM (current base) | DiffusionGemma |
|---|---|---|
| Noise | absorbing/MASK token | uniform random tokens |
| Attention | decoder (bidirectional on canvas) | bidirectional denoiser + causal encoder patch |
| Self-cond | none | softmax×embedding → FFNN → add to inputs |
| Sampling | entropy / maskgit / origin | entropy-bounded with re-noise |
| Blocking | block-based sampling | 256-token canvases + KV reuse |
| Backbone | DREAM 7B / 14B | Gemma 4 12B → 26B-A4B MoE |

## 6. Caveats / missing pieces

- **No official code or paper URL yet found** (GitHub `google-deepmind/diffusiongemma` 404s;
  direct arXiv title search returned 0 hits as of 2026-07-31).
- Google's "Hackable Diffusion" is **JAX fine-tuning only** — not a train-from-scratch repo.
- `ItsSiddharth/Diffusion-Gemma-from-scratch` is educational PyTorch, **text-only, no
  training loop**.
- **No full train-from-scratch DiffusionGemma exists** in the wild — our DREAM-based fork
  remains the best base to build on.
- Known limitations (vLLM): audio **not supported** (no audio encoder in checkpoints);
  function-calling works best in thinking mode; `--max-num-seqs` must stay ≤4.
- Transparency analysis (arXiv 2606.20560, "How Transparent is DiffusionGemma?"):
  diffusion LMs do a large share of computation in **continuous latent space** and have
  **opaque serial depth ~28.6x higher** than autoregressive Gemma 4 — i.e. harder to
  interpret/align, and there are unresolved questions about single-step vs multi-step
  information flow.

## 7. Implications for x8D-Omni-Diffusion (byte-native)

Concrete, high-value items to borrow (in priority order):

1. **Uniform-state byte diffusion.** With a 256-byte vocabulary, uniform-state noise is
   *trivially* natural: re-noise rejected positions with a random byte 0–255. No special
   MASK state needed for the diffusion loop (MASK=256 still used for interface protocol).
2. **Self-conditioning.** Add `softmax(logits) × embed` → FFNN → add to embeddings in the
   denoising loop. Cheap, well-proven stability win.
3. **Adaptive stopping** (stability + confidence < 0.005). Extends our existing
   entropy-based samplers in `generation_utils.py`.
4. **Canvas/block stitcing with KV reuse.** We already have block-based sampling; make the
   block boundary KV-aware for long byte-streams (text, images, audio PCM).
5. **Entropy-bounded re-noise budget.** Port to byte domain: accept while cumulative
   entropy stays under budget; re-noise the rest with random bytes.

Not adopted: MoE scaling to 26B is post-Kim K3 insight (issue #5 already plans top-2 MoE);
KDA hybrid attention (issue #7) replaces the "patch" idea with native modality routing.

## 7b. Actual config.json breakdown (from HF repo)

`google/diffusiongemma-26B-A4B-it/config.json`, model_type `diffusion_gemma`,
architecture `DiffusionGemmaForBlockDiffusion`:

### Diffusion-level
- `canvas_length: 256`, `tie_word_embeddings: true`, dtype bfloat16

### text_config (Gemma 4 MoE backbone)
- `hidden_size: 2816`, `intermediate_size: 2112`, `moe_intermediate_size: 704`
  (experts are small — 704-wide FFN, 128 of them)
- `num_hidden_layers: 30`, `num_attention_heads: 16`, `head_dim: 256`
- `num_key_value_heads: 8`, `num_global_key_value_heads: 2` (only 2 full-attention KV
  heads — local/sliding layers use 8 KV heads, full layers use 2 global)
- `num_experts: 128`, `top_k_experts: 8`
- `layer_types`: sliding_attention ×6, full_attention ×1, repeating (30 layers,
  5 full-attention layers)
- `sliding_window: 1024`
- RoPE is **layer-type-dependent**:
  - full_attention: `rope_type: proportional`, `rope_theta: 1e6`,
    `partial_rotary_factor: 0.25`
  - sliding_attention: `rope_type: default`, `rope_theta: 10000`
- `final_logit_softcapping: 30.0`, `hidden_activation: gelu_pytorch_tanh`,
  `rms_norm_eps: 1e-6`, `max_position_embeddings: 262144`, `vocab_size: 262144`
- `use_bidirectional_attention: "vision"` (bidirectional only over vision region;
  text stays causal → confirms the encoder–denoiser switch in §3.1)

### vision_config (Gemma 4 vision encoder)
- `hidden_size: 1152`, `intermediate_size: 4304`, `num_hidden_layers: 27`,
  `num_attention_heads: 16`, `num_key_value_heads: 16`, `head_dim: 72`
- `patch_size: 16`, `pooling_kernel_size: 3`, `position_embedding_size: 10240`,
  `max_position_embeddings: 131072`, `rope_theta: 100`, `standardize: true`

### Special token IDs
- `boi_token_id: 255999` (begin of image)
- `eoi_token_id: 258882` (end of image)
- `image_token_id: 258880` (image patch placeholder)
- `eos_token_id: [1, 106]` (multi-stop)
- `vision_soft_tokens_per_image: 280`
- `bos_token_id: 2`, `pad_token_id: 0`

### Mapping to x8D byte-native IDs
DiffusionGemma allocates its multimodal specials near the end of a 262144 vocab.
x8D compresses this to **vocab 264** — 256 bytes + 8 specials (MASK=256, PAD=257,
BOS=258, EOS=259, IMG_START=260, IMG_END=261, AUD_START=262, AUD_END=263). The
`boi/eoi/image` role maps onto `IMG_START/IMG_END` and image bytes are the 256 raw
byte states; audio maps onto `AUD_START/AUD_END` (a modality DiffusionGemma lacks).
Equivalent "3 specials + patch bytes" cost is 260 ids vs their 262144 — 1000× smaller
vocab, which is the whole point of the byte law.

## 8. References

- DeepMind model page: https://deepmind.google/models/gemma/diffusiongemma/
- Google blog: "DiffusionGemma: 4x faster text generation"
- Visual guide (M. Grootendorst): "A Visual Guide to DiffusionGemma"
- HuggingFace: `google/diffusiongemma-26B-A4B-it`
  (https://huggingface.co/google/diffusiongemma-26B-A4B-it/tree/main)
- vLLM blog: "Diffusion Gemma" (2026-06-10):
  https://vllm-project.github.io/2026/06/10/diffusion-gemma.html
- arXiv 2606.20560 — "How Transparent is DiffusionGemma?"

## 9. Open questions for the x8D fork

1. Does DiffusionGemma's training objective also teach the model *which* positions are
   noisy (uniform noise is "self-masking" vs our explicit MASK state)? Directly informs
   whether we keep MASK=256 in the diffusion loop or move to pure byte re-noising.
2. Do we adopt the encoder–denoiser KV-sharing patch, or stay with our KDA modality
   routing (issue #7)? KV sharing is simpler; KDA is more expressive.
3. Self-conditioning buffer cost is `max_seqs × canvas × vocab` — with vocab=264 (byte
   native) this is ~1600x smaller than Gemma's 262K vocab, so **byte-native + 
   self-conditioning is uniquely cheap for us**.
4. vLLM min version 0.24.0 + `vllm/vllm-openai:gemma` required for serving — noted for
   the inference stack.
5. The 6:1 sliding:full attention pattern and the 2-global-KV-head trick are directly
   reusable as the KDA 3:1 hybrid (issue #7): 30 layers → 15 local byte-attention +
   10 global + 5 KDA fusion, keeping one shared hidden dim.
6. `final_logit_softcapping: 30` and layer-type-specific RoPE (theta 1e6/0.25 partial
   for full layers, theta 1e4 for sliding) should carry into `configuration_dream.py`.
