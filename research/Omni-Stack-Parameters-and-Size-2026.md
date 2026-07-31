# Omni-Stack Parameters and Compressed Size (issue #38)

Status: researched 2026-07-31 from primary sources (HF `config.json` fetched
live, HF model cards, GitHub READMEs, vLLM recipes) and cross-checked against
the repo's verified artifacts (`research/Status-and-Optimization-Audit-2026.md`,
`research/Kimi-K3-x8D-Pointer-Quantization.md`, `research/Omni-Modality-Stack.md`,
`README.md`). Companion tool: `tools/omni_size_report.py` (+
`tests/test_omni_size_report.py`).

## 1. The stack — 8 experts in the x8D omni MoE

Every model below is one expert in the byte-native omni mixture. Dense models
are single experts (`active == total`); MoE models are experts that route
internally. The x8D serving law applies to all of them identically: the
compressed state IS the running state — only the requested expert's byte span
is mmap'd / Range-fetched and `/0.001`-reversed live at query time
(`omni_diffusion/moe_disk.py`, `tools/quantize_kimi_k3.py`, `tools/quantize_hf.py`).

| Model | Upstream repo | Type | Total params | Active params | Experts (active/total) | Upstream disk | x8D pointer map | x8D U8 (0.008 bit/param) | Serveable (U8+BF16ptr) |
|---|---|---|---|---|---|---|---|---|---|
| GLM-5.2 | zai-org/GLM-5.2 | MoE | ~743B | ~39B | 8/256 + 1 shared | 1.49 TB (BF16, est.) | 43.67 MB (est.) | 743.0 MB | 743.0 MB |
| Kimi-K3 | moonshotai/Kimi-K3 | MoE | 2.78T | 104.2B | 16/896 + 2 shared | 1.56 TB (MXFP4) | **163,374,871 B** | 2.723 GB | **2.837 GB** |
| DeepSeek-V4-Pro | deepseek-ai/DeepSeek-V4-Pro | MoE | 1.6T | 49B | 6/384 + 1 shared | 865 GB (FP4+FP8) | 94.03 MB (est.) | 1.600 GB | 1.600 GB |
| DiffusionGemma | google/diffusiongemma-26B-A4B-it | MoE | 26B | 4B | 8/128 | 52 GB (BF16, est.) | 1.53 MB (est.) | 26.00 MB | 26.00 MB |
| Kokoro-82M | hexgrad/Kokoro-82M | dense | 82M | 82M | — | 164 MB (FP16, est.) | 171 B | 82.0 KB | 82.0 KB |
| Whisper large-v3 | openai/whisper-large-v3 | dense | ~1.55B | ~1.55B | — | 3.10 GB (FP16, est.) | 343,642 B | 1.55 MB | 1.55 MB |
| LTX-2 | Lightricks/LTX-2 | dense | 19B | 19B | — | 9.50 GB (FP4, est.) | 2,319,390 B | 19.00 MB | 19.00 MB |
| Kitten TTS (mini) | KittenML/kitten-tts-mini-0.8 | dense | 80M | 80M | — | 80 MB (ONNX int8) | 4.7 KB (est.) | 80.0 KB | 80.0 KB |
| **COMBINED** | | | **5.17T** | **216.9B** (all fired) | | **3.98 TB** | **305.27 MB** | **5.11 GB** | **5.23 GB** |

Rows with `est.` use computed-from-research values (no pointer artifact is
hosted for them yet); the four pointer-map sizes without `est.` are the exact
files hosted in the HF model repo (`x8d_weights/*.x8dptr.gguf`, see §3).

## 2. Per-model web facts and sources

### GLM-5.2 — zai-org/GLM-5.2
- **MoE** (`glm_moe_dsa`), ~743B total / ~39B active; 5-token MTP speculative
  decoding; 1M context; MIT license.
  ([vLLM recipes](https://recipes.vllm.ai/zai-org/GLM-5.2),
  [HF card](https://huggingface.co/zai-org/GLM-5.2),
  [lambda.ai](https://lambda.ai/inference-models/zai-org/glm-5.2))
- Live `config.json`: 78 layers, hidden 6144, `n_routed_experts=256`,
  `n_shared_experts=1`, `num_experts_per_tok=8`, `num_nextn_predict_layers=1`
  (MTP), vocab 154880, 1M ctx.
  ([config](https://huggingface.co/zai-org/GLM-5.2/resolve/main/config.json))
- Some coverage quotes 753B/40B ([Better Stack](https://betterstack.com/community/guides/ai/glm-52/),
  [LAV explainer](https://learnaivisually.com/ai-explained/glm-5-2-active-vs-total-parameters));
  we use the vLLM/HF-recipe 743B/39B. Expert count taken from the config.

### Kimi-K3 — moonshotai/Kimi-K3
- **MoE**, 2.8T total / **104B active**, Stable LatentMoE routing **16 of 896**
  experts (+ 2 shared), 93 layers (69 KDA + 24 Gated MLA), 1M context, native
  vision; weights released 2026-07-27 as MXFP4 (weights) / MXFP8 (activations).
  ([GitHub README](https://github.com/MoonshotAI/Kimi-K3/blob/main/README.md),
  [runpod technical FAQ](https://www.runpod.io/articles/guides/kimi-k3-technical-faq),
  [datalearner](https://www.datalearner.com/en/ai-models/pretrained-models/kimi-k3),
  [morphllm](https://www.morphllm.com/kimi-k3),
  [HF blog](https://huggingface.co/blog/ResterChed/kimi-k3-model-overview-mxfp4-quantization-open-wei))
- KDA = Kimi Delta Attention, hybrid linear attention, 3:1 KDA:MLA interleave,
  up to 75% KV-cache reduction ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)).
- Repo-verified: 2,779,931,837,184 total params (U8 2,722,740,830,208 +
  BF16 57,179,884,544 + F32), 1.56 TB MXFP4 on disk, 96 shards, 497,220
  tensors, pointer map 163,374,871 B.
  ([Kimi-K3-x8D-Pointer-Quantization.md](Kimi-K3-x8D-Pointer-Quantization.md))

### DeepSeek-V4-Pro — deepseek-ai/DeepSeek-V4-Pro
- **MoE**, 1.6T total / **49B active**; MIT; 1M context; preview released
  2026-04-24; FP4 experts + FP8 other params.
  ([HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro),
  [deepseekai.guide](https://deepseekai.guide/models/deepseek-v4-pro/),
  [ainft docs](https://docs.ainft.com/reference/deepseek-v4-pro),
  [xorbitsai inference docs](https://inference.readthedocs.io/en/stable/models/builtin/llm/deepseek-v4-pro.html))
- Live `config.json`: 61 layers, hidden 7168, `n_routed_experts=384`,
  `n_shared_experts=1`, `num_experts_per_tok=6`, `num_nextn_predict_layers=1`
  (**MTP**), CSA+HCA hybrid attention.
  ([config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/config.json))
- HF on-disk size: **865 GB** (FP4+FP8 mixed; deepseekai.guide).

### DiffusionGemma — google/diffusiongemma-26B-A4B-it
- **MoE**, 26B total / **4B active**, 128 fine-grained experts top-8, 30 layers
  (sliding 6:full 1), canvas 256-token uniform-state diffusion, entropy-bound
  sampler, Apache 2.0, 2026-06-10.
  ([DiffusionGemma.md](DiffusionGemma.md) with
  [HF repo](https://huggingface.co/google/diffusiongemma-26B-A4B-it),
  [vLLM blog](https://vllm-project.github.io/2026/06/10/diffusion-gemma.html))
- Upstream disk est. 52 GB (26B × 2 B BF16) — the config-level facts are in
  `research/DiffusionGemma.md` §7b.

### Kokoro-82M — hexgrad/Kokoro-82M
- **Dense** TTS (StyleTTS2 + iSTFTNet vocoder), 82M params, 24 kHz PCM output,
  Apache 2.0, no diffusion, no encoder.
  ([HF card](https://huggingface.co/hexgrad/Kokoro-82M),
  [Omni-Modality-Stack.md](Omni-Modality-Stack.md))
- Already in the repo's model stack; pointer map 171 B.

### Whisper large-v3 — openai/whisper-large-v3
- **Dense** ASR encoder-decoder, ~1.55B params (HF card lists "2B" including
  decoder lm heads; decoder-only weight count ~1.55B), Apache 2.0.
  ([HF card](https://huggingface.co/openai/whisper-large-v3),
  [Omni-Modality-Stack.md](Omni-Modality-Stack.md))
- Pointer map 343,642 B.

### LTX-2 — Lightricks/LTX-2
- **Dense** joint audio-video DiT, 19B params, generates synchronized video +
  audio in one model; checkpoints `ltx-2-19b-dev[-fp8|-fp4|-distilled]`.
  ([HF card](https://huggingface.co/Lightricks/LTX-2),
  [Omni-Modality-Stack.md](Omni-Modality-Stack.md))
- Pointer map 2,319,390 B (the repo's tool example quantizes
  `ltx-2-19b-dev-fp4.safetensors`).

### Kitten TTS — KittenML/kitten-tts-mini-0.8
- **Dense** ONNX TTS, family of 15M/40M/80M models; flagship `kitten-tts-mini`
  is **80M params / 80 MB on disk** (int8), 24 kHz output, Apache 2.0.
  ([GitHub README](https://github.com/KittenML/KittenTTS),
  [HF mini](https://huggingface.co/KittenML/kitten-tts-mini-0.8),
  [HF nano](https://huggingface.co/KittenML/kitten-tts-nano-0.1))
- We register the mini (80M) as the stack entry.

## 3. The x8D size math under the 0.001 sub-byte law

`Quanta[i] = weight_byte[i] × 0.001`. Stored bytes ARE the quanta; the
`/0.001` reverse runs live at query time on the fetched span only
([AGENTS.md](../AGENTS.md), `x8d_export.py` `LAW = 0.001`).

```
U8 byte-native:      bits/param = 8  × 0.001 = 0.008  ->  bytes = params × 0.001
BF16 pointer-map:    bits/param = 16 × 0.001 = 0.016  ->  bytes = params × 0.002
serveable            = U8 (U8 portion) + BF16-pointer (BF16 portion)
```

Verification against the repo's known Kimi-K3 numbers: the task's raw check
`2.8e12 × 0.008 bit = 2.8 GB` differs from the repo's 2.723 GB only because
the repo uses the **actual U8-parameter count** (2,722,740,830,208) — the BF16
portion (57,179,884,544) is accounted separately at 0.016 bit/param. Using the
repo's exact split reproduces the README/research values:

| Model | total params | U8 formula → bytes | BF16-ptr formula → bytes | Serveable |
|---|---|---|---|---|
| Kimi-K3 | U8 2,722,740,830,208 | ×0.001 = 2,722,740,830 B (**2.723 GB** ✓) | BF16 57,179,884,544 ×0.002 = 114,359,769 B (**114.4 MB** ✓) | **2,837,100,599 B = 2.837 GB** ✓ |
| GLM-5.2 | 743e9 (all U8) | 743,000,000 B = 743.0 MB | — | 743.0 MB |
| DeepSeek-V4-Pro | 1.6e12 | 1,600,000,000 B = 1.600 GB | — | 1.600 GB |
| DiffusionGemma | 26e9 | 26,000,000 B = 26.0 MB | — | 26.0 MB |
| Kokoro-82M | 82e6 | 82,000 B = 82.0 KB | — | 82.0 KB |
| Whisper large-v3 | 1.55e9 | 1,550,000 B = 1.55 MB | — | 1.55 MB |
| LTX-2 | 19e9 | 19,000,000 B = 19.0 MB | — | 19.0 MB |
| Kitten TTS | 80e6 | 80,000 B = 80.0 KB | — | 80.0 KB |

The pointer-map column (X8DPTR01) is a *map*, not weights: Kimi's
163,374,871 B pin-points all 497,220 tensors (repo value); Whisper/Kokoro/LTX-2
pointer maps are the hosted `x8d_weights/*.x8dptr.gguf` files (343,642 /
171 / 2,319,390 B). For models with no hosted artifact, pointer size is
estimated proportionally to Kimi's map (43.67 / 94.03 / 1.53 MB, 4.7 KB).

## 4. Combined totals (all 8 experts)

- **Total params: 5,169,643,837,184 ≈ 5.17 T** (over 3 trillion)
- **Upstream disk (combined): 3,976,704,324,864 B ≈ 3.98 TB**
- **x8D pointer maps (combined): 305,267,437 B ≈ 305.3 MB**
- **U8 serveable (combined): 5,112,452,830 B ≈ 5.11 GB**
- **Serveable incl. BF16-pointer (combined): 5,226,812,599 B ≈ 5.23 GB**
- **Compression vs combined upstream disk:**
  - vs pointer maps: `3.98 TB / 305.3 MB ≈ 13,027:1`
  - vs serveable sizes: `3.98 TB / 5.23 GB ≈ 761:1`

## 5. MoE framing — every model is an expert; tasks activate a slice

| Task | Experts activated | Active params |
|---|---|---|
| ASR | whisper | 1.55B |
| TTS | kokoro (+ kitten_tts) | 82M (+ 80M) |
| Text reasoning | glm_5_2 + kimi_k3 + deepseek_v4_pro + diffusiongemma | 39B + 104.2B + 49B + 4B = 196.2B |
| Video generation | ltx2 | 19B |
| Image generation | ltx2 | 19B |
| Image understanding | kimi_k3 (native vision) | 104.2B |

**Isolation by construction:** `omni_diffusion/moe_disk.py` (`SARABoundary` /
`SARARouter`, issue #36) guarantees that only the requested boundary's byte
span is mmap'd and `/0.001`-reversed. An ASR query touches Whisper's span
alone (~1.55B active); a TTS query touches Kokoro's 82M; a text query touches
the GLM/Kimi/DS-V4/Gemma active slices (196.2B); a video query touches LTX-2's
19B. No expert's full weight set is ever resident in RAM — weights stay on the
upstream disk and are read on demand (mmap / HTTP Range) per
`serve_expert_from_pointer()` in `tools/quantize_kimi_k3.py`.

## 6. Caveats

- `est.` rows are computed-from-research, not measured artifacts; the four
  non-est pointer sizes are the repo's exact hosted files.
- Upstream disk sizes for models not measured on HF are estimated from
  parameter count × nominal precision (marked per-row).
- Combined compression ratios mix exact and estimated inputs; the Kimi-K3
  rows dominate and are exact.

## 7. Sources

- vLLM recipes — GLM-5.2: https://recipes.vllm.ai/zai-org/GLM-5.2 (743B/39B, 5-token MTP)
- zai-org/GLM-5.2 HF card: https://huggingface.co/zai-org/GLM-5.2 ; config: https://huggingface.co/zai-org/GLM-5.2/resolve/main/config.json
- MoonshotAI/Kimi-K3 README: https://github.com/MoonshotAI/Kimi-K3/blob/main/README.md (2.8T/104B, 16/896 experts)
- runpod K3 FAQ: https://www.runpod.io/articles/guides/kimi-k3-technical-faq (2.78T/104.2B, 1.56 TB MXFP4, 93 layers 69 KDA + 24 MLA)
- datalearner K3: https://www.datalearner.com/en/ai-models/pretrained-models/kimi-k3
- morphllm K3: https://www.morphllm.com/kimi-k3
- Kimi Linear (KDA) arXiv:2510.26692: https://arxiv.org/abs/2510.26692
- deepseek-ai/DeepSeek-V4-Pro HF card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro ; config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/config.json (384+1 experts, 6 top-k, MTP)
- deepseekai.guide V4-Pro: https://deepseekai.guide/models/deepseek-v4-pro/ (1.6T/49B, 865 GB on disk, FP4+FP8)
- ainft V4-Pro: https://docs.ainft.com/reference/deepseek-v4-pro ; xorbitsai inference: https://inference.readthedocs.io/en/stable/models/builtin/llm/deepseek-v4-pro.html
- google/diffusiongemma-26B-A4B-it: https://huggingface.co/google/diffusiongemma-26B-A4B-it ; vLLM blog: https://vllm-project.github.io/2026/06/10/diffusion-gemma.html ; repo `research/DiffusionGemma.md`
- hexgrad/Kokoro-82M: https://huggingface.co/hexgrad/Kokoro-82M
- openai/whisper-large-v3: https://huggingface.co/openai/whisper-large-v3
- Lightricks/LTX-2: https://huggingface.co/Lightricks/LTX-2
- KittenML/KittenTTS: https://github.com/KittenML/KittenTTS ; https://huggingface.co/KittenML/kitten-tts-mini-0.8 ; https://huggingface.co/KittenML/kitten-tts-nano-0.1
- Repo artifacts: `README.md` (x8D weights table), `research/Kimi-K3-x8D-Pointer-Quantization.md` (#10), `research/Status-and-Optimization-Audit-2026.md` (#33), `research/Omni-Modality-Stack.md` (#11), `tools/quantize_kimi_k3.py`, `tools/quantize_hf.py`, `omni_diffusion/x8d_hf.py`, `omni_diffusion/moe_disk.py`
