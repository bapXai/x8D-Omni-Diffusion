# vLLM-Omni Gap Analysis (2026-07-31, #46)

**Source audited:** `vllm-project/vllm-omni` @ 0.24/0.25 line
(APACHE-2.0), cloned to `/var/folders/14/x5vnmhdx2vx8n13bvs4wjvbc0000gn/T/opencode/vllm-omni`.
**Scope:** learn what x8D is missing for **omni routing, multi-modal UI, MTP**,
etc. — but keep it **byte-native only**, for **our selected models**, at
**0.001 sub-byte compression**, in **pure Python**.

**Ground rule (byte law):** this is NOT a port of vLLM-Omni. It is a
*capability map*. We take only the concepts that survive the byte law:
protocol shapes, stage-chain routing, block-predictor (MTP) structure, and
delta/streaming state. We never import a tokenizer, never store float
weights, never materialize tensors in RAM.

---

## 1. What vLLM-Omni actually is

A vLLM fork (~1123 Python files, **zero** C++/CUDA of its own — kernels come
from upstream `vllm` + torch) that extends vLLM to serve omni-modality and
non-autoregressive (diffusion) models: text, image, video, audio, action.

| Subsystem | Where | What it does |
|---|---|---|
| Omni routing / staging | `engine/orchestrator.py`, `engine/stage_pool.py`, `core/sched/`, `model_executor/stage_input_processors/` | `omni_task` modality switch + stage-chain pipelines (AR→DiT, thinker→talker→codec) + replica pools + inter-stage payload schema (`OmniPayload`) |
| Disaggregation | `distributed/omni_connectors/` (SHM, RDMA), `omni_coordinator/` (ZMQ) | Stage processes/boxes exchange KV + payloads; coordinator does membership + LB |
| MTP / spec-decode | `worker/gpu_model_runner.py` `talker_mtp`, `models/common/qwen3_code_predictor.py` | `talker_mtp` head predicts a whole audio-code frame per text step (re-prefill loop, Gumbel-max); upstream draft/verify inherited from vLLM |
| Diffusion | `diffusion/` (sched, engine, executor, offloader, cache) | **continuous-latent** DiT serving (image/video/audio) — VAE encode/decode, Euler/UniPC schedulers, TeaCache/cache-dit step skipping, layerwise CPU offload |
| API/UI | `entrypoints/openai/` | Rich OpenAI wire: `/v1/audio/speech` (+SSE), speech WS, `/v1/realtime`, `/v1/images/generations`, `/v1/videos` jobs, `modality`-tagged chat SSE chunks |
| Quantization | `quantization/` | Delegates to upstream vLLM (GGUF/AWQ/GPTQ/FP8/MXFP4/NF4); `LazyWeightMixin` online-quantize-at-load; `ComponentQuantizationConfig` per-layer routing; config.json auto-detect |

## 2. Key finding: they have NO sub-byte, and no discrete diffusion

- **Every** DiT pipeline is continuous float32 latents + VAE. There is **no
  byte/token discrete diffusion** in the repo. Our 264-state byte canvas
  (256 bytes + 8 specials) is orthogonal.
- **No sub-byte packing.** Finest is bitsandbytes NF4 (4-bit). Our U8 ×
  0.001 = 0.008 bit/weight ceiling is strictly beyond them.
- Their mmap weight loading (`_load_weights_via_mmap`) is a *transient
  source*: mmap the safetensors → copy the rank's shard into pinned RAM →
  release handles. The file is **not** the running state. Ours is
  (`MappedX8DReader`, page-cache served, `/0.001` live).
- Their dequant happens inside fused GPU GEMM kernels; there is no
  "compressed state IS the running state" design anywhere.

**Conclusion:** we do not copy their quantization or their tensor serving.
We adopt their *routing/protocol/pipeline concepts* where they map onto byte
canvases.

## 3. Capability comparison (verified)

| Capability | vLLM-Omni | x8D (us) | Verdict |
|---|---|---|---|
| Modality detection (`omni_task`) | serving layer sets `additional_information.omni_task=["t2i"|"chat"]`; model forward switches constraints | `detect_modality()` in `omni_chat_probe.py` + `[IMG_START(260)]`/`[AUD_START(262)]` markers | **We already have**; theirs is per-stage, ours is canvas-marker based (byte-native, better) |
| Stage-chain pipeline | orchestrator routes stage 0 → stage N via `process_engine_inputs` + `custom_process_input_func` (AR→DiT, thinker→talker→codec) | SARA `SARARouter` maps modality → isolated customer boundary (moe_disk.py) | **Partial gap**: we route *between customers*; they route *within a model* across stages. A pure-Python stage chain (text canvas → image canvas → audio canvas) over byte payloads is missing |
| Inter-stage payload contract | `OmniPayload` / `OmniPayloadStruct` (msgspec-validated, tensor-serialized) | None — single-stage server currently | **Gap**: adopt the *shape* (hidden/embed/ids/codes/meta) but as raw `list[bytes]` payloads, msgspec replaced by stdlib schema |
| Streaming across stages | `bridge_states` + watermark delta tracking (`last_prompt_len`/`last_output_len`) | Our SSE emits one full content delta + usage | **Gap**: incremental byte-delta tracking per canvas segment |
| MTP head | `talker_mtp`: predicts a BLOCK (audio codes) per step, re-prefill loop, Gumbel-max, gated by confidence at `0.001`?? (no — vLLM has no 0.001 gate) | DSpark 8x8 block-parallel spec-decode with confidence head + **0.001 threshold** + re-mask (x8d_spec_decode.py) | **We already have** the block-predictor concept + threshold; theirs is torch/CUDA-graph, ours is stdlib. Port the *hidden-state→block* architecture when we get a real head (#7) |
| Non-AR diffusion engine | `StepScheduler`/`DiffusionEngine` (request-mode / step-mode, per-denoise-step accounting, post_decode) | `ByteDiffusionSampler` (mask_canvas → denoise steps → decode) in pure Python | **Conceptual parity**; their step-mode scheduling is a clean model for a byte-canvas step engine |
| Web UI | No bundled UI; API only (OpenAI wire) | ChatGPT-style `web/` UI served at `/` | **We win**; no port |
| OpenAI API surface | `/v1/chat/completions` (+modality), `/v1/audio/speech` (+SSE), speech WS, `/v1/realtime`, `/v1/images/*`, `/v1/videos/*` jobs | `/v1/chat/completions` (+SSE), `/v1/models`, `/healthz`, `/telemetry` | **Gap**: speech/image/video wire protocols (see §4) |
| Quantization | delegates to vLLM; no sub-byte | U8 × 0.001 sub-byte, mmap live, lossless | **We win decisively**; nothing to port |
| Online quantize-at-load | `LazyWeightMixin` (meta device → JIT materialize + `replace_parameter`) | Import-time conversion via `x8d_export.py` / pointer maps | **Minor idea**: adopt JIT-on-load so a raw checkpoint converts lazily at first touch |
| Config-driven quant routing | `ComponentQuantizationConfig` longest-prefix + `resolve_quant_config_from_disk` (method mismatch → hard error) | Uniform U8 everywhere; no per-submodule routing | **Gap (small)**: add per-boundary quant policy (e.g. norm layers could stay denser) |
| Weight loading from disk | meta→mmap-view then copied+released | mmap IS the running state | **We win**; theirs is a staging trick |
| KV prefix cache / step caches | `OmniTensorPrefixCache`, TeaCache/cache-dit | none (bytes are cheap) | Not needed — byte canvas, no KV |

## 4. Port-order recommendation (byte-native, our models, pure stdlib)

Ranked by value ÷ cost. Only concepts, adapted to 264-byte vocab.

1. **`modality` field on chat SSE chunks** — the single highest-value, lowest
   cost change. `OmniChatCompletionStreamResponse` adds `"modality"` per
   delta. For us: detect `[IMG_START]`/`[AUD_START]` markers in the denoised
   canvas and tag each SSE chunk `text|image|audio`. Unlocks the web UI to
   render audio play buttons / image tiles later. **~30 lines.**

2. **Audio/image wire protocols (protocol-only)** — `/v1/audio/speech` SSE
   (`event: speech.audio.delta/done/error`, base64 bytes) and
   `/v1/images/generations` (`b64_json`). We have raw bytes at ids 0-255
   framed by `[AUD_START]/[IMG_START]`; a byte-canvas denoise of a framed
   canvas IS a PCM/image output. Add the endpoints, keep backend = byte
   pipeline. **~150 lines + tests.**

3. **Pure-Python stage chain over byte canvases** — port the *shape* of
   `Orchestrator._forward_to_next_stage`: stage 0 = text byte canvas,
   stage N = image/audio byte canvas, payload = `{ids, meta}` (raw lists,
   no tensors). Stage detection via canvas markers. This is the omni-routing
   piece that is genuinely missing (we route customers, not stages). **~300
   lines.**

4. **Incremental byte-delta streaming** — `bridge_states` watermark pattern:
   per SSE segment, emit only `inc` bytes since `last_len`. Makes long
   generation streamable. **~100 lines.**

5. **Block-predictor head for MTP** — keep DSpark masks; add a
   hidden-state→8x8-block predictor seam so a future torch head (#7) slots
   in. Design-only now; stdlib `_block_surrogate` stays until torch arrives.

**Not porting** (byte-law or scope rejects): msgspec/ZMQ payload layer
(stdlib instead), OmniKVTransferManager (no KV), continuous diffusion
schedulers (byte-space uses entropy-bound discrete steps), GPU offload hooks,
upstream vLLM sampler kernels, world-model/fullduplex experimental stack.

## 5. Concrete first deliverable (selected)

Items **1 + 2 + 4** together form a coherent, testable milestone:
"**multi-modal OpenAI wire + modality-tagged streaming**" — SSE chat chunks
carry `modality`, `/v1/audio/speech` and `/v1/images/generations` are added
over the byte-canvas pipeline with incremental byte deltas. Pure stdlib,
our models, 0.001 containers unchanged. See issue #46 acceptance criteria.
