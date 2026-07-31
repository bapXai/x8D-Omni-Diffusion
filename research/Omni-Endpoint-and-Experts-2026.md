# Omni-Endpoint-and-Experts-2026 — OpenAI-compatible endpoint probe + active-expert report (issue #39)

Status: verified live 2026-07-31. Drives the byte-native OpenAI-compatible
endpoint (`tools/openai_chat_server.py`) through a real HTTP round-trip and
reports which MoE expert's parameters are ACTIVE per routed modality using
`SARARouter` (issue #36) + `MoEOnDisk` (`omni_diffusion/moe_disk.py`).

## 1. What was built

| Artifact | Purpose |
|---|---|
| `tools/omni_chat_probe.py` | Pure-stdlib probe: POSTs `/v1/chat/completions` (real HTTP on an ephemeral port, or offline via `handle_request_body`), prints response + usage-in-bytes + the modality path that ran + the active-expert report. |
| `tests/test_openai_server_live.py` | 26 stdlib tests. Offline-first: drives the real `ChatCompletionHandler.do_GET/do_POST` routing through a socket-free fake (an `object.__new__` handler with a BytesIO `wfile`), plus the importable request core, plus the SARA/MoEOnDisk active-expert report. One real-socket round-trip is `@unittest.skipUnless(_SOCKET_OK, ...)`-gated. |
| `research/Omni-Endpoint-and-Experts-2026.md` | This doc: probe output + active-expert-per-modality table. |

No changes to `tools/openai_chat_server.py` were needed — its request core
was already importable/testable without a port (`process_chat_completion`,
`handle_request_body`, `build_models_response`), exactly as designed.

SARA note: the #36 agent landed `SARABoundary` / `SARA_REGISTRY` /
`SARARouter` in `moe_disk.py` before this probe was finalized, so the probe
uses the real SARARouter (modality -> isolated customer boundary) rather
than a static map.

## 2. Live probe output (captured 2026-07-31)

```
$ python3 tools/omni_chat_probe.py
==========================================================================
x8D OpenAI-compatible endpoint probe (issue #39)
==========================================================================
mode                 : http (ThreadingHTTPServer on port 65006)
GET /healthz         -> 200 {"status": "ok"}
GET /v1/models       -> 200 x8d-byte-diffusion / owned_by bapX
POST /v1/chat/completions -> 200  (0.5 ms)
  id            : chatcmpl-38265595d5d44409977e667056d8e620
  object/model  : chat.completion / x8d-byte-diffusion
  content       : "x8D says: 'Hello byte world, tell me about MoE routing and byte diffusion.'. Byte-law pipeline (vocab=264, no tokens) denoised canvas -> '��\x14��ϛ��oG�G0�K�2%��3�ޡh��\x1f\x07/�\x00��|� aqzH�.)��7��?�h��"
  usage (BYTES) : prompt 95 / completion 257 / total 352  (byte law: tokens == bytes; wire-compatible *_tokens keys)
  pipeline path : text  (chat endpoint runs ByteTokenizer.encode -> mask_canvas(48) -> denoise -> decode; ids 0-255)

Active-expert report (SARARouter -> boundary; MoEOnDisk mmap + live /0.001; #36/#39)
--------------------------------------------------------------------------
  modality requested : text
  SARA boundary     : kimi-k3  (mode=moe, upstream moonshotai/Kimi-K3)
  SARA active/total : 104,200,000,000 / 2,779,931,837,184 params
  demo routed block : layers.0.experts.0
    layers.0.experts.0.w1            64 B
    layers.0.experts.0.w2            64 B
    layers.0.experts.0.w3            64 B
  routed params     : 192  (only these bytes are /0.001-reversed)
  shared block      : layers.0.experts.4  (always active)
    layers.0.experts.4.w1            64 B
    layers.0.experts.4.w2            64 B
    layers.0.experts.4.w3            64 B
  shared params     : 192
  TOTAL active      : 384 params
  /0.001 reverse    : EXACT (round(q/0.001)==b)
  container on disk : 0.001 MB (mmap; zero RAM residency)

SARA routing table (#36) — active params per modality (SARA boundary vs demo block)
--------------------------------------------------------------------------
  text    -> kimi-k3              SARA  104.2B active   demo layers.0.experts.0  384 B   (wired today (this request))
  image   -> ltx2                 SARA   19.0B active   demo layers.0.experts.1  384 B   (routable; pipeline not wired yet)
  audio   -> whisper-large-v3     SARA    1.6B active   demo layers.0.experts.2  384 B   (routable; pipeline not wired yet)
  video   -> ltx2                 SARA   19.0B active   demo layers.0.experts.3  384 B   (routable; pipeline not wired yet)
```

`--offline` runs the identical request through `handle_request_body` with no
socket (used by `tests/test_openai_server_live.py::test_offline_probe_through_request_core`).
`--modality audio` (or a prompt carrying `[AUD_START]`) routes the expert
report to the `whisper-large-v3` boundary while the endpoint's chat pipeline
still runs the text byte path.

## 3. Active-expert-per-modality table (SARA routing, #36)

| Modality | SARA customer | Mode | Upstream repo | Active params (SARA) | Total params | Demo routed block | Demo active bytes | Wired in endpoint today |
|---|---|---|---|---|---|---|---|---|
| text | `kimi-k3` | moe | moonshotai/Kimi-K3 | 104.2B | 2.78T | `layers.0.experts.0` | 384 B | **yes** (text chat) |
| image | `ltx2` | dense | Lightricks/LTX-2 | 19.0B | 19.0B | `layers.0.experts.1` | 384 B | no (multimodal chat pending) |
| audio | `whisper-large-v3` | dense | openai/whisper-large-v3 | 1.55B | 1.55B | `layers.0.experts.2` | 384 B | no |
| video | `ltx2` | dense | Lightricks/LTX-2 | 19.0B | 19.0B | `layers.0.experts.3` | 384 B | no |
| (shared) | always-on | — | — | — | — | `layers.0.experts.4` | 192 B | yes (DeepSeek-style shared expert) |

SARA semantics that hold here (verified by `test_sara_boundaries_isolated`):
- Dense models (Kokoro-82M, Whisper large-v3, LTX-2) register as a single
  expert (`mode="dense"`); internal-MoE models (GLM-5.2 753B, Kimi-K3
  2.78T, DeepSeek-V4-Pro 1.6T) register their own isolated expert pool
  (`mode="moe"`).
- `SARARouter.is_isolated(a, b)` is true for every pair: routing to one
  boundary never mmaps or `/0.001`-reverses another customer's byte span.
- The mechanical guarantee is demonstrated on a synthetic x8D GGUF
  container: `MoEOnDisk.load_expert` touches ONLY the routed block (plus
  the always-active shared block) and the `/0.001` reverse is byte-exact
  (`round((b*0.001)/0.001) & 0xFF == b`), verified as `reverse_exact: True`.

## 4. Test coverage (#39, `tests/test_openai_server_live.py`, 26 tests)

- **Healthz / models / 404 routing** through the real handler (`do_GET`)
  with a socket-free fake: `/healthz -> {"status":"ok"}`, `/v1/models`
  (id `x8d-byte-diffusion`, owned_by `bapX`), unknown path -> 404.
- **POST routing** (`do_POST`) through the same fake: valid completion,
  malformed JSON -> 400 `invalid_request_error`, `stream=true` -> 400
  `unsupported`, empty `messages` -> 400 `invalid_request_error`.
- **Usage is BYTES**: the wire uses OpenAI's `*_tokens` keys, but their
  values are raw UTF-8 byte counts (no tokenizer exists — vocab is 264).
  `test_usage_keys_are_byte_counts` proves prompt/completion/total byte math
  exactly; `test_byte_keys_note` documents the byte-native naming.
- **Role validation**: system-only lists complete with an empty user prompt;
  the last `user` message wins; non-str content coerced; unknown roles and
  non-dict entries tolerated.
- **Byte pipeline**: `byte_pipeline` is deterministic; the reference sampler
  satisfies the `denoise(canvas, steps, seed)` contract.
- **Active-expert report**: per-modality isolation, SARA route map
  (text->kimi-k3, image/video->ltx2, audio->whisper), dense-vs-moe split,
  `reverse_exact`, `expert_table`, `offline_probe`.
- **Live socket round-trip** (gated `@unittest.skipUnless(_SOCKET_OK)`):
  a real `ThreadingHTTPServer` on an ephemeral port serving healthz/models/
  chat completion over `urllib`.

Full suite: `python3 -m unittest discover -s tests -v` — 244 tests OK
(6 network/transformers-gated skips), also green under
`python3 -W error::ResourceWarning -m unittest discover -s tests -v`.

## 5. Sources / related
- `tools/openai_chat_server.py` (#29) — byte-native OpenAI-compat endpoint.
- `omni_diffusion/moe_disk.py` — `SARABoundary`/`SARA_REGISTRY`/`SARARouter`
  (#36) + `MoEOnDisk` (#9, mmap + live `/0.001` reverse).
- `omni_diffusion/x8d_export.py` — X8DGGUF1 U8 container + 0.001 law.
- `research/Status-and-Optimization-Audit-2026.md` (#33).
- `research/MoE-Omni-Diffusion-Language-Modeling-2026.md` (#40) — MoE
  routing theory + omni any-to-any + diffusion-LM + byte-law justification.
