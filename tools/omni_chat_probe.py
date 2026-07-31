# coding=utf-8
"""Omni-chat probe for the OpenAI-compatible byte endpoint (issue #39).

Pure Python standard library only. Two responsibilities:

1. **Endpoint probe** — drives ``POST /v1/chat/completions`` (plus
   ``GET /healthz`` and ``GET /v1/models``) through the byte-native
   pipeline of ``tools/openai_chat_server.py``. By default it runs a real
   HTTP round-trip against a ``ThreadingHTTPServer`` bound to an ephemeral
   port; with ``--offline`` it goes straight through the importable
   request core (``handle_request_body``) and never binds a socket.

2. **Active-expert report** — reports which expert's parameters are ACTIVE
   for the routed modality. It uses the **SARA router** (issue #36) to map
   the modality to its isolated customer boundary (real, researched param
   counts: kimi-k3 104.2B active, whisper 1.55B, ltx2 19B, kokoro 82M),
   and ``MoEOnDisk`` to demonstrate the on-disk mechanism on a synthetic
   x8D GGUF container: mmap + live ``/0.001`` reverse on the routed expert
   block only, no other block's bytes touched.

SARA note (issue #36): ``SARARouter`` IS present in ``moe_disk.py`` (the
#36 agent landed it), so this probe uses it directly. Dense-as-single-
expert and per-modality isolation are honoured: every expert is a self-
contained block of ``layers.<layer>.experts.<expert>.w{1,2,3}`` tensors.

Run:  python3 tools/omni_chat_probe.py [--prompt "..." ] [--offline]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.moe_disk import MoEOnDisk, SARARouter  # noqa: E402
from omni_diffusion.x8d_export import save_gguf  # noqa: E402
from tools.openai_chat_server import (  # noqa: E402
    MODEL_ID,
    build_models_response,
    handle_request_body,
)

#: Default diffusion canvas length for the byte pipeline (mirrors config).
CANVAS_STEPS: int = 48

#: SARA router (issue #36): modality/customer -> isolated boundary. Dense
#: models are a single expert; internal-MoE models route to their own pool.
SARAROUTER: SARARouter = SARARouter()

#: Demo-container expert block per modality (routing_key -> (layer, expert)).
#: The demo container carries one self-contained expert block per modality;
#: only the routed block (plus the shared block) is mmap'd at query time.
EXPERT_BY_MODALITY: Dict[str, tuple] = {
    "text": (0, 0),
    "image": (0, 1),
    "audio": (0, 2),
    "video": (0, 3),
}

#: DeepSeek-style shared expert: always active for every modality.
SHARED_EXPERT: tuple = (0, 4)

#: Demo expert FFN geometry (bytes == parameters under the byte law).
_IN_FEAT, _MID_FEAT, _OUT_FEAT = 8, 8, 8

#: Modality markers recognised in prompts (mirror byte-law specials
#: IMG_START=260 / AUD_START=262 once multimodal chat content lands).
_MODALITY_MARKERS: Dict[str, tuple] = {
    "image": ("[IMG_START]", "img_start", "<image>"),
    "audio": ("[AUD_START]", "aud_start", "<audio>"),
    "video": ("[VID_START]", "video"),
}


# ---------------------------------------------------------------------------
# Endpoint probe
# ---------------------------------------------------------------------------

def _completion_body(prompt: str) -> Dict[str, Any]:
    """Build a standard OpenAI chat-completions request body."""
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a byte-native assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }


def offline_probe(prompt: str) -> Dict[str, Any]:
    """Drive the endpoint through the importable request core (no socket)."""
    raw = json.dumps(_completion_body(prompt)).encode("utf-8")
    status, payload = handle_request_body(raw)
    if status != 200:
        raise RuntimeError(f"chat completion failed with status {status}: {payload}")
    return {
        "mode": "offline",
        "port": 0,
        "healthz": {"status_code": "n/a (offline)", "body": {"status": "ok"}},
        "models": {"status_code": "n/a (offline)", "body": build_models_response()},
        "completion": payload,
        "completion_status": status,
        "latency_ms": None,
    }


def serve_probe(
    prompt: str,
    host: str = "127.0.0.1",
    port: int = 0,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Run a real HTTP round-trip against the endpoint on an ephemeral port.

    Args:
        prompt: user message to send.
        host: bind host.
        port: bind port (0 = OS-assigned ephemeral).
        timeout: per-request socket timeout in seconds.

    Returns:
        Dict with healthz, models and the chat-completion response, plus
        the actual port and the completion latency in ms.
    """
    server = ThreadingHTTPServer((host, port), _import_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/healthz", timeout=timeout) as resp:
            healthz = {"status_code": resp.status, "body": json.loads(resp.read().decode("utf-8"))}
        with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as resp:
            models = {"status_code": resp.status, "body": json.loads(resp.read().decode("utf-8"))}

        body = json.dumps(_completion_body(prompt)).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            completion = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.monotonic() - start) * 1000.0
        return {
            "mode": "http",
            "port": server.server_address[1],
            "healthz": healthz,
            "models": models,
            "completion": completion,
            "completion_status": resp.status,
            "latency_ms": latency_ms,
        }
    finally:
        server.shutdown()
        server.server_close()


def _import_handler() -> type:
    """Import ChatCompletionHandler lazily (keeps import order clean)."""
    from tools.openai_chat_server import ChatCompletionHandler

    return ChatCompletionHandler


# ---------------------------------------------------------------------------
# Active-expert report (MoEOnDisk + SARA stand-in)
# ---------------------------------------------------------------------------

def _expert_ffn_sizes() -> Dict[str, int]:
    """Bytes per projection for the demo SwiGLU-style expert FFN."""
    return {
        "w1": _IN_FEAT * _MID_FEAT,
        "w2": _MID_FEAT * _OUT_FEAT,
        "w3": _MID_FEAT * _MID_FEAT,
    }


def build_demo_experts(path: str) -> Dict[str, int]:
    """Write a small synthetic x8D GGUF container with one block per modality.

    Each block is ``layers.<layer>.experts.<expert>.w{1,2,3}`` U8 tensors;
    a shared expert (always active) is included as in DeepSeek-style MoE.

    Args:
        path: output .gguf path.

    Returns:
        Mapping of block name -> total byte count (bytes == params).
    """
    payloads: Dict[str, bytes] = {}
    counts: Dict[str, int] = {}
    specs = list(EXPERT_BY_MODALITY.items()) + [("shared", SHARED_EXPERT)]
    ffn = _expert_ffn_sizes()
    for name, (layer, expert) in specs:
        total = 0
        for proj, size in ffn.items():
            # deterministic pseudo-random weights in [0, 255]
            seed = expert * 13 + (1 if proj == "w1" else 2 if proj == "w2" else 3)
            payloads[f"layers.{layer}.experts.{expert}.{proj}"] = bytes(
                ((i * 37) + seed * 11) % 256 for i in range(size)
            )
            total += size
        counts[name] = total
    save_gguf(payloads, path)
    return counts


def active_expert_report(gguf_path: str, modality: str) -> Dict[str, Any]:
    """Route ``modality`` to its expert block and count ACTIVE parameters.

    Uses the SARA router (issue #36) for the production routing decision
    (modality -> customer boundary with researched active/total params),
    then ``MoEOnDisk`` to mechanically serve the routed expert block from
    the demo container: only that block's (plus the shared block's) byte
    spans are mmap'd and ``/0.001``-reversed live. Isolation guarantee:
    no other block's bytes are read.

    Args:
        gguf_path: on-disk x8D GGUF container.
        modality: one of text / image / audio / video.

    Returns:
        Dict with the SARA boundary, the routing key, per-tensor byte
        counts, routed/shared/total active parameters, container disk
        size and whether the ``/0.001`` reverse was byte-exact.

    Raises:
        KeyError: unknown modality (no SARA boundary mapped).
    """
    boundary = SARAROUTER.route(modality)
    routed = EXPERT_BY_MODALITY.get(modality)
    if routed is None:
        raise KeyError(f"no demo expert block mapped for modality {modality!r}")
    moe = MoEOnDisk(gguf_path)
    try:
        layer, expert = routed
        routed_keys = [moe.expert_key(layer, expert, p) for p in ("w1", "w2", "w3")]
        shared_keys = [moe.expert_key(*SHARED_EXPERT, p) for p in ("w1", "w2", "w3")]
        routed_params = 0
        routed_sizes: Dict[str, int] = {}
        for proj, key in zip(("w1", "w2", "w3"), routed_keys):
            w = moe.load_expert(layer, expert, proj)
            routed_sizes[key] = len(w)
            routed_params += len(w)
        shared_params = 0
        shared_sizes: Dict[str, int] = {}
        for proj, key in zip(("w1", "w2", "w3"), shared_keys):
            w = moe.load_expert(*SHARED_EXPERT, proj)
            shared_sizes[key] = len(w)
            shared_params += len(w)
        # byte-law reverse exactness: round((b*0.001)/0.001) & 0xFF == b
        sample = moe.load_expert(layer, expert, "w1")
        stored = moe.reader.tensor_bytes(routed_keys[0])
        reverse_exact = stored is not None and sample == list(stored)
        return {
            "modality": modality,
            "sara": {
                "customer": boundary.customer,
                "mode": boundary.mode,
                "upstream_repo": boundary.upstream_repo,
                "active_params": boundary.active_params,
                "total_params": boundary.total_params,
            },
            "routing_key": f"layers.{layer}.experts.{expert}",
            "routed_expert": expert,
            "routed_tensors": routed_sizes,
            "routed_params": routed_params,
            "shared_expert": f"layers.{SHARED_EXPERT[0]}.experts.{SHARED_EXPERT[1]}",
            "shared_tensors": shared_sizes,
            "shared_params": shared_params,
            "active_params": routed_params + shared_params,
            "disk_mb": moe.size_mb(),
            "reverse_exact": reverse_exact,
        }
    finally:
        moe.close()


def expert_table(gguf_path: str) -> List[Dict[str, Any]]:
    """Active-parameter row per modality for the SARA routing table."""
    rows: List[Dict[str, Any]] = []
    for modality in EXPERT_BY_MODALITY:
        rep = active_expert_report(gguf_path, modality)
        rows.append(
            {
                "modality": modality,
                "customer": rep["sara"]["customer"],
                "sara_active_params": rep["sara"]["active_params"],
                "routing_key": rep["routing_key"],
                "active_params": rep["active_params"],
                "wired_today": modality == "text",
            }
        )
    return rows


def detect_modality(text: str) -> str:
    """Pick the modality path from prompt markers; defaults to 'text'.

    The chat endpoint's pipeline is text-only today (``ByteTokenizer.encode``
    -> ids 0-255); image/audio content will flow through ``encode_image`` /
    ``encode_audio`` once multimodal chat content lands. The markers mirror
    the byte-law specials IMG_START=260 / AUD_START=262.
    """
    lowered = text.lower()
    for modality, markers in _MODALITY_MARKERS.items():
        if any(marker.lower() in lowered for marker in markers):
            return modality
    return "text"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(rep: Dict[str, Any], pipeline_modality: str) -> str:
    """Render the endpoint + active-expert report as a text block."""
    comp = rep["completion"]
    usage = comp.get("usage", {})
    choice = comp["choices"][0]
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append("x8D OpenAI-compatible endpoint probe (issue #39)")
    lines.append("=" * 74)
    mode = rep["mode"]
    lines.append(
        f"mode                 : {mode}"
        + (f" (ThreadingHTTPServer on port {rep['port']})" if mode == "http" else " (no socket; handle_request_body)")
    )
    hz = rep["healthz"]
    lines.append(f"GET /healthz         -> {hz['status_code']} {json.dumps(hz['body'])}")
    md = rep["models"]
    model = md["body"]["data"][0]
    lines.append(
        f"GET /v1/models       -> {md['status_code']} {model['id']} / owned_by {model['owned_by']}"
    )
    lat = rep["latency_ms"]
    lat_s = f"{lat:.1f} ms" if lat is not None else "n/a (offline)"
    lines.append(f"POST /v1/chat/completions -> {rep['completion_status']}  ({lat_s})")
    lines.append(f"  id            : {comp['id']}")
    lines.append(f"  object/model  : {comp['object']} / {comp['model']}")
    lines.append(f"  content       : {choice['message']['content']!r}")
    lines.append("  usage (BYTES) : "
                 f"prompt {usage.get('prompt_tokens', '?')} / "
                 f"completion {usage.get('completion_tokens', '?')} / "
                 f"total {usage.get('total_tokens', '?')}"
                 "  (byte law: tokens == bytes; wire-compatible *_tokens keys)")
    lines.append(
        "  pipeline path : text  (chat endpoint runs ByteTokenizer.encode -> "
        f"mask_canvas({CANVAS_STEPS}) -> denoise -> decode; ids 0-255)"
    )
    lines.append("")
    lines.append("Active-expert report (SARARouter -> boundary; MoEOnDisk mmap + live /0.001; #36/#39)")
    lines.append("-" * 74)
    sara = rep["_expert"]["sara"]
    lines.append(f"  modality requested : {pipeline_modality}")
    lines.append(f"  SARA boundary     : {sara['customer']}  (mode={sara['mode']}, upstream {sara['upstream_repo']})")
    lines.append(f"  SARA active/total : {sara['active_params']:,} / {sara['total_params']:,} params")
    lines.append(f"  demo routed block : {rep['_expert']['routing_key']}")
    for key, size in rep["_expert"]["routed_tensors"].items():
        lines.append(f"    {key:<32} {size} B")
    lines.append(f"  routed params     : {rep['_expert']['routed_params']}  (only these bytes are /0.001-reversed)")
    lines.append(f"  shared block      : {rep['_expert']['shared_expert']}  (always active)")
    for key, size in rep["_expert"]["shared_tensors"].items():
        lines.append(f"    {key:<32} {size} B")
    lines.append(f"  shared params     : {rep['_expert']['shared_params']}")
    lines.append(f"  TOTAL active      : {rep['_expert']['active_params']} params")
    lines.append(f"  /0.001 reverse    : {'EXACT (round(q/0.001)==b)' if rep['_expert']['reverse_exact'] else 'MISMATCH'}")
    lines.append(f"  container on disk : {rep['_expert']['disk_mb']:.3f} MB (mmap; zero RAM residency)")
    lines.append("")
    lines.append("SARA routing table (#36) — active params per modality (SARA boundary vs demo block)")
    lines.append("-" * 74)
    for row in rep["_table"]:
        wired = "wired today (this request)" if row["wired_today"] else "routable; pipeline not wired yet"
        lines.append(
            f"  {row['modality']:<7} -> {row['customer']:<20} "
            f"SARA {row['sara_active_params']/1e9:>6.1f}B active   "
            f"demo {row['routing_key']} {row['active_params']:>4} B   ({wired})"
        )
    lines.append("")
    return "\n".join(lines)


def print_report(
    report: Dict[str, Any],
    expert: Dict[str, Any],
    table: List[Dict[str, Any]],
    pipeline_modality: str,
) -> None:
    """Print the combined endpoint + active-expert report."""
    print(_fmt({**report, "_expert": expert, "_table": table}, pipeline_modality))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python3 tools/omni_chat_probe.py``."""
    parser = argparse.ArgumentParser(
        description="Probe the OpenAI-compatible byte endpoint + report active MoE experts (#39)"
    )
    parser.add_argument(
        "--prompt",
        default="Hello byte world, tell me about MoE routing and byte diffusion.",
        help="user prompt to send (default: %(default)s)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="drive handle_request_body directly instead of a real HTTP socket",
    )
    parser.add_argument(
        "--modality",
        choices=tuple(EXPERT_BY_MODALITY),
        default=None,
        help="expert block to route (default: auto-detect from prompt markers)",
    )
    parser.add_argument("--port", type=int, default=0, help="HTTP bind port (default: ephemeral)")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="x8d-probe-") as tmp:
        gguf_path = os.path.join(tmp, "moe_demo.x8d.gguf")
        build_demo_experts(gguf_path)
        if args.offline:
            report = offline_probe(args.prompt)
        else:
            report = serve_probe(args.prompt, port=args.port)
        modality = args.modality or detect_modality(args.prompt)
        expert = active_expert_report(gguf_path, modality)
        table = expert_table(gguf_path)
        print_report(report, expert, table, modality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
