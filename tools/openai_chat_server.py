# coding=utf-8
"""OpenAI-compatible chat endpoint + ChatGPT-style web UI for the byte pipeline.

Pure Python standard library only (``http.server``, ``json``, ``time``,
``uuid``, ``argparse``) — no torch, no transformers, no tokenizer. Mirrors
the concept of the upstream ``bapXai/x8Dsub-byte`` openai_chat_server.py
but grounded in the byte law: vocabulary is 264 (bytes 0-255 + specials
256-263), ``usage`` is reported in BYTES not tokens, and each prompt is
pushed through the encode -> mask -> denoise -> decode byte pipeline.

Features (issue #43, #45):
- Serves the ChatGPT-style UI from ``web/`` at ``/``.
- ``POST /v1/chat/completions`` supports OpenAI-shaped **streaming** via
  Server-Sent Events (``stream: true``) plus non-streaming mode.
- ``GET /telemetry`` returns the Colibrì-style I/O + RSS dashboard line.
- ``--disk-repo <dir>`` runs in **low-RAM from-disk mode**: a MappedX8DReader
  maps an x8D container and every completion is served by reverse-slicing
  payloads out of the kernel page cache (no GPU, ~1 GB RAM target).

Endpoints:
    GET  /                     -> web UI (index.html)
    GET  /healthz              -> {"status": "ok"}
    GET  /telemetry            -> telemetry dashboard
    GET  /v1/models            -> {"object": "list", "data": [{...}]}
    POST /v1/chat/completions  -> OpenAI completion (stream + non-stream)

Run:  python3 tools/openai_chat_server.py --port 666
      python3 tools/openai_chat_server.py --port 666 --disk-repo ./x8d_weights
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.models.dream.byte_tokenizer import (  # noqa: E402
    MASK_TOKEN_ID,
    ByteTokenizer,
)
from omni_diffusion.x8d_telemetry import Telemetry  # noqa: E402

DEFAULT_PORT: int = 666
DEFAULT_HOST: str = "0.0.0.0"

MODEL_ID: str = "x8d-byte-diffusion"
MODEL_OWNER: str = "bapX"

#: Fixed model creation timestamp (module import time).
_MODEL_CREATED: int = int(time.time())

#: Diffusion canvas length / denoise steps mirroring the query-pipeline tests.
CANVAS_STEPS: int = 48
_CANVAS_SEED: int = 0

#: Repository root (web/ lives alongside tools/).
_REPO_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WEB_ROOT: str = os.path.join(_REPO_ROOT, "web")

#: Module-level telemetry collector (shared by handler + /telemetry).
_TELEMETRY = Telemetry(label="x8d-web")

#: Runtime mode + optional disk-backed reader (set by run_server).
_SERVER_MODE: str = "memory"
_DISK_READER = None


# ---------------------------------------------------------------------------
# Byte pipeline (encode -> mask -> denoise -> decode)
# ---------------------------------------------------------------------------

class _LocalByteDiffusionSampler:
    """Fallback sampler if the reference class is not importable.

    Byte-identical to the ``ByteDiffusionSampler`` used by
    ``tests/test_queries.py``: a fully-masked canvas is iteratively filled
    with a seeded pseudo-byte mix, so results are deterministic (seed 0).
    """

    def __init__(self, vocab_size: int = 264) -> None:
        self.vocab_size = vocab_size

    def denoise(self, canvas: List[int], steps: int = CANVAS_STEPS, seed: int = 0) -> List[int]:
        """Fill mask slots with a seeded byte mix; returns the final canvas."""
        rng = random.Random(seed)
        out = list(canvas)
        for step in range(steps):
            done = True
            for i, tok in enumerate(out):
                if tok == MASK_TOKEN_ID:
                    out[i] = (rng.randint(0, 255) + step) & 0xFF
                    done = False
            if done:
                break
        return out


def _load_reference_sampler() -> type:
    """Return the query-pipeline sampler class when importable.

    Tries to import ``ByteDiffusionSampler`` from ``tests/test_queries.py``
    (pure stdlib). Falls back to ``_LocalByteDiffusionSampler`` so the
    server runs anywhere, even without a tests checkout.
    """
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "x8d_server_query_sampler", os.path.join(_REPO_ROOT, "tests", "test_queries.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError("cannot build module spec for test_queries.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.ByteDiffusionSampler
    except Exception:
        return _LocalByteDiffusionSampler


_TOKENIZER: ByteTokenizer = ByteTokenizer()
_SAMPLER_CLASS: type = _load_reference_sampler()
_SAMPLER = _SAMPLER_CLASS()


def byte_pipeline(text: str) -> str:
    """Run ``text`` through the byte-native diffusion pipeline.

    Encodes the UTF-8 bytes as [BOS .. bytes .. EOS], masks a canvas of the
    same length, denoises deterministically, then decodes the result to a
    UTF-8 string (content ids are always 0-255; specials are skipped).
    """
    ids = _TOKENIZER.encode(text.encode("utf-8"), add_special_tokens=True)
    canvas = _TOKENIZER.mask_canvas(len(ids))
    _TELEMETRY.begin_block()
    try:
        denoised = _SAMPLER.denoise(canvas, steps=CANVAS_STEPS, seed=_CANVAS_SEED)
    finally:
        _TELEMETRY.end_block()
        _TELEMETRY.record_io(len(ids))
    return _TOKENIZER.decode_text(denoised)


def _disk_denoise(text: str) -> str:
    """Low-RAM path: reverse an x8D payload slice out of the mmap as filler.

    In ``--disk-repo`` mode the sampler is replaced by a from-disk reverse:
    a payload's U8 coordinates are sliced straight out of the memory map
    (zero-copy, page-cache served) and fed through the live ``/0.001``
    inverse to form the denoised canvas. RSS stays ~1 GB — the file is
    never loaded into RAM.
    """
    reader = _DISK_READER
    if reader is None or not reader.names():
        return byte_pipeline(text)
    name = reader.names()[0]
    _TELEMETRY.record_io(len(text.encode("utf-8")))
    coords = reader.reverse(name)
    return _TOKENIZER.decode_text(
        [int(c // 1) & 0xFF for c in coords][: 512]
    )


# ---------------------------------------------------------------------------
# Request core (importable / testable without a socket)
# ---------------------------------------------------------------------------

class ChatCompletionError(ValueError):
    """Raised for invalid chat-completion requests (maps to HTTP 400)."""

    def __init__(self, message: str, error_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


def error_response(message: str, error_type: str = "invalid_request_error") -> Dict:
    """Build an OpenAI-style ``{"error": {...}}`` payload."""
    return {"error": {"message": message, "type": error_type}}


def build_models_response() -> Dict:
    """Build the ``GET /v1/models`` payload."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": _MODEL_CREATED,
                "owned_by": MODEL_OWNER,
            }
        ],
    }


def extract_last_user_message(messages: List) -> Optional[str]:
    """Return the content of the last ``user`` message, or None."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)
    return None


def _count_prompt_bytes(messages: List) -> int:
    """Byte length of every message's content (byte law: tokens == bytes)."""
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content.encode("utf-8"))
        elif isinstance(content, (bytes, bytearray)):
            total += len(content)
    return total


def _truncate_bytes(text: str, limit: int) -> str:
    """Truncate to ``limit`` UTF-8 bytes without splitting a codepoint."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    while limit > 0 and (encoded[limit - 1] & 0xC0) == 0x80:
        limit -= 1
    return encoded[:limit].decode("utf-8", errors="replace")


def _chat_content(last_user: str) -> str:
    """Deterministic assistant content: echo + byte-pipeline denoise result."""
    if _SERVER_MODE == "disk":
        denoised = _disk_denoise(last_user)
    else:
        denoised = byte_pipeline(last_user)
    return (
        f"x8D says: '{last_user}'. Byte-law pipeline (vocab=264, no tokens, "
        f"mode={_SERVER_MODE}) denoised canvas -> {denoised!r}"
    )


def _process_stream(body: Dict, emit) -> str:
    """Run a streaming completion, emitting SSE chunks via ``emit``.

    The byte pipeline is fully deterministic and non-tokenized, so we emit a
    single content delta (the full denoised byte canvas) and then a
    ``usage`` chunk carrying byte counts — the same shape the web UI and
    ``curl -N`` consumers expect.

    Args:
        body: parsed request body.
        emit: callback invoked with each complete ``data:`` payload (a dict).

    Returns:
        The full assistant content (used for usage accounting).
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ChatCompletionError("missing 'messages' (a non-empty list is required)")
    last_user = extract_last_user_message(messages)
    if last_user is None:
        last_user = ""
    content = _chat_content(last_user)
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = None
        if max_tokens and max_tokens > 0:
            content = _truncate_bytes(content, max_tokens)

    emit({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    })

    prompt_bytes = _count_prompt_bytes(messages)
    completion_bytes = len(content.encode("utf-8"))
    emit({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_ID,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_bytes,
            "completion_tokens": completion_bytes,
            "total_tokens": prompt_bytes + completion_bytes,
        },
    })
    return content


def process_chat_completion(body: Dict, stream: bool = False) -> Optional[Dict]:
    """Turn an OpenAI chat-completions body into a response dict.

    Args:
        body: parsed JSON request body.
        stream: if True, streaming is handled by the caller via
            :func:`_process_stream`; this function only validates.

    Returns:
        OpenAI-shaped ``chat.completion`` dict (non-stream), or None when
        ``stream=True`` (use :func:`_process_stream` instead). ``usage``
        counts are BYTE counts (prompt/completion/total), never tokens.

    Raises:
        ChatCompletionError: malformed body or missing/invalid ``messages``.
    """
    if not isinstance(body, dict):
        raise ChatCompletionError("request body must be a JSON object")

    if body.get("stream"):
        return None

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ChatCompletionError("missing 'messages' (a non-empty list is required)")

    model = body.get("model") or MODEL_ID
    if not isinstance(model, str):
        model = MODEL_ID

    last_user = extract_last_user_message(messages)
    if last_user is None:
        last_user = ""

    content = _chat_content(last_user)
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = None
        if max_tokens and max_tokens > 0:
            content = _truncate_bytes(content, max_tokens)

    prompt_bytes = _count_prompt_bytes(messages)
    completion_bytes = len(content.encode("utf-8"))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_bytes,
            "completion_tokens": completion_bytes,
            "total_tokens": prompt_bytes + completion_bytes,
        },
    }


def handle_request_body(raw: bytes, stream: bool = False):
    """Parse and process a raw request body; never raises.

    Args:
        raw: raw request body bytes.
        stream: request is a streaming SSE request.

    Returns:
        ``(status_code, response_payload)`` — for non-stream: 200 with the
        completion or 400 with an OpenAI-style error dict. For stream:
        ``(200, None, chunks)`` on success, ``(400, error_dict, None)`` on
        error.
    """
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        err = error_response(f"malformed JSON: {exc}")
        return (400, err, None) if stream else (400, err)
    try:
        if stream:
            chunks: List[Dict] = []

            def emit(payload: Dict) -> None:
                chunks.append(payload)

            _process_stream(body, emit)
            return 200, None, chunks
        return 200, process_chat_completion(body)
    except ChatCompletionError as exc:
        err = error_response(exc.message, exc.error_type)
        return (400, err, None) if stream else (400, err)


# ---------------------------------------------------------------------------
# HTTP handler + server
# ---------------------------------------------------------------------------

_MIME: Dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".ico": "image/x-icon",
}


class ChatCompletionHandler(BaseHTTPRequestHandler):
    """``BaseHTTPRequestHandler`` serving the web UI + OpenAI endpoints."""

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:
        """Route GET: / (UI), /healthz, /telemetry, /v1/models, static files."""
        path = self.path.split("?")[0]
        if path == "/v1/models":
            self._send_json(200, build_models_response())
        elif path == "/healthz":
            self._send_json(200, {"status": "ok", "mode": _SERVER_MODE})
        elif path == "/telemetry":
            self._send_json(200, _telemetry_payload())
        elif path == "/":
            self._send_static("index.html")
        elif path == "/favicon.ico":
            self._send_static("favicon.ico", ok_on_missing=True)
        elif path.startswith("/"):
            name = path.lstrip("/")
            if "/" in name or ".." in name:
                self._send_json(404, error_response(f"not found: {path}"))
                return
            self._send_static(name, ok_on_missing=True)

    def do_POST(self) -> None:
        """Route POST: /v1/chat/completions (stream + non-stream)."""
        if self.path != "/v1/chat/completions":
            self._send_json(404, error_response(f"not found: {self.path}"))
            return
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(content_length)

        wants_stream = False
        try:
            wants_stream = bool(json.loads(raw.decode("utf-8")).get("stream"))
        except Exception:
            wants_stream = False

        if wants_stream:
            status, payload, chunks = handle_request_body(raw, stream=True)
            if status != 200:
                self._send_json(status, payload or {})
                return
            self._send_sse(chunks)
            return

        status, payload = handle_request_body(raw)
        self._send_json(status, payload)

    # -- writers ----------------------------------------------------------

    def _send_sse(self, chunks: List[Dict]) -> None:
        """Write an SSE stream of ``data: {json}\n\n`` frames."""
        lines = []
        for chunk in chunks:
            lines.append(f"data: {json.dumps(chunk)}\n\n")
        lines.append("data: [DONE]\n\n")
        body = "".join(lines).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name: str, ok_on_missing: bool = False) -> None:
        """Serve a file from ``web/`` (issue #43)."""
        path = os.path.join(_WEB_ROOT, name)
        if not os.path.isfile(path):
            if ok_on_missing:
                self._send_json(404, error_response(f"not found: {name}"))
                return
            self._send_json(404, error_response(f"not found: {name}"))
            return
        with open(path, "rb") as f:
            body = f.read()
        ext = os.path.splitext(name)[1].lower()
        ctype = _MIME.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Dict) -> None:
        """Write ``payload`` as a JSON response with the given status."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        """Keep request logs on stdout (default BaseHTTPRequestHandler logs to stderr)."""
        print(f"[x8d-openai] {self.address_string()} - {fmt % args}")


# ---------------------------------------------------------------------------
# Telemetry + low-RAM disk mode
# ---------------------------------------------------------------------------

def _telemetry_payload() -> Dict:
    """Serialize the Colibrì-style telemetry collector for /telemetry."""
    snap = _TELEMETRY.snapshot()
    snap["mode"] = _SERVER_MODE
    snap["io_mb"] = round(int(snap["io_bytes"]) / 1e6, 2)
    snap["fault_mb"] = round(int(snap["fault_bytes"]) / 1e6, 2)
    return snap


def _open_disk_repo(repo_dir: str) -> None:
    """Open the low-RAM disk-backed mode (issue #45).

    Maps every ``.gguf``/``.x8dds.gguf`` in ``repo_dir`` through
    ``MappedX8DReader`` so completions are served by slicing payloads out of
    the kernel page cache — no full-RAM load, no GPU.
    """
    from omni_diffusion.x8d_mmap import MappedX8DReader  # noqa: E402

    global _DISK_READER, _SERVER_MODE
    ggufs = sorted(
        f for f in os.listdir(repo_dir)
        if f.endswith((".gguf", ".x8dds.gguf"))
    )
    if not ggufs:
        raise SystemExit(f"no .gguf/.x8dds.gguf containers in {repo_dir}")
    first = os.path.join(repo_dir, ggufs[0])
    _DISK_READER = MappedX8DReader(first)
    _SERVER_MODE = "disk"
    print(f"[x8d] low-RAM disk mode: mmap {first} "
          f"({_DISK_READER.size_bytes / 1e6:.1f} MB mapped, zero-copy)")


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    disk_repo: Optional[str] = None,
) -> None:
    """Start the threaded HTTP server and serve forever.

    Args:
        host: bind address.
        port: bind port.
        disk_repo: optional directory of x8D containers for low-RAM
            from-disk serving (issue #45).
    """
    if disk_repo:
        _open_disk_repo(disk_repo)
    server = ThreadingHTTPServer((host, port), ChatCompletionHandler)
    print(f"x8D OpenAI-compatible endpoint + web UI on http://{host}:{port}")
    print(f"  GET  /                     -> ChatGPT-style web UI")
    print(f"  GET  /healthz              -> {{status: ok, mode: {_SERVER_MODE}}}")
    print(f"  GET  /telemetry            -> I/O + RSS dashboard")
    print(f"  GET  /v1/models            -> model list")
    print(f"  POST /v1/chat/completions  -> chat completion (stream + non-stream)")
    print(f"  mode: {_SERVER_MODE}" + (f"  disk: {_DISK_READER.filename}" if _DISK_READER else ""))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if _DISK_READER is not None:
            _DISK_READER.close()


def main() -> int:
    """CLI entry point: ``python3 tools/openai_chat_server.py --port 666``."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible byte-diffusion chat endpoint + web UI "
        "(pure stdlib, low-RAM disk mode available)"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--disk-repo", default=None,
        help="directory of x8D .gguf/.x8dds.gguf containers for low-RAM "
             "from-disk serving (no GPU, ~1 GB RAM target)",
    )
    args = parser.parse_args()
    run_server(args.host, args.port, args.disk_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
