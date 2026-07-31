# coding=utf-8
"""OpenAI-compatible chat endpoint for the byte-native x8D pipeline.

Pure Python standard library only (``http.server``, ``json``, ``time``,
``uuid``, ``argparse``) — no torch, no transformers, no tokenizer. Mirrors
the concept of the upstream ``bapXai/x8Dsub-byte`` openai_chat_server.py
but grounded in the byte law: vocabulary is 264 (bytes 0-255 + specials
256-263), ``usage`` is reported in BYTES not tokens, and each prompt is
pushed through the encode -> mask -> denoise -> decode byte pipeline.

The request-handling core (``process_chat_completion``,
``handle_request_body``, ``build_models_response``) is importable and
testable WITHOUT binding a port.

Endpoints:
    GET  /healthz            -> {"status": "ok"}
    GET  /v1/models          -> {"object": "list", "data": [{...}]}
    POST /v1/chat/completions -> OpenAI-shaped completion (stream unsupported)

Run:  python3 tools/openai_chat_server.py --port 666
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

DEFAULT_PORT: int = 666
DEFAULT_HOST: str = "0.0.0.0"

MODEL_ID: str = "x8d-byte-diffusion"
MODEL_OWNER: str = "bapX"

#: Fixed model creation timestamp (module import time).
_MODEL_CREATED: int = int(time.time())

#: Diffusion canvas length / denoise steps mirroring the query-pipeline tests.
CANVAS_STEPS: int = 48
_CANVAS_SEED: int = 0


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

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        spec = importlib.util.spec_from_file_location(
            "x8d_server_query_sampler", os.path.join(repo_root, "tests", "test_queries.py")
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
    denoised = _SAMPLER.denoise(canvas, steps=CANVAS_STEPS, seed=_CANVAS_SEED)
    return _TOKENIZER.decode_text(denoised)


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
    denoised = byte_pipeline(last_user)
    return (
        f"x8D says: '{last_user}'. Byte-law pipeline (vocab=264, no tokens) "
        f"denoised canvas -> {denoised!r}"
    )


def process_chat_completion(body: Dict) -> Dict:
    """Turn an OpenAI chat-completions body into a response dict.

    Args:
        body: parsed JSON request body.

    Returns:
        OpenAI-shaped ``chat.completion`` dict. ``usage`` counts are BYTE
        counts (prompt/completion/total), never token counts.

    Raises:
        ChatCompletionError: malformed body, missing/invalid ``messages``,
            or ``stream=true`` (unsupported).
    """
    if not isinstance(body, dict):
        raise ChatCompletionError("request body must be a JSON object")

    if body.get("stream"):
        raise ChatCompletionError("streaming is not supported", error_type="unsupported")

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


def handle_request_body(raw: bytes) -> Tuple[int, Dict]:
    """Parse and process a raw request body; never raises.

    Args:
        raw: raw request body bytes.

    Returns:
        ``(status_code, response_payload)`` — 200 with the completion, or
        400 with an OpenAI-style error dict (malformed JSON, missing
        messages, unsupported stream).
    """
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return 400, error_response(f"malformed JSON: {exc}")
    try:
        return 200, process_chat_completion(body)
    except ChatCompletionError as exc:
        return 400, error_response(exc.message, exc.error_type)


# ---------------------------------------------------------------------------
# HTTP handler + server
# ---------------------------------------------------------------------------

class ChatCompletionHandler(BaseHTTPRequestHandler):
    """``BaseHTTPRequestHandler`` serving the OpenAI-compatible endpoints."""

    def do_GET(self) -> None:
        """Route GET: /v1/models and /healthz."""
        if self.path == "/v1/models":
            self._send_json(200, build_models_response())
        elif self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(
                404, error_response(f"not found: {self.path}")
            )

    def do_POST(self) -> None:
        """Route POST: /v1/chat/completions."""
        if self.path != "/v1/chat/completions":
            self._send_json(404, error_response(f"not found: {self.path}"))
            return
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(content_length)
        status, payload = handle_request_body(raw)
        self._send_json(status, payload)

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


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Start the threaded HTTP server and serve forever."""
    server = ThreadingHTTPServer((host, port), ChatCompletionHandler)
    print(f"x8D OpenAI-compatible endpoint on http://{host}:{port}")
    print(f"  GET  /healthz             -> {{status: ok}}")
    print(f"  GET  /v1/models           -> model list")
    print(f"  POST /v1/chat/completions -> chat completion (byte pipeline)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    """CLI entry point: ``python3 tools/openai_chat_server.py --port 666``."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible byte-diffusion chat endpoint (pure stdlib)"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
