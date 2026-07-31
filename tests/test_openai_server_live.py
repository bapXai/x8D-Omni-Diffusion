# coding=utf-8
"""Live-style tests for the OpenAI-compatible byte endpoint (issue #39).

Offline-first: exercises the importable request core
(``process_chat_completion`` / ``handle_request_body`` /
``build_models_response``), the real ``ChatCompletionHandler`` routing
through a socket-free fake (an ``object.__new__`` handler with a BytesIO
``wfile``), and the omni probe's on-disk active-expert report. A true HTTP
round-trip on an ephemeral port is gated behind
``@unittest.skipUnless(_SOCKET_OK, ...)``.

Pure Python stdlib unittest — no torch, no transformers, no socket by
default. Existing tests in ``test_openai_server.py`` stay untouched.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.openai_chat_server import (  # noqa: E402
    MODEL_ID,
    MODEL_OWNER,
    byte_pipeline,
    error_response,
    handle_request_body,
    process_chat_completion,
)
from tools.omni_chat_probe import (  # noqa: E402
    EXPERT_BY_MODALITY,
    SHARED_EXPERT,
    active_expert_report,
    build_demo_experts,
    detect_modality,
    expert_table,
    offline_probe,
)
from omni_diffusion.moe_disk import SARARouter  # noqa: E402


def _valid_body(message="Hello byte world"):
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a byte-native assistant."},
            {"role": "user", "content": message},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }


# ---------------------------------------------------------------------------
# Socket-free handler driver (tests real do_GET / do_POST routing)
# ---------------------------------------------------------------------------

def _run_handler(method: str, path: str, raw_body: bytes = b"") -> Tuple[int, Dict]:
    """Drive ``ChatCompletionHandler.do_GET/do_POST`` without a socket.

    Builds a bare handler instance via ``object.__new__``, wires a BytesIO
    ``wfile``, and parses the status line + JSON body from the response
    bytes. No port is ever bound.
    """
    from tools.openai_chat_server import ChatCompletionHandler

    handler = object.__new__(ChatCompletionHandler)
    handler.command = method
    handler.path = path
    handler.requestline = f"{method} {path} HTTP/1.0"
    handler.request_version = "HTTP/1.0"
    handler.protocol_version = "HTTP/1.0"
    handler.headers = {"Content-Length": str(len(raw_body))}
    handler.rfile = io.BytesIO(raw_body)
    handler.wfile = io.BytesIO()
    handler.log_message = lambda *args, **kwargs: None
    if method == "GET":
        handler.do_GET()
    else:
        handler.do_POST()
    raw = handler.wfile.getvalue()
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("latin-1")
    status = int(status_line.split()[1])
    payload = json.loads(body.decode("utf-8"))
    return status, payload


# ---------------------------------------------------------------------------
# Socket probe (used only by the gated live test)
# ---------------------------------------------------------------------------

def _socket_ok() -> bool:
    """Return True when a real loopback HTTP round-trip succeeds."""
    from tools.openai_chat_server import ChatCompletionHandler

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ChatCompletionHandler)
    except OSError:
        return False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/healthz", timeout=5) as resp:
            return resp.status == 200 and json.loads(resp.read().decode("utf-8")) == {"status": "ok"}
    except Exception:
        return False
    finally:
        server.shutdown()
        server.server_close()


_SOCKET_OK = _socket_ok()


# ---------------------------------------------------------------------------
# Offline tests
# ---------------------------------------------------------------------------

class HealthzRoutingTest(unittest.TestCase):
    """GET routing through the real handler without a socket."""

    def test_healthz_shape(self):
        status, payload = _run_handler("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

    def test_models_shape(self):
        status, payload = _run_handler("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "list")
        model = payload["data"][0]
        self.assertEqual(model["id"], MODEL_ID)
        self.assertEqual(model["object"], "model")
        self.assertEqual(model["owned_by"], MODEL_OWNER)
        self.assertIsInstance(model["created"], int)

    def test_unknown_path_404(self):
        status, payload = _run_handler("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, error_response("not found: /nope"))


class ChatCompletionHttpTest(unittest.TestCase):
    """POST routing through the real handler without a socket."""

    def test_post_completion_through_handler(self):
        status, payload = _run_handler("POST", "/v1/chat/completions", json.dumps(_valid_body("hi")).encode())
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertIn("hi", payload["choices"][0]["message"]["content"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIsInstance(payload["usage"][key], int)

    def test_post_malformed_json_400(self):
        status, payload = _run_handler("POST", "/v1/chat/completions", b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("malformed JSON", payload["error"]["message"])

    def test_post_stream_true_400(self):
        raw = json.dumps({**_valid_body(), "stream": True}).encode()
        status, payload = _run_handler("POST", "/v1/chat/completions", raw)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "unsupported")

    def test_post_empty_messages_400(self):
        raw = json.dumps({"model": MODEL_ID, "messages": []}).encode()
        status, payload = _run_handler("POST", "/v1/chat/completions", raw)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    def test_post_unknown_path_404(self):
        status, payload = _run_handler("POST", "/v1/other", json.dumps(_valid_body()).encode())
        self.assertEqual(status, 404)


class UsageBytesTest(unittest.TestCase):
    """usage is reported in BYTES (byte law: tokens == bytes)."""

    def test_usage_keys_are_byte_counts(self):
        body = _valid_body("héllo🙂")
        expected_prompt = len("You are a byte-native assistant.".encode("utf-8")) + len("héllo🙂".encode("utf-8"))
        resp = process_chat_completion(body)
        usage = resp["usage"]
        self.assertEqual(usage["prompt_tokens"], expected_prompt)
        self.assertEqual(
            usage["completion_tokens"], len(resp["choices"][0]["message"]["content"].encode("utf-8"))
        )
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    def test_byte_keys_note(self):
        # The wire uses OpenAI's *_tokens keys, but their values are raw byte
        # counts — there is no tokenizer in this pipeline (vocab = 264 bytes).
        resp = process_chat_completion(_valid_body("bytes not tokens"))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIn(key, resp["usage"])
            self.assertIsInstance(resp["usage"][key], int)


class RoleValidationTest(unittest.TestCase):
    """How the handler treats roles / non-str content."""

    def test_system_only_messages_still_complete(self):
        resp = process_chat_completion(
            {"model": MODEL_ID, "messages": [{"role": "system", "content": "sys"}]}
        )
        self.assertEqual(resp["object"], "chat.completion")
        # no user message -> last-user extraction yields "" (empty prompt)
        content = resp["choices"][0]["message"]["content"]
        self.assertIn("''", content)

    def test_last_user_message_wins(self):
        resp = process_chat_completion(
            {
                "model": MODEL_ID,
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "middle"},
                    {"role": "user", "content": "final"},
                ],
            }
        )
        self.assertIn("final", resp["choices"][0]["message"]["content"])
        self.assertNotIn("first", resp["choices"][0]["message"]["content"])

    def test_non_str_content_coerced(self):
        resp = process_chat_completion(
            {
                "model": MODEL_ID,
                "messages": [
                    {"role": "user", "content": 42},
                    {"role": "user", "content": None},
                ],
            }
        )
        content = resp["choices"][0]["message"]["content"]
        self.assertIsInstance(content, str)
        # None content -> "" is the last user message
        self.assertIn("''", content)

    def test_unknown_role_tolerated(self):
        resp = process_chat_completion(
            {
                "model": MODEL_ID,
                "messages": [
                    {"role": "mystery", "content": "ignored"},
                    {"role": "user", "content": "actual"},
                ],
            }
        )
        self.assertIn("actual", resp["choices"][0]["message"]["content"])

    def test_non_dict_message_entries_skipped(self):
        resp = process_chat_completion(
            {"model": MODEL_ID, "messages": [["bad", "entry"], {"role": "user", "content": "ok"}]}
        )
        self.assertIn("ok", resp["choices"][0]["message"]["content"])


class BytePipelineLiveTest(unittest.TestCase):
    """The encode -> mask -> denoise -> decode path used per request."""

    def test_pipeline_returns_deterministic_str(self):
        self.assertIsInstance(byte_pipeline("hello"), str)
        self.assertEqual(byte_pipeline("abc"), byte_pipeline("abc"))

    def test_reference_sampler_contract(self):
        from tools.openai_chat_server import _load_reference_sampler

        sampler_cls = _load_reference_sampler()
        sampler = sampler_cls()
        self.assertTrue(callable(sampler.denoise))
        canvas = [256, 97, 256]  # MASK 'a' MASK
        out = sampler.denoise(canvas, steps=48, seed=0)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(0 <= b <= 263 for b in out))


class ActiveExpertReportTest(unittest.TestCase):
    """On-disk MoE active-param report (MoEOnDisk + SARA stand-in, #39/#36)."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="x8d-live-")
        cls.gguf = os.path.join(cls.tmpdir, "moe_demo.x8d.gguf")
        cls.counts = build_demo_experts(cls.gguf)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_expert_isolation(self):
        # every modality maps to a distinct, self-contained expert block
        blocks = list(EXPERT_BY_MODALITY.values())
        self.assertEqual(len(blocks), len(set(blocks)))
        self.assertEqual(SHARED_EXPERT, (0, 4))

    def test_active_expert_report_text(self):
        rep = active_expert_report(self.gguf, "text")
        self.assertEqual(rep["modality"], "text")
        self.assertEqual(rep["sara"]["customer"], "kimi-k3")
        self.assertEqual(rep["sara"]["mode"], "moe")
        self.assertEqual(rep["routing_key"], "layers.0.experts.0")
        self.assertEqual(rep["routed_params"], self.counts["text"])
        self.assertEqual(rep["shared_params"], self.counts["shared"])
        self.assertEqual(rep["active_params"], self.counts["text"] + self.counts["shared"])
        self.assertTrue(rep["reverse_exact"])

    def test_sara_routing_modality_map(self):
        router = SARARouter()
        self.assertEqual(router.route("text").customer, "kimi-k3")
        self.assertEqual(router.route("language").customer, "kimi-k3")
        self.assertEqual(router.route("image").customer, "ltx2")
        self.assertEqual(router.route("video").customer, "ltx2")
        self.assertEqual(router.route("audio").customer, "whisper-large-v3")
        self.assertEqual(router.route("asr").customer, "whisper-large-v3")
        self.assertEqual(router.route("tts").customer, "kokoro-82m")
        # direct customer key short-circuits to its own boundary
        self.assertEqual(router.route("kimi-k3").customer, "kimi-k3")

    def test_sara_boundaries_isolated(self):
        router = SARARouter()
        for a in ("kimi-k3", "ltx2", "whisper-large-v3", "kokoro-82m"):
            for b in ("kimi-k3", "ltx2", "whisper-large-v3", "kokoro-82m"):
                self.assertTrue(router.is_isolated(a, b))
        # dense models are single-expert, MoE models register their own pool
        dense = {r.customer for r in router.registry if r.mode == "dense"}
        self.assertEqual(dense, {"kokoro-82m", "whisper-large-v3", "ltx2"})
        self.assertEqual(router.active_params("kimi-k3"), 104_200_000_000)

    def test_all_modalities_route(self):
        table = expert_table(self.gguf)
        self.assertEqual([row["modality"] for row in table], ["text", "image", "audio", "video"])
        for row in table:
            self.assertEqual(row["active_params"], self.counts["text"] + self.counts["shared"])
        by_mod = {row["modality"]: row["customer"] for row in table}
        self.assertEqual(by_mod["text"], "kimi-k3")
        self.assertEqual(by_mod["image"], "ltx2")
        self.assertEqual(by_mod["audio"], "whisper-large-v3")

    def test_unknown_modality_raises(self):
        with self.assertRaises(KeyError):
            active_expert_report(self.gguf, "telepathy")

    def test_detect_modality_markers(self):
        self.assertEqual(detect_modality("plain text"), "text")
        self.assertEqual(detect_modality("draw [IMG_START] a cat"), "image")
        self.assertEqual(detect_modality("transcribe [AUD_START] this clip"), "audio")

    def test_offline_probe_through_request_core(self):
        report = offline_probe("ping via probe")
        self.assertEqual(report["mode"], "offline")
        self.assertEqual(report["healthz"]["body"], {"status": "ok"})
        self.assertEqual(report["completion"]["object"], "chat.completion")
        self.assertEqual(report["completion"]["model"], MODEL_ID)
        self.assertIn("ping via probe", report["completion"]["choices"][0]["message"]["content"])
        self.assertIn("usage", report["completion"])


@unittest.skipUnless(_SOCKET_OK, "ephemeral HTTP socket not available")
class LiveSocketTest(unittest.TestCase):
    """Real HTTP round-trip on an ephemeral port (network-gated)."""

    def test_full_http_round_trip(self):
        from tools.omni_chat_probe import serve_probe

        report = serve_probe("hello through a real socket")
        self.assertEqual(report["mode"], "http")
        self.assertEqual(report["healthz"]["status_code"], 200)
        self.assertEqual(report["healthz"]["body"], {"status": "ok"})
        self.assertEqual(report["models"]["status_code"], 200)
        model = report["models"]["body"]["data"][0]
        self.assertEqual(model["id"], MODEL_ID)
        self.assertEqual(model["owned_by"], MODEL_OWNER)
        self.assertEqual(report["completion_status"], 200)
        comp = report["completion"]
        self.assertEqual(comp["object"], "chat.completion")
        self.assertEqual(comp["model"], MODEL_ID)
        self.assertIn("usage", comp)
        self.assertIsNotNone(report["latency_ms"])


if __name__ == "__main__":
    unittest.main()
