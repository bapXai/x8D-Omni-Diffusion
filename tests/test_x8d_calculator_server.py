"""Socket-free integration tests for the System-1 calculator tool endpoint."""
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.openai_chat_server import ChatCompletionHandler, process_chat_completion


def run_post(path: str, obj: object) -> tuple[int, dict]:
    """Invoke the real HTTP handler in memory without binding a socket."""
    raw = json.dumps(obj).encode("utf-8")
    return run_raw_post(path, raw)


def run_raw_post(path: str, raw: bytes, content_length: str | None = None) -> tuple[int, dict]:
    h = object.__new__(ChatCompletionHandler)
    h.command = "POST"
    h.path = path
    h.requestline = f"POST {path} HTTP/1.0"
    h.request_version = "HTTP/1.0"
    h.protocol_version = "HTTP/1.0"
    h.headers = {"Content-Length": str(len(raw)) if content_length is None else content_length}
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.log_message = lambda *args, **kwargs: None
    h.do_POST()
    data = h.wfile.getvalue()
    head, _, response = data.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, json.loads(response.decode("utf-8"))


class UnreadableBody:
    def read(self, *args):
        raise AssertionError("unexpected request-body read")


class CalculatorHttpTest(unittest.TestCase):
    def test_tool_endpoint_exact(self):
        code, result = run_post("/v1/tools/calculate", {"expression": "20% of 150"})
        self.assertEqual(code, 200)
        self.assertEqual(result["result"], "30")
        self.assertEqual(result["fraction"], "30")
        self.assertFalse(result["rounded"])

    def test_tool_endpoint_handles_errors(self):
        code, result = run_post("/v1/tools/calculate", {"expression": "1/0"})
        self.assertEqual(code, 400)
        self.assertEqual(result["error"]["code"], "division_by_zero")
        code, result = run_post("/v1/tools/calculate", {"expression": "1+1", "shell": "whoami"})
        self.assertEqual(code, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_invalid_content_length_before_read(self):
        for length in ("-1", "invalid"):
            with self.subTest(length=length):
                with patch(
                    __name__ + ".io.BytesIO", side_effect=[UnreadableBody(), io.BytesIO()]
                ):
                    code, result = run_raw_post("/v1/tools/calculate", b"", length)
                self.assertEqual(code, 400)
                self.assertEqual(result["error"]["code"], "invalid_request")

    def test_raw_json_errors_and_size_limit(self):
        nested = b"[" * 1000 + b"0" + b"]" * 1000
        self.assertLessEqual(len(nested), 2048)
        previous_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(200)
            # Python 3.14's C decoder has a separate recursion budget. Use
            # the stdlib Python scanner to exercise real decoder exhaustion
            # consistently across runtimes without exceeding the body cap.
            decoder = json.JSONDecoder()
            decoder.scan_once = json.scanner.py_make_scanner(decoder)
            with patch.object(json, "_default_decoder", decoder):
                with self.assertRaises(RecursionError):
                    json.loads(nested)
                for raw in (nested, b"\xff", b"{"):
                    with self.subTest(body=raw[:10]):
                        code, result = run_raw_post("/v1/tools/calculate", raw)
                        self.assertEqual(code, 400)
                        self.assertEqual(result["error"]["code"], "invalid_request")
        finally:
            sys.setrecursionlimit(previous_limit)
        code, result = run_raw_post("/v1/tools/calculate", b" " * 2049)
        self.assertEqual(code, 413)
        self.assertEqual(result["error"]["code"], "resource_limit")

    def test_chat_math_bypasses_untrained_sampler(self):
        for request, expected in [
            ("What is 3 * 7?", "21"),
            ("0.1 + 0.2", "0.3"),
            ("calculate 1 / 0", "Calculator error: division by zero"),
        ]:
            with self.subTest(request=request):
                response = process_chat_completion({
                    "messages": [{"role": "user", "content": request}],
                })
                content = response["choices"][0]["message"]["content"]
                self.assertEqual(content, expected)
                self.assertEqual(response["usage"]["completion_tokens"], len(content.encode("utf-8")))

    def test_non_math_stays_on_existing_model_path(self):
        response = process_chat_completion({
            "messages": [{"role": "user", "content": "Describe a game strategy"}],
        })
        self.assertIn("x8D says", response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
