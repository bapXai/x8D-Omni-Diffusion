"""Socket-free integration tests for the System-1 calculator tool endpoint."""
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.openai_chat_server import ChatCompletionHandler, process_chat_completion


def run_post(path: str, obj: object) -> tuple[int, dict]:
    """Invoke the real HTTP handler in memory without binding a socket."""
    raw = json.dumps(obj).encode("utf-8")
    h = object.__new__(ChatCompletionHandler)
    h.command = "POST"
    h.path = path
    h.requestline = f"POST {path} HTTP/1.0"
    h.request_version = "HTTP/1.0"
    h.protocol_version = "HTTP/1.0"
    h.headers = {"Content-Length": str(len(raw))}
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.log_message = lambda *args, **kwargs: None
    h.do_POST()
    data = h.wfile.getvalue()
    head, _, response = data.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, json.loads(response.decode("utf-8"))


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
