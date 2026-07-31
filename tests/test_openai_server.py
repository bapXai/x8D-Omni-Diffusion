# coding=utf-8
"""Tests for the OpenAI-compatible byte-native chat endpoint (#29).

Offline (no socket): exercises ``process_chat_completion``,
``handle_request_body``, ``build_models_response`` and the byte pipeline
directly. Pure Python stdlib unittest.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.openai_chat_server import (  # noqa: E402
    ChatCompletionError,
    MODEL_ID,
    build_models_response,
    byte_pipeline,
    error_response,
    extract_last_user_message,
    handle_request_body,
    process_chat_completion,
)


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


class ChatCompletionCoreTest(unittest.TestCase):
    def test_valid_request_returns_openai_shape(self):
        resp = process_chat_completion(_valid_body("hi"))
        for key in ("id", "object", "created", "model", "choices", "usage"):
            self.assertIn(key, resp)
        self.assertTrue(resp["id"].startswith("chatcmpl-"))
        self.assertEqual(resp["object"], "chat.completion")
        self.assertIsInstance(resp["created"], int)
        self.assertEqual(resp["model"], MODEL_ID)
        choice = resp["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIsInstance(choice["message"]["content"], str)
        self.assertEqual(choice["finish_reason"], "stop")
        usage = resp["usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIsInstance(usage[key], int)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    def test_content_is_deterministic(self):
        a = process_chat_completion(_valid_body("ping"))
        b = process_chat_completion(_valid_body("ping"))
        self.assertEqual(a["choices"][0]["message"]["content"], b["choices"][0]["message"]["content"])

    def test_last_user_message_extracted(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "prev"},
            {"role": "user", "content": "final"},
        ]
        self.assertEqual(extract_last_user_message(messages), "final")
        self.assertIsNone(extract_last_user_message([{"role": "system", "content": "x"}]))

    def test_usage_is_byte_counts(self):
        body = _valid_body("héllo🙂")
        expected_prompt = len("You are a byte-native assistant.".encode("utf-8")) + len("héllo🙂".encode("utf-8"))
        resp = process_chat_completion(body)
        self.assertEqual(resp["usage"]["prompt_tokens"], expected_prompt)
        self.assertEqual(resp["usage"]["completion_tokens"], len(resp["choices"][0]["message"]["content"].encode("utf-8")))

    def test_max_tokens_truncates_by_bytes(self):
        resp = process_chat_completion(_valid_body("short"))
        content = resp["choices"][0]["message"]["content"]
        resp2 = process_chat_completion(_valid_body("short"))
        limited = process_chat_completion({**_valid_body("short"), "max_tokens": 10})
        limited_content = limited["choices"][0]["message"]["content"]
        self.assertLessEqual(len(limited_content.encode("utf-8")), 10)
        self.assertNotEqual(limited_content, content)
        # does not split mid-UTF-8-codepoint
        resp2["choices"][0]["message"]["content"].encode("utf-8")


class ErrorHandlingTest(unittest.TestCase):
    def test_malformed_json_maps_to_400_error(self):
        status, payload = handle_request_body(b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("malformed JSON", payload["error"]["message"])

    def test_non_object_body_raises(self):
        with self.assertRaises(ChatCompletionError):
            process_chat_completion([1, 2, 3])

    def test_missing_messages_maps_to_400_error(self):
        with self.assertRaises(ChatCompletionError) as ctx:
            process_chat_completion({"model": MODEL_ID})
        self.assertIn("messages", str(ctx.exception))
        status, payload = handle_request_body(b'{"model": "x8d"}')
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    def test_empty_messages_rejected(self):
        with self.assertRaises(ChatCompletionError):
            process_chat_completion({"model": MODEL_ID, "messages": []})

    def test_stream_true_unsupported(self):
        with self.assertRaises(ChatCompletionError) as ctx:
            process_chat_completion({**_valid_body(), "stream": True})
        self.assertEqual(ctx.exception.error_type, "unsupported")
        status, payload = handle_request_body(
            json_dumps({**_valid_body(), "stream": True})
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "unsupported")

    def test_error_response_shape(self):
        err = error_response("boom", "invalid_request_error")
        self.assertEqual(err, {"error": {"message": "boom", "type": "invalid_request_error"}})


class ModelsAndHealthTest(unittest.TestCase):
    def test_models_response_shape(self):
        resp = build_models_response()
        self.assertEqual(resp["object"], "list")
        self.assertEqual(len(resp["data"]), 1)
        model = resp["data"][0]
        self.assertEqual(model["id"], MODEL_ID)
        self.assertEqual(model["object"], "model")
        self.assertIsInstance(model["created"], int)
        self.assertEqual(model["owned_by"], "bapX")

    def test_stream_false_accepted(self):
        resp = process_chat_completion({**_valid_body(), "stream": False})
        self.assertEqual(resp["object"], "chat.completion")


class BytePipelineTest(unittest.TestCase):
    def test_pipeline_returns_str(self):
        out = byte_pipeline("Hello, byte world")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_pipeline_multibyte(self):
        out = byte_pipeline("नमस्ते 世界 🙂")
        self.assertIsInstance(out, str)

    def test_pipeline_deterministic(self):
        self.assertEqual(byte_pipeline("abc"), byte_pipeline("abc"))

    def test_message_text_roundtrips_to_str_content(self):
        for text in ("hi", "What is the weather?", "नमस्ते"):
            resp = process_chat_completion(_valid_body(text))
            content = resp["choices"][0]["message"]["content"]
            self.assertIsInstance(content, str)
            self.assertIn(text, content)


def json_dumps(obj) -> bytes:
    import json

    return json.dumps(obj).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
