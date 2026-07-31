# coding=utf-8
"""Tests for framing dataset rows into fixed-length byte-diffusion canvases."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.models.dream.byte_tokenizer import (  # noqa: E402
    AUD_END_TOKEN_ID,
    AUD_START_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    IMG_END_TOKEN_ID,
    IMG_START_TOKEN_ID,
    PAD_TOKEN_ID,
)
from omni_diffusion.x8d_dataset import flatten_fields  # noqa: E402
from omni_diffusion.x8d_dataset_canvas import (  # noqa: E402
    canvases_to_bytes,
    rows_to_canvases,
)


def _flat_bytes(rows):
    """Concatenated flattened field bytes for a list of rows (expected)."""
    out = bytearray()
    for row in rows:
        for _path, data in flatten_fields(row):
            out.extend(data)
    return bytes(out)


class RowsToCanvasesTest(unittest.TestCase):
    def test_canvases_all_have_canvas_length(self):
        rows = [
            {"text": "hello world", "audio": (bytes(range(256)) * 2)[:300]},
            {"text": "नमस्ते", "image": b"\x89PNG\x0d\x0a"},
        ]
        canvases = rows_to_canvases(rows, canvas_length=64, with_markers=False)
        self.assertGreater(len(canvases), 0)
        for c in canvases:
            self.assertEqual(len(c), 64)
        self.assertEqual(len(canvases) * 64, sum(len(c) for c in canvases))

    def test_total_length_is_multiple_of_canvas_length(self):
        rows = [{"a": "x" * 100}, {"b": b"y" * 100}]
        canvases = rows_to_canvases(rows, canvas_length=32, with_markers=False)
        total = sum(len(c) for c in canvases)
        self.assertEqual(total % 32, 0)

    def test_pad_padding_present_on_tail(self):
        rows = [{"text": "abc"}]  # 3 bytes, not a canvas multiple
        canvases = rows_to_canvases(rows, canvas_length=16, with_markers=False)
        self.assertEqual(len(canvases), 1)
        self.assertEqual(len(canvases[0]), 16)
        self.assertIn(PAD_TOKEN_ID, canvases[0])
        self.assertEqual(canvases[0][3:], [PAD_TOKEN_ID] * 13)

    def test_no_padding_when_exact_multiple(self):
        rows = [{"text": "a" * 64}]
        canvases = rows_to_canvases(rows, canvas_length=64, with_markers=False)
        self.assertEqual(len(canvases), 1)
        self.assertNotIn(PAD_TOKEN_ID, canvases[0])

    def test_empty_rows_produce_no_canvases(self):
        self.assertEqual(rows_to_canvases([], canvas_length=32), [])

    def test_markers_present_when_with_markers(self):
        rows = [
            {"text": "hi", "image": b"\x01\x02", "audio": b"\x03\x04"},
        ]
        canvases = rows_to_canvases(rows, canvas_length=64, with_markers=True)
        blob = [b for c in canvases for b in c]
        self.assertIn(BOS_TOKEN_ID, blob)
        self.assertIn(EOS_TOKEN_ID, blob)
        self.assertIn(IMG_START_TOKEN_ID, blob)
        self.assertIn(IMG_END_TOKEN_ID, blob)
        self.assertIn(AUD_START_TOKEN_ID, blob)
        self.assertIn(AUD_END_TOKEN_ID, blob)

    def test_markers_absent_when_without_markers(self):
        rows = [{"text": "hi", "image": b"\x01\x02", "audio": b"\x03\x04"}]
        marker_ids = {
            BOS_TOKEN_ID,
            EOS_TOKEN_ID,
            IMG_START_TOKEN_ID,
            IMG_END_TOKEN_ID,
            AUD_START_TOKEN_ID,
            AUD_END_TOKEN_ID,
        }
        canvases = rows_to_canvases(rows, canvas_length=64, with_markers=False)
        for c in canvases:
            for b in c:
                self.assertTrue(0 <= b <= 263)
                self.assertNotIn(b, marker_ids)

    def test_all_canvas_bytes_within_0_263(self):
        rows = [
            {"text": "नमस्ते", "image": bytes(range(256)), "audio": b"\x00\xff"},
            {"num": 2**40, "pi": 3.14159, "flag": True},
        ]
        for with_markers in (True, False):
            canvases = rows_to_canvases(rows, canvas_length=32, with_markers=with_markers)
            for c in canvases:
                for b in c:
                    self.assertTrue(0 <= b <= 263)


class CanvasesToBytesTest(unittest.TestCase):
    def test_roundtrip_recovers_field_bytes(self):
        rows = [
            {"text": "hello world", "audio": (bytes(range(256)) * 2)[:300]},
            {"text": "नमस्ते", "image": b"\x89PNG\x0d\x0a", "num": 42},
        ]
        expected = _flat_bytes(rows)
        for with_markers in (True, False):
            canvases = rows_to_canvases(rows, canvas_length=37, with_markers=with_markers)
            self.assertEqual(canvases_to_bytes(canvases), expected)

    def test_roundtrip_binary_all_bytes(self):
        rows = [{"payload": bytes(range(256)) * 3}]
        expected = _flat_bytes(rows)
        for with_markers in (True, False):
            canvases = rows_to_canvases(rows, canvas_length=64, with_markers=with_markers)
            self.assertEqual(canvases_to_bytes(canvases), expected)

    def test_roundtrip_empty_and_empty_rows(self):
        self.assertEqual(canvases_to_bytes([]), b"")
        self.assertEqual(canvases_to_bytes(rows_to_canvases([], canvas_length=32)), b"")

    def test_roundtrip_nested_rows(self):
        rows = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {"items": [1, 2, 3]},
        ]
        expected = _flat_bytes(rows)
        for with_markers in (True, False):
            canvases = rows_to_canvases(rows, canvas_length=50, with_markers=with_markers)
            self.assertEqual(canvases_to_bytes(canvases), expected)

    def test_invalid_canvas_length(self):
        with self.assertRaises(ValueError):
            rows_to_canvases([{"text": "x"}], canvas_length=0)


if __name__ == "__main__":
    unittest.main()
