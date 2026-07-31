# coding=utf-8
"""Tests for byte-native HF dataset import + 8x8 block compression."""

import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_dataset import (  # noqa: E402
    MAGIC,
    X8DDatasetError,
    block_compress_dataset,
    byte_stream_to_rows,
    field_to_bytes,
    flatten_fields,
    read_manifest,
    resolve_hf_dataset,
    rows_to_byte_stream,
)
from omni_diffusion.x8d_export import load_gguf  # noqa: E402

try:
    resolve_hf_dataset("sarvamai/indivibe", length=1)
    _NETWORK_OK = True
except Exception:
    _NETWORK_OK = False


class FieldToBytesTest(unittest.TestCase):
    def test_str_utf8(self):
        self.assertEqual(field_to_bytes("नमस्ते"), "नमस्ते".encode("utf-8"))
        self.assertEqual(field_to_bytes(""), b"")

    def test_bool(self):
        self.assertEqual(field_to_bytes(True), b"\x01")
        self.assertEqual(field_to_bytes(False), b"\x00")

    def test_int_boundaries(self):
        self.assertEqual(field_to_bytes(0), struct.pack("<Q", 0))
        self.assertEqual(field_to_bytes(2**64 - 1), struct.pack("<Q", 2**64 - 1))
        with self.assertRaises(X8DDatasetError):
            field_to_bytes(2**64)
        with self.assertRaises(X8DDatasetError):
            field_to_bytes(-1)

    def test_float(self):
        self.assertEqual(field_to_bytes(1.5), struct.pack("<d", 1.5))
        self.assertEqual(field_to_bytes(-0.25), struct.pack("<d", -0.25))

    def test_bytes_types(self):
        self.assertEqual(field_to_bytes(b"\x00\xff"), b"\x00\xff")
        self.assertEqual(field_to_bytes(bytearray(b"\x01\x02")), b"\x01\x02")

    def test_nested_and_unencodable(self):
        for value in ([], [1], {}, {"a": 1}, (1, 2), None, object()):
            self.assertIsNone(field_to_bytes(value))


class FlattenFieldsTest(unittest.TestCase):
    def test_nested_dict_list_paths(self):
        row = {
            "messages": [{"role": "user", "content": "hi"}],
            "meta": {"score": 1.5, "ok": True},
            "ids": [10, 20],
        }
        fields = dict(flatten_fields(row))
        self.assertEqual(fields["messages[0].role"], b"user")
        self.assertEqual(fields["messages[0].content"], b"hi")
        self.assertEqual(fields["meta.score"], struct.pack("<d", 1.5))
        self.assertEqual(fields["meta.ok"], b"\x01")
        self.assertEqual(fields["ids[0]"], struct.pack("<Q", 10))
        self.assertEqual(fields["ids[1]"], struct.pack("<Q", 20))

    def test_skips_none(self):
        fields = flatten_fields({"a": None, "b": 5})
        self.assertEqual(fields, [("b", struct.pack("<Q", 5))])


class RowStreamRoundtripTest(unittest.TestCase):
    def test_roundtrip_unicode_and_emoji(self):
        rows = [{"text": "नमस्ते", "emoji": "🚀", "n": 7}]
        self.assertEqual(byte_stream_to_rows(rows_to_byte_stream(rows)), rows)

    def test_roundtrip_empty_rows(self):
        self.assertEqual(byte_stream_to_rows(rows_to_byte_stream([])), [])

    def test_roundtrip_empty_string_vs_bytes(self):
        rows = [{"s": "", "b": b""}]
        self.assertEqual(byte_stream_to_rows(rows_to_byte_stream(rows)), rows)

    def test_roundtrip_scalar_types(self):
        rows = [
            {"n": 2**64 - 1, "f": 3.5, "flag": False, "raw": b"\x00\xff", "txt": "hi"},
            {"nested": {"list": [1, 2]}},
        ]
        self.assertEqual(byte_stream_to_rows(rows_to_byte_stream(rows)), rows)

    def test_magic_prefix(self):
        self.assertTrue(rows_to_byte_stream([{"a": 1}]).startswith(MAGIC))

    def test_bad_magic_raises(self):
        with self.assertRaises(X8DDatasetError):
            byte_stream_to_rows(b"NOTX8D....")
        with self.assertRaises(X8DDatasetError):
            byte_stream_to_rows(b"")

    def test_truncation_raises(self):
        stream = rows_to_byte_stream([{"a": "hello"}, {"b": "world"}])
        for cut in (len(MAGIC), len(MAGIC) + 4, len(MAGIC) + 8, len(stream) - 1):
            with self.assertRaises(X8DDatasetError):
                byte_stream_to_rows(stream[:cut])


class BlockCompressTest(unittest.TestCase):
    def _tmpdir(self):
        path = tempfile.mkdtemp(prefix="x8dds_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def test_block_compress_manifest_and_lossless(self):
        out = self._tmpdir()
        rows = [
            {"text": "नमस्ते", "n": 42},
            {"messages": [{"role": "user", "content": "hi"}], "ok": True},
        ]
        manifest = block_compress_dataset(rows, out, "sarvamai_indivibe", seed=0)
        for key in (
            "dataset",
            "config",
            "split",
            "rows_count",
            "stream_bytes",
            "gguf",
            "gguf_bytes",
            "method",
            "threshold",
            "created",
            "roundtrip_lossless",
        ):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["rows_count"], 2)
        self.assertEqual(manifest["method"], "x8d-spec-8x8")
        self.assertEqual(manifest["threshold"], 0.001)
        self.assertTrue(manifest["roundtrip_lossless"])

        stream = rows_to_byte_stream(rows)
        self.assertEqual(manifest["stream_bytes"], len(stream))
        payloads, _ = load_gguf(os.path.join(out, manifest["gguf"]))
        self.assertEqual(payloads["sarvamai_indivibe"], stream)

        on_disk = read_manifest(out)
        self.assertEqual(on_disk["roundtrip_lossless"], True)
        self.assertEqual(on_disk["rows_count"], 2)


@unittest.skipUnless(_NETWORK_OK, "network required")
class ResolveHFDatasetLiveTest(unittest.TestCase):
    def test_resolve_indivibe_chat_test(self):
        result = resolve_hf_dataset(
            "sarvamai/indivibe", config="chat", split="test", length=5
        )
        self.assertEqual(result["dataset"], "sarvamai/indivibe")
        self.assertEqual(result["config"], "chat")
        self.assertEqual(result["split"], "test")
        self.assertGreater(len(result["rows"]), 0)
        self.assertIsInstance(result["num_rows_total"], int)

    def test_flatten_and_roundtrip_live_rows(self):
        result = resolve_hf_dataset(
            "sarvamai/indivibe", config="chat", split="test", length=5
        )
        rebuilt = byte_stream_to_rows(rows_to_byte_stream(result["rows"]))
        self.assertEqual(rebuilt, result["rows"])
        self.assertTrue(any(flatten_fields(r) for r in result["rows"]))


if __name__ == "__main__":
    unittest.main()
