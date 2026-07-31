# coding=utf-8
"""Tests for the x8Dsub-byte 0.001 compression + x8D .gguf container export."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_export import (  # noqa: E402
    GGUF_MAGIC,
    LAW,
    X8DHeaderError,
    dequantize,
    load_gguf,
    mmap_load_gguf,
    percent_reduction,
    quantize,
    save_gguf,
    to_u8,
    verify_framework_alignment,
)


class QuantaTest(unittest.TestCase):
    def test_law_is_0_001(self):
        self.assertEqual(LAW, 0.001)

    def test_quantize_applies_0_001_scale(self):
        self.assertEqual(quantize([128, 255, 0]), [0.128, 0.255, 0.0])

    def test_dequantize_inverts_0_001_scale(self):
        self.assertEqual(dequantize([0.128, 0.255, 0.0]), b"\x80\xff\x00")

    def test_to_u8_projects_back_losslessly(self):
        weights = bytes(range(256))
        self.assertEqual(to_u8(quantize(weights)), weights)

    def test_quantize_dequantize_roundtrip_all_bytes(self):
        weights = os.urandom(4096)
        self.assertEqual(dequantize(quantize(weights)), weights)

    def test_percent_reduction(self):
        # BF16 stores 2 bytes per weight byte; x8D stores 1.
        self.assertAlmostEqual(percent_reduction(2.0, 1.0), 50.0)


class GGUFContainerTest(unittest.TestCase):
    def _tmpfile(self):
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.remove, path)
        return path

    def test_save_and_load_roundtrip(self):
        path = self._tmpfile()
        save_gguf({"weights": b"\x00\x01\xfe\xff"}, path)
        payloads, meta = load_gguf(path)
        self.assertEqual(payloads["weights"], b"\x00\x01\xfe\xff")
        self.assertEqual(meta["law"], 0.001)

    def test_save_int_list_payload_byte_identical(self):
        path = self._tmpfile()
        save_gguf({"w": [0, 128, 200, 255]}, path)
        payloads, _ = load_gguf(path)
        self.assertEqual(payloads["w"], bytes([0, 128, 200, 255]))

    def test_save_quanta_float_payload_lossless(self):
        path = self._tmpfile()
        weights = bytes([0, 1, 64, 128, 200, 255])
        quanta = quantize(weights)
        save_gguf({"w": list(quanta)}, path)
        payloads, _ = load_gguf(path)
        self.assertEqual(payloads["w"], weights)

    def test_save_empty_container(self):
        path = self._tmpfile()
        save_gguf({}, path)
        payloads, _ = load_gguf(path)
        self.assertEqual(payloads, {})

    def test_multi_payload(self):
        path = self._tmpfile()
        save_gguf({"a": b"\x01", "b": b"\x02\x03", "c": b"\x04\x05\x06"}, path)
        payloads, _ = load_gguf(path)
        self.assertEqual(payloads, {"a": b"\x01", "b": b"\x02\x03", "c": b"\x04\x05\x06"})

    def test_stores_raw_u8_no_json_pollution(self):
        path = self._tmpfile()
        save_gguf({"w": b"\x00\xff"}, path)
        with open(path, "rb") as f:
            blob = f.read()
        self.assertTrue(blob.startswith(GGUF_MAGIC))
        self.assertNotIn(b"{", blob)
        self.assertNotIn(b'"', blob)

    def test_bad_magic_raises(self):
        path = self._tmpfile()
        with open(path, "wb") as f:
            f.write(b"NOTX8D...")
        with self.assertRaises(X8DHeaderError):
            load_gguf(path)

    def test_mmap_zero_copy_load(self):
        path = self._tmpfile()
        payload = os.urandom(1_000_000)
        save_gguf({"big": payload}, path)
        mapping, meta = mmap_load_gguf(path)
        try:
            self.assertEqual(mapping[: len(GGUF_MAGIC)], GGUF_MAGIC)
            self.assertEqual(meta["size_bytes"], os.path.getsize(path))
        finally:
            mapping.close()


class FrameworkAlignmentTest(unittest.TestCase):
    def test_verify_small_alignment(self):
        result = verify_framework_alignment(data_size=65_536)
        self.assertTrue(result["lossless"])
        self.assertEqual(result["effective_ratio"], 1.0)
        self.assertEqual(result["original_size"], 65_536)


if __name__ == "__main__":
    unittest.main()
