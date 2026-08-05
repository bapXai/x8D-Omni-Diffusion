# coding=utf-8
"""Tests for the zero-copy mmap x8D container reader (issue #41).

Pure stdlib unittest. Verifies the Colibrì ``COLI_MMAP=1``-style path over the
sub-byte container: offset-index addressing, zero-copy views, live /0.001
reverse, X8DDS stream frame walking, and clean resource handling.
"""

import os
import struct
import sys
import tempfile
import unittest

from omni_diffusion.x8d_export import save_gguf
from omni_diffusion.x8d_mmap import (
    BLOCK_SIZE,
    X8DDS_MAGIC,
    MappedX8DReader,
    build_payload_index,
    reverse_bytes,
    X8DMmapError,
)
from omni_diffusion.x8d_dataset import rows_to_byte_stream
from omni_diffusion.x8d_spec_decode import dspark_batch_mask, DSparkMaskConfig
from omni_diffusion.x8d_export import LAW


def _make_container(payloads):
    fd, path = tempfile.mkstemp(suffix=".gguf")
    os.close(fd)
    save_gguf(payloads, path)
    return path


def _make_x8dds_container(rows):
    stream = rows_to_byte_stream(rows)
    return _make_container({"dataset": stream}), stream


class BuildPayloadIndexTest(unittest.TestCase):
    def test_empty_index(self):
        self.assertEqual(build_payload_index(b""), {})

    def test_single_payload(self):
        body = struct.pack("<I", 4) + b"abcd" + struct.pack("<Q", 5) + b"hello"
        index = build_payload_index(body, base=16)
        self.assertEqual(index, {"abcd": (16 + 4 + 4 + 8, 5)})

    def test_truncated_fields_raise(self):
        with self.assertRaises(X8DMmapError):
            build_payload_index(struct.pack("<I", 4) + b"ab")
        with self.assertRaises(X8DMmapError):
            build_payload_index(struct.pack("<I", 10) + b"ab")


class MappedX8DReaderTest(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, "_tmp_paths", []):
            try:
                os.remove(path)
            except OSError:
                pass

    def test_load_roundtrip(self):
        path = _make_container({"w1": bytes(range(256)), "w2": b"\x00\xff"})
        self._tmp_paths = [path]
        with MappedX8DReader(path) as r:
            self.assertEqual(r.names(), ["w1", "w2"])
            self.assertEqual(r.load("w1"), bytes(range(256)))
            self.assertEqual(r.load("w2"), b"\x00\xff")
            offset, length = r.offsets("w1")
            self.assertEqual(length, 256)
            # header-free container: "w1" payload sits at 4 + 2 + 8
            self.assertEqual(offset, 4 + 2 + 8)
            self.assertEqual(r.slice_at(offset, 256), bytes(range(256)))

    def test_zero_copy_view(self):
        path = _make_container({"w": b"abcdefgh"})
        self._tmp_paths = [path]
        with MappedX8DReader(path) as r:
            view = r.view("w")
            self.assertIsInstance(view, memoryview)
            self.assertEqual(bytes(view), b"abcdefgh")
            view.release()

    def test_live_reverse_law(self):
        path = _make_container({"w": b"\x01\xff\x80"})
        self._tmp_paths = [path]
        with MappedX8DReader(path) as r:
            coords = r.reverse("w")
            self.assertEqual(coords, [1.0 / LAW, 255.0 / LAW, 128.0 / LAW])
        self.assertEqual(reverse_bytes(b"\x02"), [2.0 / LAW])

    def test_bad_magic_rejected(self):
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(b"NOTX8D" + b"\x00" * 10)
        self._tmp_paths = [path]
        with self.assertRaises(Exception):
            MappedX8DReader(path)

    def test_slice_out_of_range(self):
        path = _make_container({"w": b"ab"})
        self._tmp_paths = [path]
        with MappedX8DReader(path) as r:
            with self.assertRaises(X8DMmapError):
                r.slice_at(0, r.size_bytes + 1)

    def test_unknown_payload(self):
        path = _make_container({"w": b"ab"})
        self._tmp_paths = [path]
        with MappedX8DReader(path) as r:
            with self.assertRaises(KeyError):
                r.offsets("nope")

    def test_mmap_matches_load_gguf(self):
        from omni_diffusion.x8d_export import load_gguf

        path = _make_container({"a": bytes(range(64)), "b": b"z" * 1000})
        self._tmp_paths = [path]
        payloads, _ = load_gguf(path)
        with MappedX8DReader(path) as r:
            for name, expected in payloads.items():
                self.assertEqual(r.load(name), expected)

    def test_closed_mapping_errors(self):
        path = _make_container({"w": b"ab"})
        self._tmp_paths = [path]
        r = MappedX8DReader(path)
        r.close()
        with self.assertRaises(Exception):
            _ = r.load("w")


class X8DDsStreamTest(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, "_tmp_paths", []):
            try:
                os.remove(path)
            except OSError:
                pass

    def test_stream_header(self):
        rows = [{"text": "hello"}, {"text": "world"}]
        path, stream = _make_x8dds_container(rows)
        self._tmp_paths = [path]
        self.assertTrue(stream.startswith(X8DDS_MAGIC))
        with MappedX8DReader(path) as r:
            row_count, data_offset = r.stream_header("dataset")
            self.assertEqual(row_count, 2)
            self.assertTrue(data_offset > len(X8DDS_MAGIC) + 8)

    def test_frame_walk_block_aligned(self):
        rows = [{"text": "x" * 200}]
        path, _ = _make_x8dds_container(rows)
        self._tmp_paths = [path]
        with MappedX8DReader(path) as r:
            frames = list(r.frames("dataset", block_size=BLOCK_SIZE))
            self.assertTrue(frames)
            for offset, view in frames:
                self.assertEqual(len(view), BLOCK_SIZE)
                self.assertTrue(offset >= 16)
                self.assertIsInstance(view, memoryview)
                view.release()

    def test_not_a_stream_raises(self):
        path = _make_container({"w": b"plain bytes"})
        self._tmp_paths = [path]
        with MappedX8DReader(path) as r:
            with self.assertRaises(X8DMmapError):
                r.stream_header("w")

    def test_mmap_slices_feed_dspark(self):
        rows = [{"text": "byte-native canvas " * 20}]
        path, stream = _make_x8dds_container(rows)
        self._tmp_paths = [path]
        cfg = DSparkMaskConfig()
        with MappedX8DReader(path) as r:
            frames = list(r.frames("dataset", block_size=BLOCK_SIZE))
            self.assertTrue(frames)
            offset, view = frames[0]
            generated = dspark_batch_mask([bytes(view)], cfg, seed=7)
            self.assertEqual(len(generated), 1)
            self.assertEqual(len(generated[0]), BLOCK_SIZE)
            self.assertEqual(r.slice_at(offset, BLOCK_SIZE), bytes(view))
            for _, v in frames:
                v.release()


if __name__ == "__main__":
    unittest.main()
