# coding=utf-8
"""Tests for x8d_hf.py (HF shard -> x8D .gguf) and moe_disk.py (on-disk MoE)."""

import json
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.moe_disk import X8DGgufReader, MoEOnDisk  # noqa: E402
from omni_diffusion.x8d_export import save_gguf  # noqa: E402
from omni_diffusion.x8d_hf import (  # noqa: E402
    SafetensorsShard,
    convert_shard_to_gguf,
    dtype_nbytes,
    expected_span_length,
    load_pointer_map,
    parse_safetensors_header,
)

TMPDIR = os.path.join(os.path.dirname(__file__), "_tmp_x8dhf")
if not os.path.isdir(TMPDIR):
    os.makedirs(TMPDIR)


def _make_shard(path: str, tensors: dict):
    """Write a minimal valid safetensors shard (no package needed)."""
    header = {}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        begin = len(data)
        data += payload
        end = len(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}
    enc = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(enc)))
        f.write(enc)
        f.write(bytes(data))


class X8DHfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shard = os.path.join(TMPDIR, "shard.safetensors")
        # 3 tensors: one big expert w1 (4x4), w2 (4x2), and a 1D bias
        _make_shard(
            cls.shard,
            {
                "layers.0.experts.3.w1": ("BF16", [4, 4], bytes(range(16))),
                "layers.0.experts.3.w2": ("BF16", [4, 2], bytes(range(8))),
                "model.lm_head.weight": ("BF16", [8], bytes(range(8))),
            },
        )

    def test_parse_safetensors_header(self):
        with open(self.shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = parse_safetensors_header(f.read(hlen))
        self.assertIn("layers.0.experts.3.w1", header)

    def test_shard_offsets_and_read(self):
        s = SafetensorsShard(self.shard)
        try:
            offs = s.tensor_offsets("layers.0.experts.3.w1")
            self.assertEqual(offs, (8 + len(json.dumps(s.index).encode()), 8 + len(json.dumps(s.index).encode()) + 16))
            self.assertEqual(s.read_tensor("layers.0.experts.3.w1"), bytes(range(16)))
        finally:
            s.close()

    def test_convert_shard_to_gguf_subset(self):
        out = os.path.join(TMPDIR, "subset.gguf")
        path, stats = convert_shard_to_gguf(
            self.shard, out, tensor_names=["layers.0.experts.3.w1"]
        )
        self.assertEqual(stats["tensors"], 1)
        self.assertEqual(stats["bytes_in"], 16)
        self.assertEqual(stats["bytes_out"], 16)
        with open(path, "rb") as f:
            blob = f.read()
        # magic-free: no X8DGGUF1 prefix
        self.assertNotIn(b"X8DGGUF1", blob)
        # header-free framing: <u32 name_len><name><u64 data_len><data>
        self.assertEqual(
            os.path.getsize(path),
            stats["bytes_out"] + 4 + len("layers.0.experts.3.w1") + 8,
        )

    def test_convert_shard_to_gguf_all(self):
        out = os.path.join(TMPDIR, "all.gguf")
        path, stats = convert_shard_to_gguf(self.shard, out)
        self.assertEqual(stats["tensors"], 3)
        self.assertEqual(stats["bytes_in"], 16 + 8 + 8)

    def test_convert_spec_decode_mode(self):
        out = os.path.join(TMPDIR, "spec.gguf")
        path, stats = convert_shard_to_gguf(
            self.shard, out, tensor_names=["layers.0.experts.3.w1"], spec_decode=True
        )
        self.assertEqual(stats["tensors"], 1)


class MoEDiskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gguf = os.path.join(TMPDIR, "moe.gguf")
        # expert w1: 4x4 identity-ish (bytes encode row-major)
        w1 = bytes([128 + 64] * 16)  # constant 0.5 after normalize
        w2 = bytes(range(8))
        save_gguf(
            {
                "layers.0.experts.3.w1": w1,
                "layers.0.experts.3.w2": w2,
                "layers.0.experts.7.w1": bytes(range(16)),
            },
            cls.gguf,
        )

    def test_reader_index(self):
        r = X8DGgufReader(self.gguf)
        try:
            self.assertEqual(len(r.names()), 3)
            self.assertTrue(r.contains("layers.0.experts.3.w1"))
            self.assertEqual(r.tensor_size("layers.0.experts.3.w1"), 16)
        finally:
            r.close()

    def test_load_expert_live_reverse(self):
        m = MoEOnDisk(self.gguf)
        try:
            w = m.load_expert(0, 3, "w1", shape=(4, 4))
            self.assertEqual(len(w), 16)
            # /0.001 reverse of stored quanta: (b*0.001)/0.001 == b
            self.assertEqual(w, [0x80 + 64] * 16)
        finally:
            m.close()

    def test_expert_not_found(self):
        m = MoEOnDisk(self.gguf)
        try:
            self.assertFalse(m.expert_available(1, 0, "w1"))
            with self.assertRaises(KeyError):
                m.load_expert(1, 0, "w1")
        finally:
            m.close()

    def test_matmul_fp32_cpu(self):
        m = MoEOnDisk(self.gguf)
        try:
            hidden = [1.0, 2.0, 3.0, 4.0]
            out = m.matmul_fp32(0, 3, hidden, "w1")
            self.assertEqual(len(out), 4)
            # constant weight 0.5, sum = 1+2+3+4 = 10, * 0.5 = 5
            self.assertAlmostEqual(out[0], 5.0, places=6)
        finally:
            m.close()

    def test_gguf_roundtrip_through_reader(self):
        r = X8DGgufReader(self.gguf)
        try:
            self.assertEqual(r.tensor_bytes("layers.0.experts.7.w1"), bytes(range(16)))
        finally:
            r.close()

    def test_size_mb_returns_float(self):
        # regression #20: size_mb must return a number, not a bound method
        m = MoEOnDisk(self.gguf)
        try:
            v = m.size_mb()
            self.assertIsInstance(v, float)
            self.assertGreater(v, 0.0)
        finally:
            m.close()


class DtypeSpanTest(unittest.TestCase):
    def test_dtype_nbytes(self):
        self.assertEqual(dtype_nbytes("F32"), 4)
        self.assertEqual(dtype_nbytes("bf16"), 2)
        self.assertEqual(dtype_nbytes("U8"), 1)
        self.assertEqual(dtype_nbytes("FP4"), None)
        self.assertEqual(dtype_nbytes("nonsense"), None)

    def test_expected_span_length(self):
        self.assertEqual(expected_span_length([2, 3, 4], "F32"), 96)
        self.assertEqual(expected_span_length([256, 256], "BF16"), 131072)
        self.assertEqual(expected_span_length([4, 4], "nonsense"), None)

    def test_load_pointer_map_local(self):
        tmp = os.path.join(TMPDIR, "ptr.gguf")
        ptrs = {
            "a": {"repo": "r", "shard": "s", "begin": 0, "end": 4, "dtype": "U8", "shape": [4]},
            "b": {"repo": "r", "shard": "s", "begin": 4, "end": 12, "dtype": "F32", "shape": [2]},
        }
        import sys as _sys

        _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
        from quantize_kimi_k3 import save_pointer_gguf  # noqa: E402

        save_pointer_gguf(ptrs, tmp)
        loaded = load_pointer_map("unused", "unused", local_path=tmp)
        self.assertEqual(loaded, ptrs)
        os.remove(tmp)


if __name__ == "__main__":
    unittest.main()
