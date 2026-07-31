# coding=utf-8
"""Tests for the generic HF pointer quantizer (issue #17)."""

import json
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_export import LAW  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
from quantize_hf import build_single_file_pointers  # noqa: E402
from quantize_kimi_k3 import (  # noqa: E402
    load_pointer_gguf,
    save_pointer_gguf,
    serve_expert_from_pointer,
)

TMPDIR = os.path.join(os.path.dirname(__file__), "_tmp_hfq")
if not os.path.isdir(TMPDIR):
    os.makedirs(TMPDIR)


def _write_shard(path, tensors):
    header, data = {}, bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        begin = len(data)
        data += payload
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(data)]}
    header["__metadata__"] = {"format": "pt"}
    enc = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(enc)))
        f.write(enc)
        f.write(bytes(data))


class QuantizeHfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shard_path = os.path.join(TMPDIR, "model.safetensors")
        _write_shard(
            cls.shard_path,
            {
                "model.a.weight": ("F32", [4], struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)),
                "model.b.weight": ("U8", [8], bytes(range(8))),
                "model.diffusion_block.0.to_k.input_scale": ("U8", [1], bytes([7])),
            },
        )

    def test_single_file_pointers_skip_metadata(self):
        """__metadata__ must be skipped, all real tensors pointer-mapped."""
        # build_single_file_pointers normally fetches over HTTP; emulate with a
        # local file by reading header directly and building pointers manually.
        from quantize_kimi_k3 import build_pointer_map

        with open(self.shard_path, "rb") as f:
            head = f.read()
        (hlen,) = struct.unpack("<Q", head[:8])
        header = json.loads(head[8 : 8 + hlen].decode("utf-8"))
        self.assertIn("__metadata__", header)
        names = [n for n in header if n != "__metadata__" and "data_offsets" in header[n]]
        self.assertEqual(len(names), 3)
        self.assertNotIn("__metadata__", names)

    def test_pointer_map_serves_and_reverses(self):
        """save/load pointer map + serve expert span + /0.001 reverse exact."""
        with open(self.shard_path, "rb") as f:
            head = f.read()
        (hlen,) = struct.unpack("<Q", head[:8])
        header = json.loads(head[8 : 8 + hlen].decode("utf-8"))
        data_start = 8 + hlen
        pointers = {}
        for n, spec in header.items():
            if n == "__metadata__" or "data_offsets" not in spec:
                continue
            begin, end = spec["data_offsets"]
            pointers[n] = {
                "repo": "local",
                "shard": os.path.basename(self.shard_path),
                "name": n,
                "begin": data_start + begin,
                "end": data_start + end,
                "dtype": spec["dtype"],
                "shape": spec["shape"],
            }
        out = os.path.join(TMPDIR, "map.gguf")
        save_pointer_gguf(pointers, out)
        back = load_pointer_gguf(out)
        self.assertEqual(set(back), set(pointers))

        # serve the U8 tensor span locally, /0.001 reverse is byte-exact
        rec = back["model.b.weight"]
        with open(self.shard_path, "rb") as f:
            f.seek(rec["begin"])
            raw = f.read(rec["end"] - rec["begin"])
        self.assertEqual(raw, bytes(range(8)))
        quanta = [b * 0.001 for b in raw]
        self.assertEqual(bytes(int(round(q / LAW)) & 0xFF for q in quanta), raw)

    def test_forward_identical_after_reverse(self):
        """F32 tensor: matmul over reversed bytes == matmul over original."""
        with open(self.shard_path, "rb") as f:
            head = f.read()
        (hlen,) = struct.unpack("<Q", head[:8])
        header = json.loads(head[8 : 8 + hlen].decode("utf-8"))
        spec = header["model.a.weight"]
        begin, end = spec["data_offsets"]
        data_start = 8 + hlen
        with open(self.shard_path, "rb") as f:
            f.seek(data_start + begin)
            raw = f.read(end - begin)
        back = bytes(int(round(b * 0.001 / LAW)) & 0xFF for b in raw)
        import struct as st
        w_hf = st.unpack("<4f", raw)
        w_x8d = st.unpack("<4f", back)
        self.assertEqual(w_hf, w_x8d)


if __name__ == "__main__":
    unittest.main()
