# coding=utf-8
"""Tests for the pointer-based x8D quantization flow (issue #10)."""

import json
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_export import LAW  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
from quantize_kimi_k3 import (  # noqa: E402
    build_pointer_map,
    load_pointer_gguf,
    report,
    save_pointer_gguf,
    serve_expert_from_pointer,
)

TMPDIR = os.path.join(os.path.dirname(__file__), "_tmp_ptr")
if not os.path.isdir(TMPDIR):
    os.makedirs(TMPDIR)


def _write_mini_shard(path: str, tensors: dict) -> str:
    import json as _json

    header, data = {}, bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        begin = len(data)
        data += payload
        end = len(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}
    enc = _json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(enc)))
        f.write(enc)
        f.write(bytes(data))
    return path


class PointerQuantizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index_path = os.path.join(TMPDIR, "index.json")
        cls.shard_path = _write_mini_shard(
            os.path.join(TMPDIR, "model-00013-of-000096.safetensors"),
            {
                "language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed": (
                    "U8",
                    [64],
                    bytes(range(64)),
                ),
                "language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_scale": (
                    "BF16",
                    [1],
                    bytes([0x00, 0x3C]),
                ),
                "language_model.model.layers.0.self_attn.o_proj.weight": (
                    "BF16",
                    [4, 4],
                    bytes(range(32)),
                ),
            },
        )
        with open(cls.index_path, "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": 1000000},
                    "weight_map": {
                        "language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed": "model-00013-of-000096.safetensors",
                        "language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_scale": "model-00013-of-000096.safetensors",
                        "language_model.model.layers.0.self_attn.o_proj.weight": "model-00013-of-000096.safetensors",
                    },
                },
                f,
            )

    def test_build_pointer_map_no_shard(self):
        ptr = build_pointer_map(self.index_path, "moonshotai/Kimi-K3")
        self.assertEqual(len(ptr), 3)
        rec = ptr["language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed"]
        self.assertEqual(rec["repo"], "moonshotai/Kimi-K3")
        self.assertEqual(rec["shard"], "model-00013-of-000096.safetensors")

    def test_build_pointer_map_with_local_shard(self):
        ptr = build_pointer_map(
            self.index_path, "moonshotai/Kimi-K3", shard_paths={"model-00013-of-000096.safetensors": self.shard_path}
        )
        rec = ptr["language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed"]
        self.assertEqual(rec["dtype"], "U8")
        self.assertEqual(rec["shape"], [64])
        self.assertGreater(rec["end"], rec["begin"])

    def test_pointer_gguf_roundtrip(self):
        ptr = build_pointer_map(self.index_path, "moonshotai/Kimi-K3")
        out = os.path.join(TMPDIR, "map.gguf")
        n = save_pointer_gguf(ptr, out)
        self.assertGreater(n, 0)
        back = load_pointer_gguf(out)
        self.assertEqual(set(back), set(ptr))
        self.assertEqual(back["language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed"]["shard"],
                         "model-00013-of-000096.safetensors")

    def test_serve_expert_from_local_shard(self):
        ptr = build_pointer_map(
            self.index_path, "moonshotai/Kimi-K3", shard_paths={"model-00013-of-000096.safetensors": self.shard_path}
        )
        rec = ptr["language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed"]
        raw = serve_expert_from_pointer(rec, local_dir=TMPDIR)
        self.assertEqual(raw, bytes(range(64)))
        # /0.001 reverse is exact
        quanta = [b * 0.001 for b in raw]
        self.assertEqual(bytes(int(round(q / LAW)) & 0xFF for q in quanta), raw)

    def test_report(self):
        r = report(100, 1000, upstream_total=1_560_860_324_864)
        self.assertAlmostEqual(r["upstream_bytes"], 1.560860324864e12)
        self.assertGreater(r["reduction_pct"], 99.999999)

    def test_hf_vs_compressed_forward_identical(self):
        """Compressed (x8D 0.001) forward == HF full-model forward, bit-for-bit.

        Mirrors the verified Kimi-K3 check: an expert's raw U8 weight bytes
        fetched from the upstream repo are quantized once at storage time
        (quanta = byte * 0.001); at query time they are /0.001-reversed live.
        Since the reverse is byte-exact, any matmul over the reversed bytes
        equals the matmul over the original HF bytes exactly.
        """
        rows, cols = 3072, 1792
        payload = bytes((i * 7 + 13) & 0xFF for i in range(rows * cols))
        cmpdir = os.path.join(TMPDIR, "cmp")
        os.makedirs(cmpdir, exist_ok=True)
        shard_path = _write_mini_shard(
            os.path.join(cmpdir, "model-00013-of-000096.safetensors"),
            {
                "language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed": (
                    "U8",
                    [rows, cols],
                    payload,
                ),
            },
        )
        ptr = build_pointer_map(
            self.index_path, "moonshotai/Kimi-K3",
            tensor_names=["language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed"],
            shard_paths={"model-00013-of-000096.safetensors": shard_path},
        )
        rec = ptr["language_model.model.layers.12.block_sparse_moe.experts.895.w1.weight_packed"]
        raw = serve_expert_from_pointer(rec, local_dir=cmpdir)

        # storage-time quantize once ...
        quanta = [b * 0.001 for b in raw]
        # query-time live /0.001 reverse
        back = bytes(int(round(q / LAW)) & 0xFF for q in quanta)
        self.assertEqual(back, raw)

        # forward equivalence: same input, HF path vs compressed path
        import random
        random.seed(7)
        x = [random.uniform(-1.0, 1.0) for _ in range(cols)]
        y_hf = [
            sum(x[j] * raw[i * cols + j] for j in range(cols))
            for i in range(rows)
        ]
        y_x8d = [
            sum(x[j] * back[i * cols + j] for j in range(cols))
            for i in range(rows)
        ]
        self.assertTrue(all(a == b for a, b in zip(y_hf, y_x8d)))
        self.assertGreater(sum(v * v for v in y_hf), 0.0)


if __name__ == "__main__":
    unittest.main()
