# coding=utf-8
"""Tests for DSpark-style speculative quantization and size comparison."""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_export import (  # noqa: E402
    GGUF_MAGIC,
    LAW,
    load_gguf,
    save_gguf,
)
from omni_diffusion.x8d_spec_decode import (  # noqa: E402
    BLOCK_SIZE,
    CONFIDENCE_THRESHOLD,
    DEFAULT_VERIFY_LEN,
    SpeculativeDecodeError,
    confidence_head_probe,
    print_size_report,
    size_report,
    speculative_quantize,
    speculative_save_gguf,
    _block_surrogate,
    _split_blocks,
    _verify_positions,
)


class SpeculativeQuantizeTest(unittest.TestCase):
    def test_block_split_exact_and_padded(self):
        blocks = _split_blocks(bytes(range(BLOCK_SIZE)))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), BLOCK_SIZE)
        # padded tail
        padded = _split_blocks(bytes(range(10)))
        self.assertEqual(len(padded[0]), BLOCK_SIZE)
        self.assertEqual(padded[0][:10], bytes(range(10)))
        self.assertTrue(all(b == 0 for b in padded[0][10:]))

    def test_confidence_in_range(self):
        for i in range(64):
            c = _block_surrogate(bytes([i & 0xFF]) * BLOCK_SIZE, step=1)
            self.assertTrue(0.0 <= c < 1.0)

    def test_verify_positions_threshold(self):
        quanta = [0.5] * 8
        confidence = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        failed = _verify_positions(quanta, confidence)
        # all above 0.001 threshold => nothing re-masked
        self.assertEqual(failed, [])
        low = [0.0005] * 8
        failed = _verify_positions(quanta, low)
        self.assertEqual(len(failed), 8)

    def test_verify_positions_clip(self):
        quanta = [0.5] * 32
        confidence = [0.0005] * 32
        failed = _verify_positions(
            quanta, confidence, heavy_load=True, verify_len=16
        )
        self.assertEqual(failed, list(range(16)))
        full = _verify_positions(quanta, confidence, verify_len=16)
        self.assertEqual(full, list(range(16)))

    def test_speculative_quantize_converges_lossless(self):
        data = os.urandom(1000)
        quanta, stats = speculative_quantize(data, max_steps=32, seed=42)
        restored = bytes(
            int(round(q / LAW)) & 0xFF for q in quanta
        )
        self.assertEqual(restored[: len(data)], data)
        self.assertEqual(stats["blocks"], (1000 + BLOCK_SIZE - 1) // BLOCK_SIZE)
        self.assertEqual(stats["converged"], stats["blocks"])

    def test_speculative_quantize_deterministic(self):
        data = os.urandom(512)
        q1, _ = speculative_quantize(data, max_steps=16, seed=7)
        q2, _ = speculative_quantize(data, max_steps=16, seed=7)
        self.assertEqual(q1, q2)

    def test_speculative_save_gguf_roundtrip(self):
        data = os.urandom(512)
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_spec.gguf")
        try:
            path, stats = speculative_save_gguf("w1", data, tmp, seed=3)
            payloads, meta = load_gguf(path)
            self.assertEqual(meta["law"], LAW)
            self.assertEqual(payloads["w1"], data)
            self.assertGreaterEqual(stats["converged"], 1)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_size_report_16b_fp16(self):
        r = size_report(num_params=16_000_000_000, baseline_bits=16)
        self.assertAlmostEqual(r["baseline_size_gb"], 32.0, places=6)
        self.assertAlmostEqual(r["x8d_storage_gb"], 16.0, places=6)
        # 16B params * (16 * 0.001) bits = 256 Mbit = 32 MB
        self.assertAlmostEqual(r["subbyte_coordinate_mb"], 32.0, places=6)
        self.assertAlmostEqual(r["disk_reduction_pct"], 50.0, places=6)
        self.assertAlmostEqual(r["coordinate_reduction_pct"], 99.9, places=6)

    def test_size_report_small(self):
        r = size_report(num_params=1000, baseline_bits=16)
        self.assertAlmostEqual(r["baseline_size_gb"], 2e-6, places=9)
        self.assertAlmostEqual(r["x8d_storage_gb"], 1e-6, places=9)

    def test_confidence_probe_override(self):
        class _Surrogate:
            def __call__(self, block, step):
                return 0.0

        block = bytes(BLOCK_SIZE)
        self.assertEqual(confidence_head_probe(block, 0), _block_surrogate(block, 0))
        self.assertEqual(confidence_head_probe(block, 0, _Surrogate()), 0.0)

    def test_print_size_report(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_size_report(num_params=1_000_000, baseline_bits=16)
        out = buf.getvalue()
        self.assertIn("1,000,000 params, 16-bit baseline", out)
        self.assertIn("disk reduction 50.0%", out)
        self.assertIn("0.016 bit/weight", out)


if __name__ == "__main__":
    unittest.main()
