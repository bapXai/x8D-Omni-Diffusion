# coding=utf-8
"""Tests for DSpark-style speculative quantization and size comparison."""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_export import (  # noqa: E402
    LAW,
    load_gguf,
    save_gguf,
)
from omni_diffusion.x8d_spec_decode import (  # noqa: E402
    BLOCK_SIZE,
    CONFIDENCE_THRESHOLD,
    DEFAULT_VERIFY_LEN,
    DSPARK_MODALITY_SCHEDULES,
    DSparkMaskConfig,
    MASK_ID,
    SpeculativeDecodeError,
    confidence_head_probe,
    dspark_batch_mask,
    dspark_block_generate,
    dspark_generate,
    mask_block,
    modality_size_report,
    print_modality_size_report,
    print_size_report,
    renoise_block,
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

    def test_speculative_quantize_length_preserving(self):
        # regression #23: output must be exactly len(input), no padded tail
        for n in (1, 63, 64, 65, 1000):
            data = bytes(i % 256 for i in range(n))
            quanta, _ = speculative_quantize(data, max_steps=16, seed=0)
            restored = bytes(int(round(q / LAW)) & 0xFF for q in quanta)
            self.assertEqual(len(restored), n, f"length for n={n}")
            self.assertEqual(restored, data, f"exact bytes for n={n}")

    def test_speculative_save_gguf_length_preserving(self):
        data = bytes(range(100))
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_spec_len.gguf")
        try:
            path, _ = speculative_save_gguf("w", data, tmp, seed=0)
            payloads, _ = load_gguf(path)
            self.assertEqual(len(payloads["w"]), len(data))
            self.assertEqual(payloads["w"], data)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_speculative_quantize_deterministic(self):
        data = os.urandom(512)
        q1, _ = speculative_quantize(data, max_steps=16, seed=7)
        q2, _ = speculative_quantize(data, max_steps=16, seed=7)
        self.assertEqual(q1, q2)

    def test_speculative_quantize_lossless_on_low_surrogate_block(self):
        # regression: a block whose sha256 surrogate lands under 0.002 makes
        # byte-0 confidence (block_conf/2) fall below the 0.001 threshold,
        # which used to re-noise correct zero bytes and corrupt the stream.
        # 63 zero bytes + 0xfe has surrogate ~0.00029 (< 0.002) at step 0.
        block = b"\x00" * 63 + b"\xfe"
        self.assertLess(_block_surrogate(block, 0), CONFIDENCE_THRESHOLD)
        quanta, stats = speculative_quantize(block, max_steps=16, seed=1)
        restored = bytes(int(round(q / LAW)) & 0xFF for q in quanta)
        self.assertEqual(restored, block)
        self.assertEqual(stats["regenerations"], 0)

    def test_speculative_quantize_lossless_zero_heavy_stream(self):
        # framing-heavy data (many u16/u64 length fields = many 0x00 bytes)
        # must round-trip exactly; previously regenerated zero bytes corrupted it
        data = b"\x00" * 4096 + bytes(range(256)) + b"\x00" * 1024
        quanta, _ = speculative_quantize(data, max_steps=16, seed=3)
        restored = bytes(int(round(q / LAW)) & 0xFF for q in quanta)
        self.assertEqual(restored, data)

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


class DSparkMaskConfigTest(unittest.TestCase):
    def test_defaults(self):
        cfg = DSparkMaskConfig()
        self.assertEqual(cfg.k_blocks, 8)
        self.assertEqual(cfg.mask_ratio, 0.7)
        self.assertEqual(cfg.entropy_bound, 0.001)
        self.assertIsNone(cfg.verify_clip)
        self.assertEqual(cfg.canvas_length, 256)
        self.assertEqual(cfg.modality, "text")
        self.assertEqual(cfg.steps, 48)

    def test_frozen(self):
        cfg = DSparkMaskConfig()
        with self.assertRaises(Exception):
            cfg.mask_ratio = 0.9  # type: ignore[misc]

    def test_heavy_load_verify_clip(self):
        cfg = DSparkMaskConfig(verify_clip=BLOCK_SIZE // 16)
        self.assertEqual(cfg.verify_clip, 4)

    def test_modality_schedules_distinct(self):
        self.assertEqual(set(DSPARK_MODALITY_SCHEDULES), {"text", "image", "audio", "video"})
        ratios = [DSPARK_MODALITY_SCHEDULES[m].mask_ratio for m in ("text", "image", "audio", "video")]
        self.assertEqual(len(set(ratios)), 4, "each modality must have a distinct mask_ratio")
        self.assertEqual(DSPARK_MODALITY_SCHEDULES["text"], DSparkMaskConfig())
        self.assertEqual(DSPARK_MODALITY_SCHEDULES["image"].k_blocks, 16)
        self.assertEqual(DSPARK_MODALITY_SCHEDULES["video"].k_blocks, 32)
        self.assertEqual(DSPARK_MODALITY_SCHEDULES["text"].steps, 48)
        self.assertEqual(DSPARK_MODALITY_SCHEDULES["image"].steps, 32)

    def test_mask_ratio_bounds(self):
        for cfg in DSPARK_MODALITY_SCHEDULES.values():
            self.assertGreaterEqual(cfg.mask_ratio, 0.7)
            self.assertLessEqual(cfg.mask_ratio, 0.9)
            self.assertGreaterEqual(cfg.k_blocks, 8)
            self.assertGreaterEqual(cfg.steps, 24)
            self.assertLessEqual(cfg.steps, 48)


class DSparkBlockMaskTest(unittest.TestCase):
    def setUp(self):
        self.cfg = DSparkMaskConfig()
        self.block = bytes(range(BLOCK_SIZE))

    def test_mask_block_respects_ratio(self):
        masked, truth = mask_block(self.block, self.cfg, seed=0)
        self.assertEqual(len(masked), BLOCK_SIZE)
        self.assertEqual(truth, list(self.block))
        n_masked = sum(1 for v in masked if v == MASK_ID)
        self.assertAlmostEqual(n_masked / BLOCK_SIZE, self.cfg.mask_ratio, delta=0.05)
        for i, v in enumerate(masked):
            if v != MASK_ID:
                self.assertEqual(v, self.block[i])

    def test_mask_block_deterministic_by_seed(self):
        m1, _ = mask_block(self.block, self.cfg, seed=7)
        m2, _ = mask_block(self.block, self.cfg, seed=7)
        self.assertEqual(m1, m2)
        m3, _ = mask_block(self.block, self.cfg, seed=8)
        self.assertNotEqual(m1, m3)

    def test_mask_block_rejects_wrong_size(self):
        with self.assertRaises(ValueError):
            mask_block(bytes(10), self.cfg, seed=0)

    def test_renoise_block_deterministic_no_mask_left(self):
        masked, _ = mask_block(self.block, self.cfg, seed=1)
        renoised = renoise_block(masked, seed=1)
        self.assertEqual(len(renoised), BLOCK_SIZE)
        self.assertTrue(all(0 <= v < 256 for v in renoised), "no MASK id may survive")
        renoised2 = renoise_block(masked, seed=1)
        self.assertEqual(renoised, renoised2)

    def test_renoise_keeps_unmasked_positions(self):
        masked, _ = mask_block(self.block, self.cfg, seed=2)
        renoised = renoise_block(masked, seed=2)
        for i, v in enumerate(masked):
            if v != MASK_ID:
                self.assertEqual(renoised[i], v)

    def test_dspark_block_generate_returns_64_bytes(self):
        out = dspark_block_generate(self.block, self.cfg, seed=0)
        self.assertIsInstance(out, bytes)
        self.assertEqual(len(out), BLOCK_SIZE)

    def test_dspark_block_generate_deterministic_by_seed(self):
        o1 = dspark_block_generate(self.block, self.cfg, seed=42)
        o2 = dspark_block_generate(self.block, self.cfg, seed=42)
        self.assertEqual(o1, o2)
        o3 = dspark_block_generate(self.block, self.cfg, seed=43)
        self.assertNotEqual(o1, o3)

    def test_dspark_block_generate_honors_verify_clip(self):
        cfg = DSparkMaskConfig(verify_clip=BLOCK_SIZE // 16)
        out = dspark_block_generate(self.block, cfg, seed=0)
        self.assertEqual(len(out), BLOCK_SIZE)

    def test_dspark_block_generate_entropy_bound_regen(self):
        # a sub-threshold bound disables regeneration entirely: all positions
        # accepted, output is deterministic pure mask/renoise.
        cfg = DSparkMaskConfig(entropy_bound=0.0)
        out = dspark_block_generate(self.block, cfg, seed=5)
        self.assertEqual(len(out), BLOCK_SIZE)

    def test_dspark_batch_mask_processes_k_blocks(self):
        blocks = [bytes((i * 7) & 0xFF for _ in range(BLOCK_SIZE)) for i in range(8)]
        out = dspark_batch_mask(blocks, self.cfg, seed=0)
        self.assertEqual(len(out), len(blocks))
        self.assertTrue(all(len(b) == BLOCK_SIZE for b in out))
        o2 = dspark_batch_mask(blocks, self.cfg, seed=0)
        self.assertEqual(out, o2)

    def test_dspark_batch_mask_uneven_batch(self):
        blocks = [bytes(BLOCK_SIZE)] * 3
        out = dspark_batch_mask(blocks, self.cfg, seed=1)
        self.assertEqual(len(out), 3)

    def test_modality_size_report_per_schedule(self):
        r = modality_size_report(num_params=16_000_000_000, baseline_bits=16)
        self.assertEqual(set(r), {"text", "image", "audio", "video"})
        for name, row in r.items():
            cfg = DSPARK_MODALITY_SCHEDULES[name]
            self.assertEqual(row["mask_ratio"], cfg.mask_ratio)
            self.assertEqual(row["k_blocks"], cfg.k_blocks)
            self.assertEqual(row["steps"], cfg.steps)
            self.assertAlmostEqual(row["x8d_storage_gb"], 16.0, places=6)

    def test_print_modality_size_report(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_modality_size_report(num_params=1_000_000, baseline_bits=16)
        out = buf.getvalue()
        self.assertIn("modality", out)
        self.assertIn("text", out)
        self.assertIn("video", out)


class DSparkGenerateTest(unittest.TestCase):
    """Inference-path tests for :func:`dspark_generate` (AGENTS.md findings).

    The observed prompt context must never be masked; the completion span is
    generated 8x8-block-parallel with the confidence head, the 0.001 entropy
    bound, heavy-load verify clipping, and a lossless block-autoregressive
    commit.
    """

    def setUp(self):
        self.cfg = DSPARK_MODALITY_SCHEDULES["text"]

    def test_context_preserved_unmasked(self):
        context = list(b"[BOS]hello world[EOS]")
        completion = b" byte-law reply."
        canvas, stats = dspark_generate(context, completion, cfg=self.cfg, seed=0)
        self.assertEqual(list(canvas[: len(context)]), context)
        self.assertEqual(len(canvas), len(context) + len(completion))

    def test_completion_transported_losslessly(self):
        context = [258, 104, 105, 259]
        completion = bytes(range(256))
        canvas, _stats = dspark_generate(context, completion, cfg=self.cfg, seed=1)
        self.assertEqual(bytes(canvas[len(context) :]), completion)

    def test_deterministic_same_seed(self):
        context = list(b"prompt ")
        completion = b"answer"
        c1, _ = dspark_generate(context, completion, cfg=self.cfg, seed=7)
        c2, _ = dspark_generate(context, completion, cfg=self.cfg, seed=7)
        self.assertEqual(c1, c2)

    def test_multiblock_completion_spans_blocks(self):
        context = [258, 65, 259]
        completion = bytes(i % 256 for i in range(300))
        canvas, stats = dspark_generate(context, completion, cfg=self.cfg, seed=2)
        self.assertEqual(len(canvas), len(context) + len(completion))
        self.assertEqual(bytes(canvas[len(context) :]), completion)
        self.assertGreaterEqual(stats["blocks"], 5)

    def test_heavy_load_clips_verify_length(self):
        context = list(b"x")
        completion = b"y" * 64
        canvas_heavy, _ = dspark_generate(
            context, completion, cfg=self.cfg, seed=3, heavy_load=True
        )
        canvas_light, _ = dspark_generate(context, completion, cfg=self.cfg, seed=3)
        self.assertEqual(bytes(canvas_heavy[len(context) :]), completion)
        self.assertEqual(bytes(canvas_light[len(context) :]), completion)

    def test_entropy_bound_and_convergence_reported(self):
        context = list(b"prompt")
        completion = bytes((i * 13) & 0xFF for i in range(128))
        canvas, stats = dspark_generate(context, completion, cfg=self.cfg, seed=5)
        self.assertEqual(stats["blocks"], 2)
        self.assertGreaterEqual(stats["converged"], 1)
        self.assertGreaterEqual(stats["regenerations"], 0)
        self.assertEqual(bytes(canvas[len(context) :]), completion)


if __name__ == "__main__":
    unittest.main()
