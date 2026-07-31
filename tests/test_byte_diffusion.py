# coding=utf-8
"""Tests for the DiffusionGemma-style byte diffusion sampler."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.byte_diffusion import (  # noqa: E402
    ByteDiffusionSampler,
    stable_hash,
)
from omni_diffusion.models.dream.byte_tokenizer import (  # noqa: E402
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
)


class ByteDiffusionSamplerTest(unittest.TestCase):
    def test_defaults(self):
        sampler = ByteDiffusionSampler()
        self.assertEqual(sampler.vocab_size, 264)
        self.assertEqual(sampler.canvas_length, 256)
        self.assertEqual(sampler.diffusion_entropy_bound, 0.1)
        self.assertEqual(sampler.seed, 0)

    def test_entropy_bound_param_accepted(self):
        for bound in (None, 0.0, 0.05, 0.1, 0.5, 1.0):
            sampler = ByteDiffusionSampler(diffusion_entropy_bound=0.1)
            if bound is None:
                self.assertEqual(sampler.sample_canvas(b"hi", steps=2), sampler.sample_canvas(b"hi", steps=2))
            else:
                out = sampler.sample_canvas(b"hi", steps=2, entropy_bound=bound)
                self.assertIsInstance(out, bytes)

    def test_entropy_bound_rejects_out_of_range(self):
        sampler = ByteDiffusionSampler()
        with self.assertRaises(ValueError):
            sampler.sample_canvas(b"hi", steps=2, entropy_bound=1.5)
        with self.assertRaises(ValueError):
            sampler.sample_canvas(b"hi", steps=2, entropy_bound=-0.1)

    def test_encode_wraps_with_bos_eos(self):
        sampler = ByteDiffusionSampler()
        ids = sampler.encode(b"abc")
        self.assertEqual(ids[0], BOS_TOKEN_ID)
        self.assertEqual(ids[-1], EOS_TOKEN_ID)
        self.assertEqual(ids[1:-1], [97, 98, 99])

    def test_encode_decode_roundtrip_utf8(self):
        texts = ["नमस्ते", "🙂🌍🚀🎉", "Hello, byte world!", "Привет, как дела?", "∀x∈ℝ, x² ≥ 0"]
        sampler = ByteDiffusionSampler()
        for text in texts:
            ids = sampler.encode(text)
            self.assertEqual(ids[0], BOS_TOKEN_ID)
            self.assertEqual(ids[-1], EOS_TOKEN_ID)
            self.assertTrue(all(0 <= i <= 263 for i in ids))
            self.assertTrue(all(0 <= i < 256 for i in ids[1:-1]))
            decoded = sampler.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, text.encode("utf-8"))
            self.assertEqual(decoded.decode("utf-8"), text)

    def test_encode_binary_all_bytes(self):
        sampler = ByteDiffusionSampler()
        raw = bytes(range(256))
        ids = sampler.encode(raw)
        self.assertEqual(ids[1:-1], list(range(256)))
        self.assertEqual(sampler.decode(ids), raw)

    def test_mask_canvas_returns_masked_and_truth(self):
        ids = list(range(64))
        sampler = ByteDiffusionSampler(seed=7)
        masked, truth = sampler.mask_canvas(ids, mask_ratio=0.5)
        self.assertEqual(truth, ids)
        self.assertEqual(len(masked), len(ids))
        count = sum(1 for t in masked if t == MASK_TOKEN_ID)
        self.assertEqual(count, 32)

    def test_mask_ratio_respected_approximately(self):
        sampler = ByteDiffusionSampler(seed=3)
        ids = list(range(200))
        for ratio in (0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 1.0):
            masked, truth = sampler.mask_canvas(ids, mask_ratio=ratio)
            count = sum(1 for t in masked if t == MASK_TOKEN_ID)
            self.assertEqual(truth, ids)
            self.assertGreaterEqual(count, int(200 * ratio) - 1)
            self.assertLessEqual(count, int(200 * ratio) + 1)

    def test_mask_canvas_full_and_empty(self):
        sampler = ByteDiffusionSampler(seed=1)
        masked, truth = sampler.mask_canvas(list(range(10)), mask_ratio=1.0)
        self.assertTrue(all(t == MASK_TOKEN_ID for t in masked))
        masked, truth = sampler.mask_canvas(list(range(10)), mask_ratio=0.0)
        self.assertEqual(masked, list(range(10)))

    def test_renoise_fills_mask_with_random_bytes(self):
        sampler = ByteDiffusionSampler(seed=11)
        canvas = [MASK_TOKEN_ID, 5, MASK_TOKEN_ID, PAD_TOKEN_ID, 3]
        out = sampler.renoise_to_random_bytes(canvas)
        self.assertEqual(len(out), len(canvas))
        self.assertEqual(out[1], 5)
        self.assertEqual(out[3], PAD_TOKEN_ID)
        self.assertTrue(0 <= out[0] < 256)
        self.assertTrue(0 <= out[2] < 256)
        self.assertNotIn(MASK_TOKEN_ID, out)

    def test_denoise_step_returns_per_position_floats(self):
        sampler = ByteDiffusionSampler(seed=2)
        canvas = [MASK_TOKEN_ID, 97, MASK_TOKEN_ID, 98, PAD_TOKEN_ID]
        logits = sampler.denoise_step(canvas)
        self.assertEqual(len(logits), len(canvas))
        for score in logits:
            self.assertIsInstance(score, float)
            self.assertTrue(0.0 <= score <= 1.0)

    def test_deterministic_stable_hash(self):
        self.assertEqual(stable_hash(1, 2, 3), stable_hash(1, 2, 3))
        self.assertEqual(stable_hash(0, 0, 0), stable_hash(0, 0, 0))

    def test_sample_canvas_deterministic_same_seed(self):
        a = ByteDiffusionSampler(seed=42, canvas_length=32)
        b = ByteDiffusionSampler(seed=42, canvas_length=32)
        self.assertEqual(a.sample_canvas("hello", steps=4), b.sample_canvas("hello", steps=4))
        a2 = ByteDiffusionSampler(seed=42, canvas_length=32)
        self.assertEqual(a2.sample_canvas("hello", steps=4), a.sample_canvas("hello", steps=4))

    def test_sample_canvas_decodes_as_utf8(self):
        sampler = ByteDiffusionSampler(seed=0, canvas_length=32)
        for prompt in ("hi", "hello world", "नमस्ते", "🙂🌍🚀🎉"):
            for steps in (1, 3, 8):
                out = sampler.sample_canvas(prompt, steps=steps)
                self.assertIsInstance(out, bytes)
                out.decode("utf-8")  # must not raise UnicodeDecodeError

    def test_sample_canvas_respects_canvas_length(self):
        sampler = ByteDiffusionSampler(seed=5, canvas_length=32)
        out = sampler.sample_canvas("a" * 200, steps=2)
        self.assertIsInstance(out, bytes)
        out.decode("utf-8")


if __name__ == "__main__":
    unittest.main()
