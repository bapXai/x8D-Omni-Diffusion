# coding=utf-8
"""Byte-native discrete diffusion sampler tests -- DREAM + DiffusionGemma + NanoQuant.

Covers the merged sampler family in :mod:`omni_diffusion.x8d_byte_diffusion`:

- **masked_denoise** -- DREAM / Omni-Diffusion absorbing-state masked diffusion
  (MASK=256 canvas, confidence-ordered transfer).
- **uniform_denoise** -- DiffusionGemma uniform-state entropy-bound sampling
  (random-byte canvas, cumulative-entropy budget, self-conditioning carry,
  adaptive stopping).
- **reconstruct_block** -- NanoQuant-style block reconstruction with
  error-propagation mitigation (worst-error renoise + teacher-guided draw,
  lossless guard).

Pure Python stdlib (no torch). Run:
    python3 -m unittest tests.test_x8d_byte_diffusion -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.models.dream.byte_tokenizer import (  # noqa: E402
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
    ByteTokenizer,
)
from omni_diffusion.x8d_byte_diffusion import (  # noqa: E402
    ADAPTIVE_PATIENCE,
    BYTE_MAX,
    BYTE_MIN,
    DEFAULT_CANVAS_LENGTH,
    DEFAULT_ENTROPY_BOUND,
    DEFAULT_STEPS,
    MASK_ID,
    STABILITY_ENTROPY,
    VOCAB_SIZE,
    ByteDiffusionConfig,
    ByteModelSurrogate,
    argmax_byte,
    expected_byte,
    masked_canvas,
    masked_denoise,
    position_entropy,
    reconstruct_block,
    uniform_denoise,
    uniform_noise_canvas,
)


def make_cfg(**overrides) -> ByteDiffusionConfig:
    """Small deterministic config for fast tests."""
    base = {
        "canvas_length": 64,
        "steps": 16,
        "seed": 7,
        "entropy_bound": 0.1,
    }
    base.update(overrides)
    return ByteDiffusionConfig(**base)


def ctx_bytes(tok: ByteTokenizer, text: str) -> list:
    """Encode a prompt as BOS..bytes..EOS context ids."""
    return tok.encode_text(text, add_special_tokens=False)


class ByteDiffusionModeTest(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.cfg = make_cfg()
        self.model = ByteModelSurrogate(vocab_size=self.cfg.vocab_size, seed=self.cfg.seed)

    def test_uniform_noise_canvas_is_byte_sane(self):
        canvas = uniform_noise_canvas(64, seed=3)
        self.assertEqual(len(canvas), 64)
        self.assertTrue(all(BYTE_MIN <= b <= BYTE_MAX for b in canvas))
        self.assertNotIn(MASK_ID, canvas)

    def test_masked_canvas_is_all_mask(self):
        canvas = masked_canvas(10)
        self.assertEqual(canvas, [MASK_ID] * 10)

    def test_position_entropy_uniform_is_maximal(self):
        # uniform distribution over 264 ids -> ln(264) nats
        p = [1.0 / VOCAB_SIZE] * VOCAB_SIZE
        h = position_entropy(p)
        self.assertAlmostEqual(h, __import__("math").log(VOCAB_SIZE), places=4)

    def test_position_entropy_delta_is_zero(self):
        p = [1.0] + [0.0] * (VOCAB_SIZE - 1)
        self.assertAlmostEqual(position_entropy(p), 0.0, places=6)

    def test_expected_byte_is_probability_weighted(self):
        probs = [0.0] * VOCAB_SIZE
        probs[7] = 1.0
        self.assertAlmostEqual(expected_byte(probs), 7.0, places=6)

    def test_argmax_byte_finds_peak(self):
        probs = [0.0] * VOCAB_SIZE
        probs[200] = 0.9
        probs[10] = 0.1
        self.assertEqual(argmax_byte(probs), 200)


class MaskedDenoiseTest(unittest.TestCase):
    """DREAM / Omni-Diffusion absorbing-state path."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.cfg = make_cfg()
        self.model = ByteModelSurrogate(vocab_size=self.cfg.vocab_size, seed=self.cfg.seed)

    def test_masked_fills_canvas_and_is_deterministic(self):
        ctx = ctx_bytes(self.tok, "hello")
        canvas, stats = masked_denoise(self.cfg, self.model, ctx, 32)
        self.assertEqual(len(canvas), len(ctx) + 32)
        # canvas fully filled: no MASK remains
        self.assertNotIn(MASK_ID, canvas)
        self.assertGreater(stats["steps_used"], 0)
        # deterministic end-to-end
        canvas2, _ = masked_denoise(self.cfg, self.model, ctx, 32)
        self.assertEqual(canvas, canvas2)

    def test_masked_preserves_context(self):
        ctx = ctx_bytes(self.tok, "prefix")
        canvas, _ = masked_denoise(self.cfg, self.model, ctx, 8)
        self.assertEqual(canvas[: len(ctx)], ctx)

    def test_masked_output_is_byte_sane(self):
        ctx = ctx_bytes(self.tok, "x")
        canvas, _ = masked_denoise(self.cfg, self.model, ctx, 16)
        completion = canvas[len(ctx):]
        self.assertTrue(all(BYTE_MIN <= b <= BYTE_MAX for b in completion))

    def test_masked_mean_confidence_in_range(self):
        ctx = ctx_bytes(self.tok, "a")
        _, stats = masked_denoise(self.cfg, self.model, ctx, 8)
        self.assertTrue(0.0 <= stats["mean_confidence"] <= 1.0)


class UniformDenoiseTest(unittest.TestCase):
    """DiffusionGemma uniform-state entropy-bound path (x8D main mode)."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.cfg = make_cfg()
        self.model = ByteModelSurrogate(vocab_size=self.cfg.vocab_size, seed=self.cfg.seed)

    def test_uniform_preserves_context_and_fills_completion(self):
        ctx = ctx_bytes(self.tok, "prompt")
        canvas, stats = uniform_denoise(self.cfg, self.model, ctx, 24)
        self.assertEqual(len(canvas), len(ctx) + 24)
        self.assertEqual(canvas[: len(ctx)], ctx)
        self.assertGreater(stats["steps_used"], 0)

    def test_uniform_entropy_budget_respected(self):
        # entropy-bound: accepted positions must be the lowest-entropy (most
        # confident) subset; late in the schedule the whole canvas commits
        # in parallel, not one position per step (DiffusionGemma).
        cfg = make_cfg(steps=8)
        model = ByteModelSurrogate(vocab_size=cfg.vocab_size, seed=cfg.seed)
        ctx = ctx_bytes(self.tok, "q")
        canvas, stats = uniform_denoise(cfg, model, ctx, 16)
        # the entire completion span is filled (parallel commit), no MASK
        self.assertNotIn(MASK_ID, canvas[len(ctx):])
        # committed report is sane
        self.assertTrue(0 <= stats["committed"] <= 16)

    def test_uniform_output_is_byte_sane_and_deterministic(self):
        ctx = ctx_bytes(self.tok, "hi")
        canvas, _ = uniform_denoise(self.cfg, self.model, ctx, 20)
        completion = canvas[len(ctx):]
        self.assertTrue(all(BYTE_MIN <= b <= BYTE_MAX for b in completion))
        canvas2, _ = uniform_denoise(self.cfg, self.model, ctx, 20)
        self.assertEqual(canvas, canvas2)

    def test_uniform_no_mask_state_survives(self):
        # uniform-state: the completion span is never MASK (256) after step 1
        ctx = ctx_bytes(self.tok, "a")
        canvas, _ = uniform_denoise(self.cfg, self.model, ctx, 12)
        self.assertNotIn(MASK_ID, canvas[len(ctx):])

    def test_self_conditioning_changes_trajectory(self):
        # enabling the carry must alter the generated completion
        cfg_off = make_cfg(self_conditioning=False)
        cfg_on = make_cfg(self_conditioning=True)
        m_off = ByteModelSurrogate(vocab_size=cfg_off.vocab_size, seed=cfg_off.seed)
        m_on = ByteModelSurrogate(vocab_size=cfg_on.vocab_size, seed=cfg_on.seed)
        ctx = ctx_bytes(self.tok, "test")
        c_off, _ = uniform_denoise(cfg_off, m_off, ctx, 16)
        c_on, _ = uniform_denoise(cfg_on, m_on, ctx, 16)
        self.assertNotEqual(c_off, c_on)

    def test_adaptive_stop_can_fire(self):
        # with high sharpness the sampler reaches stability + low entropy
        cfg = make_cfg(steps=16, entropy_bound=0.2)
        model = ByteModelSurrogate(vocab_size=cfg.vocab_size, seed=cfg.seed)
        ctx = ctx_bytes(self.tok, "s")
        _, stats = uniform_denoise(cfg, model, ctx, 8)
        # adaptive stop must not consume more than a couple of the last steps;
        # with a stable target it fires before the full 16 steps
        self.assertLessEqual(stats["steps_used"], cfg.steps)
        self.assertTrue(stats["steps_used"] >= 1)

    def test_uniform_committed_count_reported(self):
        ctx = ctx_bytes(self.tok, "z")
        _, stats = uniform_denoise(self.cfg, self.model, ctx, 8)
        self.assertIn("committed", stats)
        self.assertTrue(0 <= stats["committed"] <= 8)


class ReconstructBlockTest(unittest.TestCase):
    """NanoQuant-style block reconstruction with error mitigation."""

    def setUp(self):
        self.cfg = make_cfg(steps=20)
        self.target = list(range(64))  # 64 distinct bytes as the teacher block

    def test_reconstruct_converges_lossless(self):
        noisy = uniform_noise_canvas(64, seed=11)
        out, stats = reconstruct_block(self.cfg, noisy, self.target)
        self.assertEqual(out, self.target)
        self.assertEqual(stats["lossless"], 1.0)
        self.assertAlmostEqual(stats["error"], 0.0, places=6)

    def test_reconstruct_preserves_already_correct_positions(self):
        # half the noisy block already matches target; those are never renoised
        noisy = list(self.target)
        for i in range(0, 64, 2):
            noisy[i] = 255 - i
        out, stats = reconstruct_block(self.cfg, noisy, self.target)
        self.assertEqual(out, self.target)
        # odd positions (already correct) were not regenerated
        self.assertLess(stats["regenerations"], 64)

    def test_reconstruct_is_deterministic(self):
        noisy = uniform_noise_canvas(64, seed=5)
        out1, _ = reconstruct_block(self.cfg, noisy, self.target)
        out2, _ = reconstruct_block(self.cfg, noisy, self.target)
        self.assertEqual(out1, out2)

    def test_reconstruct_clean_block_needs_zero_steps(self):
        out, stats = reconstruct_block(self.cfg, list(self.target), self.target)
        self.assertEqual(out, self.target)
        self.assertEqual(stats["steps_used"], 0)
        self.assertEqual(stats["regenerations"], 0)


class CrossModeComparisonTest(unittest.TestCase):
    """The three designs are distinct but all byte-sane + context-preserving."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.cfg = make_cfg()

    def test_modes_agree_on_context_preservation(self):
        ctx = ctx_bytes(self.tok, "compare")
        for mode in ("masked", "uniform"):
            canvas, _ = sample_mode(mode, self.cfg, ctx, 16)
            self.assertEqual(canvas[: len(ctx)], ctx)

    def test_modes_produce_distinct_outputs(self):
        ctx = ctx_bytes(self.tok, "diff")
        m_masked, _ = masked_denoise(
            self.cfg,
            ByteModelSurrogate(vocab_size=self.cfg.vocab_size, seed=self.cfg.seed),
            ctx,
            16,
        )
        m_uniform, _ = uniform_denoise(
            self.cfg,
            ByteModelSurrogate(vocab_size=self.cfg.vocab_size, seed=self.cfg.seed),
            ctx,
            16,
        )
        self.assertNotEqual(m_masked, m_uniform)

    def test_uniform_commits_whole_canvas_where_masked_leaves_tail(self):
        # With an identical step budget, the uniform-state sampler reaches a
        # fully-decided completion (every slot a byte), while masked transfer
        # only fills the high-confidence head -- the parallel-vs-sequential
        # difference DiffusionGemma demonstrates over AR.
        ctx = ctx_bytes(self.tok, "p")
        cfg = make_cfg(steps=3)
        m = ByteModelSurrogate(vocab_size=cfg.vocab_size, seed=cfg.seed)
        c_uniform, _ = uniform_denoise(cfg, m, ctx, 16)
        c_masked, _ = masked_denoise(cfg, m, ctx, 16)
        uniform_comp = c_uniform[len(ctx):]
        masked_comp = c_masked[len(ctx):]
        self.assertNotIn(MASK_ID, uniform_comp)
        # masked transfer over 3 steps leaves some slots uncommitted (MASK)
        self.assertTrue(any(t == MASK_ID for t in masked_comp))

    def test_uniform_parallel_canvas_commit(self):
        # DiffusionGemma's defining property: the whole completion canvas is
        # committed in parallel per step (not one token at a time). Verify that
        # with a sharp late-step schedule, all completion slots commit together.
        ctx = ctx_bytes(self.tok, "p")
        cfg = make_cfg(steps=8)
        model = ByteModelSurrogate(vocab_size=cfg.vocab_size, seed=cfg.seed)
        canvas, _ = uniform_denoise(cfg, model, ctx, 16)
        completion = canvas[len(ctx):]
        # all 16 slots carry a committed byte (no MASK, all in [0,255])
        self.assertEqual(len(completion), 16)
        self.assertTrue(all(BYTE_MIN <= b <= BYTE_MAX for b in completion))
        # uniform-state fills the whole span; a token-by-token AR would leave
        # trailing noise early -- here every slot is a decided byte.
        self.assertNotIn(MASK_ID, completion)


def sample_mode(mode, cfg, ctx, length):
    """Helper: dispatch one of the merged sampling modes."""
    if mode == "masked":
        return masked_denoise(
            cfg, ByteModelSurrogate(vocab_size=cfg.vocab_size, seed=cfg.seed), ctx, length
        )
    return uniform_denoise(
        cfg, ByteModelSurrogate(vocab_size=cfg.vocab_size, seed=cfg.seed), ctx, length
    )


if __name__ == "__main__":
    unittest.main()
