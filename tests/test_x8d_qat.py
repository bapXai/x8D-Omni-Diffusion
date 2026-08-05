# coding=utf-8
"""Tests for the x8Dsub-byte QAT (Quantization-Aware Training) scaffold.

Covers the pure-stdlib core of ``omni_diffusion/x8d_qat.py`` plus the
end-to-end QAT fine-tuning loop in ``tools/finetune_qat.py``: fake-quant with
STE, roundtrip loss, the 264-vocab byte-diffusion loss, weight wrapping, config
defaults and offline training. Pure Python stdlib unittest — no torch required
(a single torch-gated test is skipped when torch is absent).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_qat import (  # noqa: E402
    DEFAULT_QUANT_CLAMP,
    QATConfig,
    U8_MAX,
    U8_MIN,
    VOCAB_SIZE,
    byte_diffusion_loss,
    hard_quantize,
    mask_canvas,
    qat_config,
    quantize_ste,
    renoise_to_random_bytes,
    ste_grad,
    wrap_for_qat,
    x8d_qat_roundtrip_loss,
)
from omni_diffusion.x8d_quanta import save_gguf  # noqa: E402
from tools.finetune_qat import (  # noqa: E402
    fine_tune_qat,
    load_quanta_weights,
    split_canvas_blocks,
    synth_dataset,
    synth_weights,
)

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class QuantizeSTETest(unittest.TestCase):
    def test_forward_rounds_like_round(self):
        values = [0.0, 0.4, 0.5, 0.6, 1.0, 127.0, 127.4, 254.5, 255.0]
        for x in values:
            self.assertEqual(quantize_ste(x), float(round(x)))

    def test_forward_clamps_to_u8_bounds(self):
        self.assertEqual(quantize_ste(-10.0), 0.0)
        self.assertEqual(quantize_ste(-0.4), 0.0)
        self.assertEqual(quantize_ste(255.0), 255.0)
        self.assertEqual(quantize_ste(256.7), 255.0)
        self.assertEqual(quantize_ste(300.0), 255.0)

    def test_byte_aligned_weights_are_identity(self):
        aligned = [0.0, 1.0, 127.0, 254.0, 255.0]
        self.assertEqual(quantize_ste(aligned), aligned)

    def test_recurses_over_containers(self):
        weights = [[0.2, 127.5], [300.0, -1.0]]
        self.assertEqual(quantize_ste(weights), [[0.0, 128.0], [255.0, 0.0]])

    def test_ste_gradient_is_identity(self):
        for x in (-3.0, 0.0, 0.5, 1.2, 127.0, 254.7, 300.0):
            self.assertEqual(ste_grad(x), 1.0)

    def test_ste_backward_passes_gradient_through(self):
        # y = quantize_ste(w); loss = 0.5*(y - t)^2.
        # round() has zero gradient almost everywhere, so QAT's STE must supply
        # the identity backward: dLoss/dw = (y - t) * ste_grad(w) = (y - t).
        w = 127.4
        t = 200.0
        y = quantize_ste(w)
        loss = 0.5 * (y - t) ** 2
        analytic_grad = (y - t) * ste_grad(w)
        self.assertEqual(y, 127.0)
        self.assertAlmostEqual(analytic_grad, y - t, places=6)
        self.assertAlmostEqual(analytic_grad, -73.0, places=6)
        self.assertNotEqual(analytic_grad, 0.0)
        self.assertTrue(loss > 0.0)

    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_quantize_ste_torch_variant(self):
        # STE pattern: x.round().clamp(lo, hi) + (x - x.detach()).
        w = torch.tensor([0.2, 127.5, 300.0, -5.0], requires_grad=True)
        q = quantize_ste(w)
        self.assertTrue(torch.allclose(q, torch.tensor([0.0, 128.0, 255.0, 0.0], dtype=torch.float32)))
        target = torch.tensor(200.0)
        loss = 0.5 * (q[0] - target) ** 2
        loss.backward()
        self.assertAlmostEqual(float(w.grad[0]), float(q[0].detach()) - 200.0, places=5)


class RoundtripLossTest(unittest.TestCase):
    def test_zero_for_byte_aligned_weights(self):
        aligned = {"w1": [0.0, 1.0, 127.0, 255.0], "w2": [3.0, 200.0]}
        self.assertEqual(x8d_qat_roundtrip_loss(aligned, aligned), 0.0)
        self.assertEqual(x8d_qat_roundtrip_loss(aligned, quantize_ste(aligned)), 0.0)

    def test_positive_for_non_aligned_weights(self):
        raw = {"w": [0.2, 1.9, 300.0, -4.0]}
        quantized = quantize_ste(raw)
        loss = x8d_qat_roundtrip_loss(raw, quantized)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, x8d_qat_roundtrip_loss(raw, {"w": [0.0, 0.0, 0.0, 0.0]}))

    def test_decreases_when_weights_already_quantized(self):
        raw = {"w": [0.2, 1.9, 300.0, -4.0]}
        far = {"w": [255.0] * 4}
        self.assertLess(x8d_qat_roundtrip_loss(raw, quantize_ste(raw)), x8d_qat_roundtrip_loss(raw, far))

    def test_rejects_mismatched_keys(self):
        with self.assertRaises(ValueError):
            x8d_qat_roundtrip_loss({"a": [1.0]}, {"b": [1.0]})


class ByteDiffusionLossTest(unittest.TestCase):
    def test_zero_for_perfect_logits(self):
        target = 65  # b"A"
        one_hot = [[-100.0] * VOCAB_SIZE]
        one_hot[0][target] = 100.0
        self.assertAlmostEqual(byte_diffusion_loss(one_hot, [target]), 0.0, places=8)

    def test_positive_for_imperfect_logits(self):
        target = 65
        wrong = target ^ 0xFF
        logits = [[0.0] * VOCAB_SIZE]
        logits[0][wrong] = 100.0
        self.assertGreater(byte_diffusion_loss(logits, [target]), 0.0)

    def test_lower_for_better_calibrated_logits(self):
        target = 65
        perfect = [[-100.0] * VOCAB_SIZE]
        perfect[0][target] = 100.0
        moderate = [[0.0] * VOCAB_SIZE]
        moderate[0][target] = 5.0
        bad = [[0.0] * VOCAB_SIZE]
        bad[0][(target + 1) % VOCAB_SIZE] = 5.0
        l_perfect = byte_diffusion_loss(perfect, [target])
        l_moderate = byte_diffusion_loss(moderate, [target])
        l_bad = byte_diffusion_loss(bad, [target])
        self.assertLess(l_perfect, l_moderate)
        self.assertLess(l_moderate, l_bad)

    def test_multi_position_batch(self):
        logits = [[-100.0] * VOCAB_SIZE, [-100.0] * VOCAB_SIZE]
        logits[0][7] = 100.0
        logits[1][9] = 100.0
        self.assertAlmostEqual(byte_diffusion_loss(logits, [7, 9]), 0.0, places=8)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            byte_diffusion_loss([[0.0] * VOCAB_SIZE], [1, 2])
        with self.assertRaises(ValueError):
            byte_diffusion_loss([[0.0] * 8], [1])


class WrapForQATTest(unittest.TestCase):
    def test_dict_weights_are_quantized(self):
        weights = {"a": [0.2, 127.5, 300.0], "b": [-1.0, 255.0]}
        wrapped = wrap_for_qat(weights)
        self.assertEqual(wrapped.qat_weights()["a"], [0.0, 128.0, 255.0])
        self.assertEqual(wrapped.qat_weights()["b"], [0.0, 255.0])
        self.assertEqual(wrapped.raw_weights, weights)
        self.assertEqual(dict(wrapped.named_parameters()), {"a": [0.0, 128.0, 255.0], "b": [0.0, 255.0]})
        self.assertEqual(wrapped.parameters(), [[0.0, 128.0, 255.0], [0.0, 255.0]])

    def test_module_like_forward_gets_quantized_weights(self):
        class FakeModule:
            def __init__(self):
                self._w = {"w": [1.2, 2.9]}

            def named_parameters(self):
                return iter(self._w.items())

            def forward(self, qweights, *args, **kwargs):
                return qweights

        wrapped = wrap_for_qat(FakeModule())
        self.assertEqual(wrapped.forward(), {"w": [1.0, 3.0]})

    def test_ste_false_hard_quantizes(self):
        weights = {"w": [1.2, 2.9]}
        wrapped = wrap_for_qat(weights, ste=False)
        self.assertEqual(wrapped.qat_weights()["w"], [1.0, 3.0])

    def test_rejects_invalid_weights(self):
        with self.assertRaises(TypeError):
            wrap_for_qat(42)


class QATConfigTest(unittest.TestCase):
    def test_defaults_match_byte_diffusion_settings(self):
        cfg = QATConfig()
        self.assertEqual(cfg.lr, 1e-4)
        self.assertEqual(cfg.epochs, 1)
        self.assertEqual(cfg.batch_size, 8)
        self.assertTrue(cfg.ste)
        self.assertEqual(cfg.quant_clamp, (0, 255))
        self.assertEqual(cfg.diffusion_steps, 48)
        self.assertEqual(cfg.entropy_bound, 0.1)
        self.assertEqual(cfg.canvas_length, 256)
        self.assertEqual(DEFAULT_QUANT_CLAMP, (U8_MIN, U8_MAX))

    def test_lowercase_alias(self):
        self.assertIs(qat_config, QATConfig)
        self.assertEqual(qat_config(canvas_length=64).canvas_length, 64)


class MaskRenoiseReuseTest(unittest.TestCase):
    def test_mask_canvas_masks_fraction(self):
        ids = list(range(64))
        masked, truth = mask_canvas(ids, mask_ratio=0.5, seed=7)
        self.assertEqual(truth, ids)
        self.assertEqual(sum(1 for t in masked if t == 256), 32)

    def test_renoise_fills_mask_with_bytes(self):
        out = renoise_to_random_bytes([256, 5, 256, 257], seed=11)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[1], 5)
        self.assertEqual(out[3], 257)
        self.assertTrue(0 <= out[0] < 256)
        self.assertTrue(0 <= out[2] < 256)


class FineTuneQATEndToEndTest(unittest.TestCase):
    def test_fine_tune_qat_returns_loss_curve_and_byte_aligned_weights(self):
        cfg = QATConfig(epochs=2, canvas_length=32, batch_size=4)
        weights = synth_weights(64, seed=3)
        data = synth_dataset(128, seed=4)
        curve, final = fine_tune_qat(weights, data, cfg, seed=5)
        self.assertEqual(len(curve), 2)
        for loss in curve:
            self.assertGreater(loss, 0.0)
        self.assertEqual(sorted(final), sorted(weights))
        for values in final.values():
            self.assertTrue(all(0.0 <= v <= 255.0 for v in values))
            self.assertTrue(all(v == float(int(v)) for v in values))

    def test_fine_tune_qat_deterministic(self):
        cfg = QATConfig(epochs=1, canvas_length=32)
        weights = synth_weights(32, seed=1)
        data = synth_dataset(64, seed=2)
        curve_a, final_a = fine_tune_qat(weights, data, cfg, seed=9)
        curve_b, final_b = fine_tune_qat(weights, data, cfg, seed=9)
        self.assertEqual(curve_a, curve_b)
        self.assertEqual(final_a, final_b)

    def test_split_canvas_blocks_pads_tail(self):
        blocks = split_canvas_blocks(bytes(range(70)), 32)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(len(blocks[0]), 32)
        self.assertEqual(len(blocks[2]), 32)
        self.assertEqual(blocks[2][6:], [0] * 26)
        with self.assertRaises(ValueError):
            split_canvas_blocks(b"x", 0)

    def test_load_quanta_weights_maps_bytes_to_floats(self):
        fd, path = tempfile.mkstemp(suffix=".x8D")
        os.close(fd)
        try:
            save_gguf({"data": bytes(range(256))}, path)
            weights = load_quanta_weights(path)
            self.assertEqual(list(weights), ["data"])
            self.assertEqual(weights["data"], [float(b) for b in range(256)])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
