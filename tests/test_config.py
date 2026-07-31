# coding=utf-8
"""Tests for the byte-native DreamConfig defaults (vocab 264)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from omni_diffusion.models.dream.configuration_dream import DreamConfig

    HAS_TRANSFORMERS = True
except ImportError:  # transformers not installed (byte core stays pure-Python)
    HAS_TRANSFORMERS = False

try:
    import torch  # noqa: F401

    HAS_TORCH = HAS_TRANSFORMERS
except ImportError:
    HAS_TORCH = False

_RESUME_JSON = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "omni_diffusion",
        "models",
        "dream",
        "config_dream_resume.json",
    )
)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers not installed; byte core runs without it")
class DreamConfigClassTest(unittest.TestCase):
    def test_defaults_are_byte_native(self):
        cfg = DreamConfig()
        self.assertEqual(cfg.vocab_size, 264)
        self.assertEqual(cfg.mask_token_id, 256)
        self.assertEqual(cfg.pad_token_id, 257)
        self.assertEqual(cfg.bos_token_id, 258)
        self.assertEqual(cfg.eos_token_id, 259)
        self.assertEqual(cfg.img_start_token_id, 260)
        self.assertEqual(cfg.img_end_token_id, 261)
        self.assertEqual(cfg.aud_start_token_id, 262)
        self.assertEqual(cfg.aud_end_token_id, 263)
        self.assertTrue(cfg.tie_word_embeddings)

    def test_diffusion_gemma_fields_present(self):
        cfg = DreamConfig()
        self.assertEqual(cfg.canvas_length, 256)
        self.assertEqual(cfg.max_denoising_steps, 48)
        self.assertEqual(cfg.diffusion_sampler, "entropy_bound")
        self.assertEqual(cfg.diffusion_entropy_bound, 0.1)
        self.assertTrue(cfg.self_conditioning)
        self.assertEqual(cfg.final_logit_softcap, None)
        self.assertEqual(cfg.sliding_window, None)  # None when use_sliding_window=False
        self.assertFalse(cfg.use_sliding_window)

    def test_resume_json_round_trips(self):
        with open(_RESUME_JSON) as f:
            data = json.load(f)
        cfg = DreamConfig(**data)
        self.assertEqual(cfg.vocab_size, 264)


class DreamConfigResumeJsonTest(unittest.TestCase):
    def test_resume_json_is_byte_native(self):
        with open(_RESUME_JSON) as f:
            data = json.load(f)
        self.assertEqual(data["vocab_size"], 264)
        self.assertEqual(data["mask_token_id"], 256)
        self.assertEqual(data["pad_token_id"], 257)
        self.assertEqual(data["bos_token_id"], 258)
        self.assertEqual(data["eos_token_id"], 259)
        self.assertEqual(data["img_start_token_id"], 260)
        self.assertEqual(data["img_end_token_id"], 261)
        self.assertEqual(data["aud_start_token_id"], 262)
        self.assertEqual(data["aud_end_token_id"], 263)
        self.assertEqual(data["tie_word_embeddings"], True)

    def test_no_bpe_token_ids_anywhere(self):
        with open(_RESUME_JSON) as f:
            blob = f.read()
        for banned in ("151643", "151666", "176264", "151936"):
            self.assertNotIn(banned, blob)


@unittest.skipUnless(HAS_TORCH, "torch not installed; byte core runs without it")
class DreamTrainableSmokeTest(unittest.TestCase):
    """End-to-end: init DreamModel, one train step, save + reload (issue #34).

    Uses a tiny config (2 layers / hidden 128) so it runs on CPU quickly.
    Exercises the transformers >=5 fixes: rope 'default'->inline, dict
    _tied_weights_keys, lazy deepspeed/funasr import, use_audio gate, and
    auto-built position_ids/attention_mask.
    """

    TINY = dict(
        vocab_size=264,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )

    def test_forward_and_train_step(self):
        from omni_diffusion.models.dream.byte_tokenizer import ByteTokenizer
        from omni_diffusion.models.dream.modeling_dream import DreamModel

        model = DreamModel(DreamConfig(**self.TINY))
        self.assertEqual(model.get_input_embeddings().weight.shape[0], 264)
        tok = ByteTokenizer()
        ids = tok.encode("byte diffusion over the 264 vocab", add_special_tokens=True)
        t = torch.tensor([ids])
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad()
        out = model(input_ids=t, labels=t)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        opt.step()
        self.assertLess(float(out.loss), float(model.config.vocab_size))

    def test_checkpoint_save_and_reload(self):
        import tempfile

        from omni_diffusion.models.dream.modeling_dream import DreamModel

        model = DreamModel(DreamConfig(**self.TINY))
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            m2 = DreamModel.from_pretrained(d)
        self.assertEqual(m2.config.vocab_size, 264)
        self.assertEqual(
            model.get_input_embeddings().weight.shape,
            m2.get_input_embeddings().weight.shape,
        )

    def test_audio_gated_off_by_default(self):
        from omni_diffusion.models.dream.modeling_dream import DreamModel

        model = DreamModel(DreamConfig(**self.TINY))
        self.assertIsNone(model.model.audio_model)


if __name__ == "__main__":
    unittest.main()
