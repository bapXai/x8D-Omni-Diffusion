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
        self.assertEqual(cfg.sliding_window, 1024)
        self.assertEqual(cfg.num_global_key_value_heads, 2)

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


if __name__ == "__main__":
    unittest.main()
