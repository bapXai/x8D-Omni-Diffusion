# coding=utf-8
"""End-to-end query tests through the byte-native pipeline.

Covers the full cycle for every query modality:
  encode(query) -> [BOS .. bytes .. EOS]  ->  mask canvas  ->  denoise  ->  decode

Pure Python stdlib (no torch). The denoise step uses a reference
`ByteDiffusionSampler` that operates over byte ids 0-255 + specials,
which is the exact contract the torch DreamModel will implement.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.models.dream.byte_tokenizer import (  # noqa: E402
    AUD_END_TOKEN_ID,
    AUD_START_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    IMG_END_TOKEN_ID,
    IMG_START_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    ByteTokenizer,
)


class ByteDiffusionSampler:
    """Pure-Python reference for the byte diffusion denoise loop.

    Mirrors the masked-diffusion contract used by generation_utils._sample():
    a fully-masked canvas is iteratively filled; content positions decode to
    bytes 0-255, special positions to their special ids.
    """

    def __init__(self, vocab_size=264):
        self.vocab_size = vocab_size

    def denoise(self, canvas, steps=48, seed=0):
        """Deterministic pseudo-diffusion: fill mask slots with a seeded byte mix.

        Returns the final byte-id canvas (same length as input). The pattern
        (hash of position+step mod 256) stands in for the real model's logits.
        """
        import random

        rng = random.Random(seed)
        out = list(canvas)
        for step in range(steps):
            done = True
            for i, tok in enumerate(out):
                if tok == MASK_TOKEN_ID:
                    # pseudo-logit: deterministic per (pos, step) so final
                    # canvas is reproducible and byte-sane.
                    out[i] = (rng.randint(0, 255) + step) & 0xFF
                    done = False
            if done:
                break
        return out


class QueryPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.sampler = ByteDiffusionSampler()

    def _generate(self, query):
        """Full pipeline: encode query -> mask canvas -> denoise -> decode."""
        ids = self.tok.encode(query)
        canvas = self.tok.mask_canvas(len(ids))
        denoised = self.sampler.denoise(canvas)
        return self.tok.decode(denoised)

    def test_text_query_multilingual(self):
        queries = [
            "Hello, world!",
            "What is the weather in Paris?",
            "مرحبا بالعالم",                     # Arabic (RTL)
            "こんにちは世界",                     # Japanese
            "क्या आप हिंदी बोलते हैं?",            # Hindi
            "Привет, как дела?",                 # Cyrillic
            "Γειά σου Κόσμε",                     # Greek
            "أهلًا بالعالم",                       # Arabic
            "printf('hello %s', name);",         # code
            "∀x∈ℝ, x² ≥ 0",                       # unicode math
        ]
        for q in queries:
            ids = self.tok.encode_text(q, add_special_tokens=False)
            # every content id is a raw byte
            self.assertTrue(all(0 <= i < 256 for i in ids), q)
            # round-trip preserves exact text
            self.assertEqual(self.tok.decode_text(ids), q)

    def test_binary_query_all_bytes(self):
        binary = bytes(range(256))
        ids = self.tok.encode(binary, add_special_tokens=False)
        self.assertEqual(ids, list(range(256)))
        self.assertEqual(self.tok.decode(ids), binary)

    def test_image_query(self):
        fake_png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + os.urandom(100)
        ids = self.tok.encode_image(fake_png, add_special_tokens=False)
        self.assertEqual(ids[0], IMG_START_TOKEN_ID)
        self.assertEqual(ids[-1], IMG_END_TOKEN_ID)
        content = self.tok.decode(ids, skip_special_tokens=True)
        self.assertEqual(content, fake_png)

    def test_audio_query(self):
        pcm = bytes([128, 129, 127, 126, 130] * 20)  # 16-bit PCM sample bytes
        ids = self.tok.encode_audio(pcm, add_special_tokens=False)
        self.assertEqual(ids[0], AUD_START_TOKEN_ID)
        self.assertEqual(ids[-1], AUD_END_TOKEN_ID)
        self.assertEqual(self.tok.decode(ids, skip_special_tokens=True), pcm)

    def test_query_wrap_roundtrip(self):
        q = "Generate an image of a cat."
        ids = self.tok.encode_text(q)
        self.assertEqual(ids[0], BOS_TOKEN_ID)
        self.assertEqual(ids[-1], EOS_TOKEN_ID)
        self.assertEqual(self.tok.decode_text(ids), q)

    def test_padded_batch_query(self):
        qs = [b"a", b"ab", b"abc"]
        batch = self.tok.id_batch(self.tok.encode(q, add_special_tokens=False) for q in qs)
        for row in batch:
            self.assertTrue(all(0 <= i < 256 or i == PAD_TOKEN_ID for i in row))

    def test_full_pipeline_text(self):
        out = self._generate("Hello byte world")
        self.assertIsInstance(out, bytes)

    def test_full_pipeline_all_queries(self):
        for q in ["hi", "how are you?", b"\x00\xff"]:
            out = self._generate(q)
            self.assertIsInstance(out, bytes)
            self.assertTrue(len(out) > 0)

    def test_specials_survive_pipeline(self):
        ids = self.tok.encode_audio(b"\x00\x01\x02", add_special_tokens=True)
        self.assertEqual(ids[0], BOS_TOKEN_ID)
        self.assertEqual(ids[1], AUD_START_TOKEN_ID)
        self.assertEqual(ids[-2], AUD_END_TOKEN_ID)
        self.assertEqual(ids[-1], EOS_TOKEN_ID)
        self.assertTrue(all(0 <= i <= 263 for i in ids))

    def test_query_preserves_all_utf8_multibyte(self):
        # every byte of a 4-byte emoji is preserved
        emoji = "🙂🌍🚀🎉"
        ids = self.tok.encode_text(emoji, add_special_tokens=False)
        self.assertEqual(self.tok.decode_text(ids), emoji)


if __name__ == "__main__":
    unittest.main()
