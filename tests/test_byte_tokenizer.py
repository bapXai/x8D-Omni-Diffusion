# coding=utf-8
"""Tests for the pure-Python byte-native tokenizer (vocab 264)."""

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
    VOCAB_SIZE,
    ByteTokenizer,
)


class ByteTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()

    def test_vocab_size_is_264(self):
        self.assertEqual(self.tok.vocab_size, 264)
        self.assertEqual(len(self.tok), 264)

    def test_special_token_ids(self):
        self.assertEqual(MASK_TOKEN_ID, 256)
        self.assertEqual(PAD_TOKEN_ID, 257)
        self.assertEqual(BOS_TOKEN_ID, 258)
        self.assertEqual(EOS_TOKEN_ID, 259)
        self.assertEqual(IMG_START_TOKEN_ID, 260)
        self.assertEqual(IMG_END_TOKEN_ID, 261)
        self.assertEqual(AUD_START_TOKEN_ID, 262)
        self.assertEqual(AUD_END_TOKEN_ID, 263)

    def test_encode_ascii(self):
        self.assertEqual(self.tok.encode(b"hello", add_special_tokens=False), [104, 101, 108, 108, 111])

    def test_encode_bytes_law_no_vocab_lookup(self):
        data = b"\x00\xff\x7f\x80\x01"
        ids = self.tok.bytes_to_ids(data)
        self.assertEqual(ids, [0, 255, 127, 128, 1])

    def test_roundtrip_all_256_bytes(self):
        all_bytes = bytes(range(256))
        ids = self.tok.encode(all_bytes, add_special_tokens=False)
        self.assertEqual(len(ids), 256)
        self.assertTrue(all(0 <= i < 256 for i in ids))
        self.assertEqual(self.tok.decode(ids), all_bytes)

    def test_roundtrip_utf8_text(self):
        text = "hello world 你好 🌍"
        ids = self.tok.encode_text(text, add_special_tokens=False)
        self.assertEqual(self.tok.decode_text(ids), text)

    def test_special_tokens_wrap(self):
        ids = self.tok.encode(b"abc")
        self.assertEqual(ids[0], BOS_TOKEN_ID)
        self.assertEqual(ids[-1], EOS_TOKEN_ID)
        self.assertEqual(ids[1:4], [97, 98, 99])

    def test_decode_skips_specials_by_default(self):
        ids = self.tok.encode(b"abc")
        self.assertEqual(self.tok.decode(ids), b"abc")

    def test_encode_image_wraps_in_img_start_end(self):
        ids = self.tok.encode_image(bytes([1, 2, 3]), add_special_tokens=False)
        self.assertEqual(ids, [IMG_START_TOKEN_ID, 1, 2, 3, IMG_END_TOKEN_ID])

    def test_encode_audio_wraps_in_aud_start_end(self):
        ids = self.tok.encode_audio(bytes([9, 8, 7]), add_special_tokens=False)
        self.assertEqual(ids, [AUD_START_TOKEN_ID, 9, 8, 7, AUD_END_TOKEN_ID])

    def test_call_returns_input_ids_and_mask(self):
        out = self.tok("hi")
        self.assertEqual(out["input_ids"], [BOS_TOKEN_ID, 104, 105, EOS_TOKEN_ID])
        self.assertEqual(out["attention_mask"], [1, 1, 1, 1])

    def test_padding_left_pads_with_257(self):
        batch = self.tok.id_batch([[1, 2, 3], [4, 5]])
        self.assertEqual(batch[0], [1, 2, 3])
        self.assertEqual(batch[1], [PAD_TOKEN_ID, 4, 5])

    def test_mask_canvas_and_renoise(self):
        canvas = self.tok.mask_canvas(8)
        self.assertEqual(canvas, [MASK_TOKEN_ID] * 8)
        renoised = self.tok.renoise_to_random_bytes([257, 258])
        self.assertTrue(all(0 <= b < 256 for b in renoised))

    def test_get_vocab_has_264_entries(self):
        vocab = self.tok.get_vocab()
        self.assertEqual(len(vocab), 264)
        self.assertEqual(vocab[256], "<mask>")

    def test_input_types(self):
        self.assertEqual(self.tok.encode("ab", add_special_tokens=False), [97, 98])
        self.assertEqual(self.tok.encode(bytearray(b"ab"), add_special_tokens=False), [97, 98])
        self.assertEqual(self.tok.encode([97, 98], add_special_tokens=False), [97, 98])

    def test_decode_non_utf8_bytes_as_bytes(self):
        raw = bytes([0xFF, 0xFE, 0xFD])
        ids = self.tok.encode(raw, add_special_tokens=False)
        self.assertEqual(self.tok.decode(ids), raw)

    def test_decode_rejects_out_of_vocab_ids(self):
        # ids >= 264 must never wrap into content bytes (regression #21)
        self.assertEqual(self.tok.decode([512, 513], as_bytes=True), b"")
        self.assertEqual(self.tok.decode([300, 400], as_bytes=True), b"")
        self.assertEqual(self.tok.decode([-1, 255], as_bytes=True), b"\xff")
        self.assertEqual(self.tok.decode([97, 263, 98], as_bytes=True), b"ab")


if __name__ == "__main__":
    unittest.main()
