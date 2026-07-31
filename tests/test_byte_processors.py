# coding=utf-8
"""Tests for byte-native image/audio processors (issue #42).

Pure stdlib unittest. Verifies the processors strip MagViT/GLM-4-Voice
legacy bloat and produce flag-bounded byte canvases at the byte-law ids
(IMG_START=260/IMG_END=261, AUD_START=262/AUD_END=263), plus the zero-copy
JSONL -> U8 x8D GGUF import pipeline.
"""

import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from omni_diffusion.data.processor.image_processor import (  # noqa: E402
    ImageProcessor,
    X8DImageProcessor,
    IMG_START_ID,
    IMG_END_ID,
)
from omni_diffusion.data.processor.audio_processor import (  # noqa: E402
    AudioProcessor,
    X8DAudioProcessor,
    add_audio_input_contiguous,
    AUD_START_ID,
    AUD_END_ID,
)

from omni_diffusion.x8d_dataset import byte_stream_to_rows, rows_to_byte_stream  # noqa: E402
from omni_diffusion.x8d_export import load_gguf  # noqa: E402
from omni_diffusion.x8d_mmap import MappedX8DReader  # noqa: E402

try:
    from tools.import_hf_dataset import (  # noqa: E402
        convert_jsonl_to_x8d,
        jsonl_to_rows,
    )
    _HAS_TOOL = True
except ImportError:
    _HAS_TOOL = False


def _write_binary(tmp_dir, name, payload):
    path = os.path.join(tmp_dir, name)
    with open(path, "wb") as f:
        f.write(payload)
    return path


class ImageProcessorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_image_to_bytes_raw(self):
        p = ImageProcessor()
        path = _write_binary(self._tmp, "img.png", b"\x89PNG\x00\x01\xff")
        self.assertEqual(p.image_to_bytes(path), [0x89, 0x50, 0x4E, 0x47, 0x00, 0x01, 0xFF])

    def test_call_flag_bounded(self):
        p = ImageProcessor()
        path = _write_binary(self._tmp, "img.jpg", b"\xff\xd8\xff")
        ids = p(path)
        self.assertEqual(ids[0], IMG_START_ID)
        self.assertEqual(ids[-1], IMG_END_ID)
        self.assertEqual(ids[1:-1], [0xFF, 0xD8, 0xFF])
        self.assertTrue(all(0 <= i <= 263 for i in ids))

    def test_default_flag_ids_match_byte_law(self):
        p = ImageProcessor()
        self.assertEqual(p.start_id, 260)
        self.assertEqual(p.end_id, 261)

    def test_legacy_256_257_ids_rejected(self):
        with self.assertRaises(ValueError):
            ImageProcessor(img_start_id=256)
        with self.assertRaises(ValueError):
            ImageProcessor(img_start_id=257)

    def test_x8d_alias(self):
        self.assertIs(X8DImageProcessor, ImageProcessor)

    def test_legacy_methods_kept(self):
        p = ImageProcessor()
        p.load_model()
        path = _write_binary(self._tmp, "img.bin", b"abc")
        self.assertEqual(p.process_images([path]), [[97, 98, 99]])
        self.assertEqual(p.process_images_with_subpatch(path), [97, 98, 99])
        self.assertEqual(p.get_image_token([97]), [97])

    def test_to_tensor_stdlib_fallback(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            p = ImageProcessor()
            path = _write_binary(self._tmp, "img.bin", b"ab")
            self.assertEqual(p.to_tensor(path), [260, 97, 98, 261])
        else:
            self.skipTest("torch installed; tensor path covered elsewhere")


class AudioProcessorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_audio_to_bytes_raw(self):
        p = AudioProcessor()
        path = _write_binary(self._tmp, "a.wav", b"RIFF\x00\xff\x00")
        self.assertEqual(p.audio_to_bytes(path), [0x52, 0x49, 0x46, 0x46, 0x00, 0xFF, 0x00])

    def test_call_flag_bounded(self):
        p = AudioProcessor()
        path = _write_binary(self._tmp, "a.wav", b"\x00\xff\x80")
        ids = p(path)
        self.assertEqual(ids[0], AUD_START_ID)
        self.assertEqual(ids[-1], AUD_END_ID)
        self.assertEqual(ids[1:-1], [0x00, 0xFF, 0x80])
        self.assertTrue(all(0 <= i <= 263 for i in ids))

    def test_default_flag_ids_match_byte_law(self):
        p = AudioProcessor()
        self.assertEqual(p.start_id, 262)
        self.assertEqual(p.end_id, 263)

    def test_legacy_258_259_ids_rejected(self):
        with self.assertRaises(ValueError):
            AudioProcessor(aud_start_id=258)
        with self.assertRaises(ValueError):
            AudioProcessor(aud_start_id=259)

    def test_x8d_alias(self):
        self.assertIs(X8DAudioProcessor, AudioProcessor)

    def test_no_glm4voice_loading(self):
        p = AudioProcessor(audio_tokenizer_path="/nonexistent/glm-4-voice-tokenizer",
                           audio_tokenizer_type="sensevoice_glm4voice")
        self.assertIsNone(p.audio_tokenizer)
        self.assertIsNone(p.audio_tokenizer_type)
        self.assertTrue(p.is_discrete)

    def test_process_audios_legacy_api(self):
        p = AudioProcessor()
        path = _write_binary(self._tmp, "a.wav", b"\x01\x02")
        self.assertEqual(p.process_audios(path, is_discrete=True), [1, 2])

    def test_to_tensor_stdlib_fallback(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            p = AudioProcessor()
            path = _write_binary(self._tmp, "a.wav", b"ab")
            self.assertEqual(p.to_tensor(path), [262, 97, 98, 263])
        else:
            self.skipTest("torch installed; tensor path covered elsewhere")


class AddAudioInputContiguousTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_inject_replaces_placeholders(self):
        path = _write_binary(self._tmp, "a.wav", b"\x00\xff")
        input_ids = [260, 97, 98, 262, 99, 263]
        new_ids, audios, spans = add_audio_input_contiguous(input_ids, [path])
        self.assertEqual(audios, [[262, 0, 255, 263]])
        self.assertEqual(spans, [[3, 7]])
        self.assertEqual(new_ids, [260, 97, 98, 262, 0, 255, 263, 99, 263])

    def test_no_placeholders_keeps_sequence(self):
        path = _write_binary(self._tmp, "a.wav", b"\x01")
        new_ids, audios, spans = add_audio_input_contiguous([1, 2, 3], [path])
        self.assertEqual(new_ids, [1, 2, 3])
        self.assertEqual(audios, [])
        self.assertEqual(spans, [])


class JsonlPipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _write_jsonl(self, rows):
        path = os.path.join(self._tmp, "shard.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def test_jsonl_to_rows_text_and_code(self):
        path = self._write_jsonl([
            {"text": "नमस्ते", "code": "print(1)"},
            {"text": "hello", "meta": {"x": 1}},
            {"code": "fn"},
        ])
        rows = jsonl_to_rows(path)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["text"], "नमस्तेprint(1)")
        self.assertEqual(rows[1]["text"], "hello")
        self.assertEqual(rows[2]["text"], "fn")

    def test_jsonl_to_rows_limit(self):
        path = self._write_jsonl([{"text": str(i)} for i in range(10)])
        rows = jsonl_to_rows(path, limit=3)
        self.assertEqual(len(rows), 3)

    @unittest.skipUnless(_HAS_TOOL, "tools/import_hf_dataset.py not importable")
    def test_convert_jsonl_lossless(self):
        path = self._write_jsonl([
            {"text": "byte-native canvas " * 20, "code": "x = 0b1010"},
            {"text": "दूसरी पंक्ति", "code": "print('दूसरी')"},
        ])
        out = os.path.join(self._tmp, "out", "sangraha.x8dds.gguf")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        manifest = convert_jsonl_to_x8d(path, out, name="sangraha")
        self.assertTrue(manifest["roundtrip_lossless"])
        self.assertEqual(manifest["rows_count"], 2)

        payloads, _ = load_gguf(out)
        stream = payloads["sangraha"]
        self.assertTrue(stream.startswith(b"X8DDS\x00\x01"))
        rows = byte_stream_to_rows(stream)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "byte-native canvas " * 20 + "x = 0b1010")

        with MappedX8DReader(out) as reader:
            self.assertEqual(reader.load("sangraha"), stream)
            self.assertTrue(reader.names())

    @unittest.skipUnless(_HAS_TOOL, "tools/import_hf_dataset.py not importable")
    def test_convert_jsonl_missing_file(self):
        from omni_diffusion.x8d_dataset import X8DDatasetError

        with self.assertRaises(X8DDatasetError):
            convert_jsonl_to_x8d(
                os.path.join(self._tmp, "nope.jsonl"),
                os.path.join(self._tmp, "o.x8dds.gguf"),
                "nope",
            )

    @unittest.skipUnless(_HAS_TOOL, "tools/import_hf_dataset.py not importable")
    def test_convert_jsonl_u8_not_float32(self):
        # The byte law stores raw U8 coordinates -- never float32 packs.
        # A float32 pack would be 4x the payload; assert the container stores
        # the exact U8 stream (raw bytes 1:1, no 4-byte float bloat).
        path = self._write_jsonl([{"text": "aaaa"}])
        out = os.path.join(self._tmp, "out2", "ds.x8dds.gguf")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        convert_jsonl_to_x8d(path, out, name="ds")
        payloads, _ = load_gguf(out)
        stream = payloads["ds"]
        expected = rows_to_byte_stream([{"text": "aaaa"}])
        self.assertEqual(stream, expected)
        self.assertLess(len(stream), 4 * len(b"aaaa") + 128)


if __name__ == "__main__":
    unittest.main()
