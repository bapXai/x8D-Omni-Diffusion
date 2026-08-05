"""Tests for :mod:`omni_diffusion.x8d_media` — real media files (#48).

Validates that image/audio/video payloads are GENUINE containers (PNG, WAV,
AVI) with correct structure, deterministically derived from the prompt bytes
— not placeholder text stuffed into a media wire.
"""

import struct
import unittest
import zlib

from omni_diffusion.x8d_media import (
    avi_bytes,
    png_encode,
    procedural_avi,
    procedural_pcm,
    procedural_png,
    procedural_rgb,
    procedural_wav,
    seeded_rng,
    wav_bytes,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_chunks(png: bytes):
    assert png.startswith(PNG_MAGIC)
    i = 8
    chunks = {}
    while i < len(png):
        length = struct.unpack_from(">I", png, i)[0]
        tag = png[i + 4 : i + 8]
        chunks[tag] = png[i + 8 : i + 8 + length]
        i += 12 + length
    return chunks


class PngMediaTest(unittest.TestCase):
    def test_magic_and_structure(self):
        png = procedural_png("a byte-native cat")
        self.assertTrue(png.startswith(PNG_MAGIC))
        chunks = _png_chunks(png)
        self.assertIn(b"IHDR", chunks)
        self.assertIn(b"IDAT", chunks)
        self.assertIn(b"IEND", chunks)

    def test_ihdr_dimensions_and_colortype(self):
        png = procedural_png("cat", size=64)
        ihdr = _png_chunks(png)[b"IHDR"]
        width, height, depth, colortype = struct.unpack_from(">IIBB", ihdr, 0)
        self.assertEqual((width, height), (64, 64))
        self.assertEqual(depth, 8)
        self.assertEqual(colortype, 2)

    def test_idat_decompresses_to_filtered_scanlines(self):
        png = procedural_png("dog", size=32)
        chunks = _png_chunks(png)
        raw = zlib.decompress(chunks[b"IHDR"] and chunks[b"IDAT"])
        self.assertEqual(len(raw), 32 * (1 + 32 * 3))

    def test_deterministic_same_prompt(self):
        self.assertEqual(procedural_png("cat"), procedural_png("cat"))
        self.assertNotEqual(procedural_png("cat"), procedural_png("dog"))

    def test_png_encode_rejects_wrong_row_count(self):
        with self.assertRaises(ValueError):
            png_encode(2, 2, [bytes(6)])


class WavMediaTest(unittest.TestCase):
    def test_riff_wave_header(self):
        wav = procedural_wav("hello tone")
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")

    def test_pcm_format_and_sample_rate(self):
        wav = procedural_wav("tone")
        fmt_len = struct.unpack_from("<I", wav, 16)[0]
        audio_format, channels, rate, _byte_rate, _align, bits = struct.unpack_from(
            "<HHIIHH", wav, 20
        )
        self.assertEqual(fmt_len, 16)
        self.assertEqual(audio_format, 1)
        self.assertEqual(channels, 1)
        self.assertEqual(rate, 16000)
        self.assertEqual(bits, 16)

    def test_pcm_is_audio_not_text(self):
        pcm = procedural_pcm("speak")
        self.assertEqual(len(pcm) % 2, 0)
        samples = struct.unpack(f"<{len(pcm)//2}h", pcm)
        self.assertTrue(any(abs(s) > 0 for s in samples))
        self.assertTrue(all(-32768 <= s <= 32767 for s in samples))

    def test_wav_deterministic(self):
        self.assertEqual(procedural_wav("tone"), procedural_wav("tone"))
        self.assertNotEqual(procedural_wav("tone"), procedural_wav("other"))

    def test_wav_bytes_wraps_pcm(self):
        pcm = procedural_pcm("x", seconds=0.25)
        wav = wav_bytes(pcm, 16000)
        self.assertEqual(struct.unpack_from("<I", wav, 40)[0], len(pcm))


class AviMediaTest(unittest.TestCase):
    def test_riff_avi_header(self):
        avi = procedural_avi("video", frames=8)
        self.assertEqual(avi[:4], b"RIFF")
        self.assertEqual(avi[8:12], b"AVI ")
        self.assertIn(b"hdrl", avi)
        self.assertIn(b"movi", avi)
        self.assertIn(b"idx1", avi)

    def test_avih_frame_count_and_rate(self):
        avi = procedural_avi("video", frames=16, fps=8)
        avih_pos = avi.find(b"avih")
        micro, _maxb, _pad, flags, total = struct.unpack_from("<5I", avi, avih_pos + 4)
        self.assertEqual(total, 16)
        self.assertEqual(micro, 1000000 // 8)
        self.assertEqual(flags, 0)

    def test_movi_holds_one_chunk_per_frame(self):
        avi = procedural_avi("video", frames=16)
        movi_start = avi.find(b"movi")
        movi_end = avi.find(b"idx1")
        self.assertEqual(avi[movi_start:movi_end].count(b"00db"), 16)

    def test_video_deterministic(self):
        self.assertEqual(procedural_avi("video"), procedural_avi("video"))
        self.assertNotEqual(procedural_avi("video"), procedural_avi("other"))

    def test_avi_bytes_packs_given_frames(self):
        rows = [procedural_rgb(f"v:{i}", 16) for i in range(4)]
        avi = avi_bytes(rows, 16, fps=4)
        self.assertEqual(avi[:4], b"RIFF")


class SeededRngTest(unittest.TestCase):
    def test_seeded_rng_is_deterministic(self):
        r1 = seeded_rng(b"prompt")
        r2 = seeded_rng(b"prompt")
        self.assertEqual([r1.random() for _ in range(5)], [r2.random() for _ in range(5)])

    def test_seeded_rng_differs_by_prompt(self):
        r1 = seeded_rng(b"cat")
        r2 = seeded_rng(b"dog")
        self.assertNotEqual(r1.random(), r2.random())


if __name__ == "__main__":
    unittest.main()
