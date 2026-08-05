"""Real, valid byte-native media generation (pure stdlib, no codec deps).

Implements the issue #48 contract: image/audio/video wire payloads must be
REAL media files (PNG / WAV / AVI), deterministically generated from the
prompt bytes — not placeholder text stuffed into a media container. Every
byte produced is still a raw 8-bit byte at ids 0-255 (byte law): PNG, WAV and
AVI are wire/container formats only, and the content is procedurally derived
from the prompt through a seeded RNG (``sha256(prompt)``) so the same prompt
always yields the same openable file.

The generated content is procedural (abstract blobs, pentatonic melody,
animated frames) — semantic generation (a recognizable object for a word)
requires a trained model, which is out of scope here (see #48).
"""

import hashlib
import math
import random
import struct
import zlib

from typing import List, Sequence


def seeded_rng(seed_bytes: bytes) -> random.Random:
    """Deterministic ``random.Random`` seeded from raw prompt bytes."""
    digest = hashlib.sha256(seed_bytes).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------
# PNG (image)
# ---------------------------------------------------------------------------


def _chunk(tag: bytes, data: bytes) -> bytes:
    """One PNG chunk: length + tag + data + CRC32 of (tag + data)."""
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def png_encode(width: int, height: int, rgb_rows: Sequence[Sequence[int]]) -> bytes:
    """Encode ``height`` scanlines of ``width * 3`` RGB bytes into a PNG.

    Args:
        width: image width in pixels.
        height: image height in pixels.
        rgb_rows: ``height`` sequences, each ``width * 3`` bytes (R,G,B,R,G,B...).

    Returns:
        A complete, valid PNG file (magic + IHDR + IDAT + IEND).
    """
    if len(rgb_rows) != height:
        raise ValueError(f"expected {height} rows, got {len(rgb_rows)}")
    raw = b"".join(b"\x00" + bytes(row) for row in rgb_rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def procedural_rgb(text: str, size: int = 64) -> List[bytes]:
    """Deterministic procedural RGB pixels (abstract seeded blobs).

    Args:
        text: the prompt; its bytes seed the generator.
        size: square side length in pixels.

    Returns:
        ``size`` scanlines, each ``size * 3`` bytes.
    """
    rng = seeded_rng(text.encode("utf-8"))
    pixels = [[0, 0, 0] for _ in range(size * size)]
    for _ in range(rng.randint(4, 9)):
        cx = rng.uniform(0.0, size)
        cy = rng.uniform(0.0, size)
        radius = rng.uniform(size * 0.12, size * 0.35)
        color = (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))
        for y in range(size):
            for x in range(size):
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                if d2 < radius * radius:
                    falloff = max(0.0, 1.0 - math.sqrt(d2) / radius)
                    px = pixels[y * size + x]
                    px[0] = min(255, px[0] + int(color[0] * falloff))
                    px[1] = min(255, px[1] + int(color[1] * falloff))
                    px[2] = min(255, px[2] + int(color[2] * falloff))
    for y in range(size):
        for x in range(size):
            px = pixels[y * size + x]
            grad = int(24 + 32 * ((x + y) / (2 * size)))
            px[0] = (px[0] + grad) // 2
            px[1] = (px[1] + grad) // 2
            px[2] = (px[2] + grad) // 2
    return [bytes(v for px in pixels[y * size : (y + 1) * size] for v in px) for y in range(size)]


def procedural_png(text: str, size: int = 64) -> bytes:
    """Real PNG bytes for a prompt (see :func:`procedural_rgb`)."""
    return png_encode(size, size, procedural_rgb(text, size))


# ---------------------------------------------------------------------------
# WAV / PCM (audio)
# ---------------------------------------------------------------------------


def wav_bytes(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap 16-bit little-endian PCM into a valid RIFF/WAVE container.

    Args:
        pcm: 16-bit LE PCM samples (``sample_rate * channels`` per second).
        sample_rate: samples per second per channel.
        channels: channel count (1 = mono).

    Returns:
        A complete, playable WAV file.
    """
    bits = 16
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits)
    data = struct.pack("<4sI", b"data", len(pcm)) + pcm
    riff_size = 4 + (8 + len(fmt)) + (8 + len(pcm))
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + struct.pack("<4sI", b"fmt ", len(fmt)) + fmt + data


def procedural_pcm(text: str, sample_rate: int = 16000, seconds: float = 1.0) -> bytes:
    """Deterministic playable 16-bit PCM (seeded pentatonic melody).

    Args:
        text: the prompt; its bytes seed the melody (scale, key, notes).
        sample_rate: samples per second.
        seconds: total duration.

    Returns:
        Raw 16-bit little-endian PCM samples (valid audio, not text).
    """
    rng = seeded_rng(text.encode("utf-8"))
    scale = [0, 2, 4, 7, 9]
    total = int(sample_rate * seconds)
    note_len = sample_rate // 8
    base = 220.0 * (1.0 + 0.5 * rng.random())
    pcm = bytearray(total * 2)
    for start in range(0, total, note_len):
        semis = scale[rng.randrange(len(scale))] + rng.randint(0, 12)
        freq = base * (2.0 ** (semis / 12.0))
        end = min(start + note_len, total)
        for i in range(start, end):
            t = (i - start) / note_len
            attack = min(1.0, t * 20.0)
            release = max(0.0, 1.0 - (t - 0.8) / 0.2)
            sample = int(math.sin(2 * math.pi * freq * (i - start) / sample_rate) * attack * release * 9000)
            pcm[i * 2] = sample & 0xFF
            pcm[i * 2 + 1] = (sample >> 8) & 0xFF
    return bytes(pcm)


def procedural_wav(text: str, sample_rate: int = 16000, seconds: float = 1.0) -> bytes:
    """Playable WAV file for a prompt (PCM + RIFF header)."""
    return wav_bytes(procedural_pcm(text, sample_rate, seconds), sample_rate)


# ---------------------------------------------------------------------------
# AVI (video, uncompressed RGB)
# ---------------------------------------------------------------------------


def avi_bytes(frame_rows: Sequence[Sequence[int]], size: int, fps: int = 8) -> bytes:
    """Pack RGB frames into a valid RIFF AVI (BI_RGB, uncompressed).

    Args:
        frame_rows: ``frames`` scanline sets (each an ``size`` list of
            ``size * 3`` bytes, from :func:`procedural_rgb`).
        size: square side length in pixels.
        fps: nominal frame rate.

    Returns:
        A complete AVI file playable by standard media players.
    """
    frames = len(frame_rows)
    frame_bytes = [b"".join(row) for row in frame_rows]
    frame_len = size * size * 3
    avih = struct.pack(
        "<10I4I",
        1_000_000 // fps,
        fps * frame_len,
        0,
        0,
        frames,
        0,
        1,
        frame_len,
        size,
        size,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIII8s",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        frames,
        frame_len,
        10000,
        frame_len,
        struct.pack("<4H", 0, 0, size, size),
    )
    strf = struct.pack("<IiiHHIIIIII", 40, size, size, 1, 24, 0, frame_len, 2835, 2835, 0, 0)
    strl_body = _chunk(b"strh", strh) + _chunk(b"strf", strf)
    strl = b"LIST" + struct.pack("<I", 4 + len(strl_body)) + b"strl" + strl_body
    hdrl_body = _chunk(b"avih", avih) + strl
    hdrl = b"LIST" + struct.pack("<I", 4 + len(hdrl_body)) + b"hdrl" + hdrl_body

    movi_body = b""
    idx = bytearray()
    offset = 4
    for frame in frame_bytes:
        flipped = b"".join(
            frame[y * size * 3 : (y + 1) * size * 3] for y in range(size - 1, -1, -1)
        )
        movi_body += b"00db" + struct.pack("<I", len(flipped)) + flipped
        idx += struct.pack("<4sIII", b"00db", 0x10, offset, len(flipped))
        offset += 8 + len(flipped)
    movi = b"LIST" + struct.pack("<I", 4 + len(movi_body)) + b"movi" + movi_body

    body = hdrl + movi + _chunk(b"idx1", bytes(idx))
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"AVI " + body


def procedural_avi(text: str, frames: int = 16, size: int = 64, fps: int = 8) -> bytes:
    """Real AVI video for a prompt (animated seeded frames)."""
    rows = [procedural_rgb(f"{text}:{i}", size) for i in range(frames)]
    return avi_bytes(rows, size, fps)
