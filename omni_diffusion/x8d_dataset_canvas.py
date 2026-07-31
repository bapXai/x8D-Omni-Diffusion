# coding=utf-8
"""Frame dataset rows into fixed-length byte-diffusion canvases.

DiffusionGemma's ``dataset -> canvas`` framing, applied to the x8D byte law:
every dataset row is flattened to raw field bytes (via
``omni_diffusion.x8d_dataset.flatten_fields``) and framed into canvases of
exactly ``canvas_length`` ids drawn from the 264-id byte vocabulary
(0-255 bytes + specials 256-263).

- Content ids are always raw 0-255 bytes (never a tokenizer vocabulary).
- Modality markers (IMG_START/AUD_START/BOS/EOS, ids 260-263) wrap each
  field's bytes when ``with_markers=True`` so the sampler sees the same
  marker markup as ``byte_tokenizer.encode_image`` / ``encode_audio``.
- The stream tail is padded with PAD(257) up to a whole number of canvases.

Pure Python standard library only.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models.dream.byte_tokenizer import (
    AUD_END_TOKEN_ID,
    AUD_START_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    IMG_END_TOKEN_ID,
    IMG_START_TOKEN_ID,
    PAD_TOKEN_ID,
    SPECIAL_IDS,
)
from .x8d_dataset import flatten_fields

#: Substrings in a flattened field path that mark image-style bytes.
_IMG_HINTS: Tuple[str, ...] = ("image", "img", "visual", "pixel", "vision")
#: Substrings in a flattened field path that mark audio-style bytes.
_AUD_HINTS: Tuple[str, ...] = ("audio", "speech", "pcm", "wave", "sound", "voice")

#: Marker ids that are stripped by ``canvases_to_bytes``.
_STRIP_IDS = SPECIAL_IDS


def _marker_for_path(path: str) -> Tuple[int, int]:
    """Choose the modality marker pair for a flattened field path.

    Args:
        path: dotted field path (e.g. ``image.bytes``, ``audio[0]``).

    Returns:
        ``(start, end)`` marker ids: IMG_START/IMG_END for image-style
        paths, AUD_START/AUD_END for audio-style paths, and BOS/EOS for
        everything else (text/code/bytes).
    """
    lowered = path.lower()
    if any(hint in lowered for hint in _IMG_HINTS):
        return IMG_START_TOKEN_ID, IMG_END_TOKEN_ID
    if any(hint in lowered for hint in _AUD_HINTS):
        return AUD_START_TOKEN_ID, AUD_END_TOKEN_ID
    return BOS_TOKEN_ID, EOS_TOKEN_ID


def rows_to_canvases(
    rows: List[Dict],
    canvas_length: int = 256,
    with_markers: bool = True,
) -> List[List[int]]:
    """Frame dataset row field bytes into fixed-length diffusion canvases.

    Each row is flattened with ``flatten_fields`` into ``(path, bytes)``
    leaves; every field's bytes become a canvas id span (optionally wrapped
    in modality markers). All rows concatenate into one id stream which is
    padded with PAD(257) up to a whole multiple of ``canvas_length`` and
    chunked into equal-length canvases.

    Note: canvases are returned as ``List[List[int]]``, not ``List[bytes]``,
    because the canvas id space is 0-263 and Python ``bytes`` can only hold
    values 0-255. Content ids are raw 0-255 bytes; special ids 256-263 are
    PAD and the modality markers.

    Args:
        rows: list of dataset row dicts (nested dicts/lists are flattened).
        canvas_length: exact length of every returned canvas.
        with_markers: wrap each field's bytes with modality markers
            (IMG_START/IMG_END, AUD_START/AUD_END, or BOS/EOS) when True.

    Returns:
        List of canvases; each has ``len == canvas_length`` and every id is
        in 0..263. Empty input yields ``[]``.

    Raises:
        ValueError: ``canvas_length`` is not positive.
    """
    if canvas_length < 1:
        raise ValueError(f"canvas_length must be >= 1, got {canvas_length}")

    stream: List[int] = []
    for row in rows:
        for path, data in flatten_fields(row):
            if with_markers:
                start, end = _marker_for_path(path)
                stream.append(start)
                stream.extend(data)
                stream.append(end)
            else:
                stream.extend(data)

    if not stream:
        return []

    pad_len = (-len(stream)) % canvas_length
    stream.extend([PAD_TOKEN_ID] * pad_len)

    return [
        stream[i : i + canvas_length]
        for i in range(0, len(stream), canvas_length)
    ]


def canvases_to_bytes(canvases: List[List[int]]) -> bytes:
    """Concatenate canvases, stripping PAD padding and marker ids.

    Args:
        canvases: canvases produced by ``rows_to_canvases`` (lists of int
            ids in 0..263).

    Returns:
        Raw content bytes: the concatenation of every canvas with all
        special ids (PAD, markers, etc.) removed. Content ids are 0-255 and
        never collide with specials, so this recovers exactly the original
        field bytes.
    """
    content = bytearray()
    for canvas in canvases:
        for b in canvas:
            if b not in _STRIP_IDS:
                content.append(b)
    return bytes(content)
