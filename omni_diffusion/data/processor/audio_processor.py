# coding=utf-8
"""Byte-native audio processor — raw PCM bytes, no GLM-4-Voice tokenizer.

The legacy GLM-4-Voice / SenseVoice tokenizer loading sequence is deleted.
Audio files are read as raw uncompressed byte streams (PCM byte samples) and
placed on the diffusion canvas between AUD_START (262) and AUD_END (263).
The byte law fixes AUD_START=262 / AUD_END=263 (MASK=256, PAD=257, BOS=258,
EOS=259, IMG_START=260, IMG_END=261); legacy 258/259 ids are rejected.

Pure stdlib core: the canonical output is a ``list`` of ints 0-263. The
torch tensor path is available via :meth:`to_tensor` and imports torch
lazily, keeping the byte core dependency-free.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

#: Byte-law special ids (see AGENTS.md foundational law / DreamConfig).
MASK_ID: int = 256
PAD_ID: int = 257
BOS_ID: int = 258
EOS_ID: int = 259
IMG_START_ID: int = 260
IMG_END_ID: int = 261
AUD_START_ID: int = 262
AUD_END_ID: int = 263

#: Legacy ids are forbidden: they collide with BOS/EOS.
_INVALID_START_IDS: frozenset = frozenset((MASK_ID, PAD_ID, BOS_ID, EOS_ID))


class AudioProcessor:
    """Turn audio files into raw byte arrays bounded by AUD_START/AUD_END.

    Pure stdlib: no GLM-4-Voice checkpoint, no SenseVoice, no ``datasets``.
    A PCM/WAV/raw file IS its byte array; downsampling is a byte-filter
    concern at the canvas layer, not a tokenizer here.
    """

    def __init__(
        self,
        audio_tokenizer_path: Optional[str] = None,
        audio_tokenizer_type: Optional[str] = None,
        text_audio_interval_ratio: Optional[object] = None,
        aud_start_id: int = AUD_START_ID,
        aud_end_id: int = AUD_END_ID,
    ) -> None:
        # Legacy tokenizer kwargs are accepted and ignored: the byte-native
        # path loads no model, no checkpoint, no weight file.
        if aud_start_id in _INVALID_START_IDS:
            raise ValueError(
                f"aud_start_id {aud_start_id} collides with MASK/PAD/BOS/EOS "
                f"({sorted(_INVALID_START_IDS)}); byte law uses AUD_START=262"
            )
        self.start_id = int(aud_start_id)
        self.end_id = int(aud_end_id)
        self.audio_tokenizer = None
        self.audio_tokenizer_type = None
        self.text_audio_interval_ratio = None
        self.is_discrete = True
        self.is_contiguous = False

    def load_model(self) -> None:
        """No-op — byte-native audio needs no model to load."""

    def audio_to_bytes(self, audio_path: str) -> List[int]:
        """Read an audio file as raw bytes (ids 0-255).

        Args:
            audio_path: path to a PCM/WAV/raw audio file.

        Returns:
            The file's byte array (byte samples, ids 0-255).
        """
        with open(audio_path, "rb") as f:
            return list(f.read())

    def __call__(self, audio_path: str) -> List[int]:
        """Map a raw PCM wave to a byte array bounded by AUD_START/AUD_END.

        Args:
            audio_path: path to an audio file.

        Returns:
            ``[AUD_START] + file_bytes + [AUD_END]`` — the diffusion canvas
            span for one audio stream, ids 0-263.
        """
        raw = self.audio_to_bytes(os.fspath(audio_path))
        return [self.start_id] + raw + [self.end_id]

    def to_tensor(self, audio_path: str):
        """Torch-optional LongTensor of the flag-bounded byte canvas.

        Imports torch lazily; falls back to the stdlib ``list`` output when
        torch is unavailable.

        Args:
            audio_path: path to an audio file.

        Returns:
            A ``torch.LongTensor`` of ids 0-263, or a plain ``list``.
        """
        ids = self(audio_path)
        try:
            import torch
        except ImportError:
            return ids
        return torch.tensor(ids, dtype=torch.long)

    def process_audios(self, audio_path, is_discrete=False, is_contiguous=False, **kwargs):
        """Byte-native: raw byte array of the audio file.

        ``is_discrete``/``is_contiguous`` are accepted for legacy call-site
        compatibility; both reduce to the same raw byte stream.

        Args:
            audio_path: path to the audio file.
            is_discrete: ignored (bytes are byte-discrete by definition).
            is_contiguous: ignored.

        Returns:
            The audio file's byte array (ids 0-255).
        """
        return self.audio_to_bytes(os.fspath(audio_path))

    def apply_to_role(self, role, **kwargs):
        """Legacy no-op retained for call-site compatibility."""
        return None


#: Explicit x8D name for the byte-native processor (identical behaviour).
X8DAudioProcessor = AudioProcessor


def add_audio_input_contiguous(input_ids, audio_paths, tokenizer=None, audio_tokenizer=None):
    """Inject raw byte audio spans into a text id sequence (byte-native).

    Legacy GLM-4-Voice tokenizer / token lookups removed: each audio file is
    read as raw bytes and its canvas span ``[AUD_START] + bytes + [AUD_END]``
    replaces the audio tag position, no checkpoint required.

    Args:
        input_ids: list of byte ids 0-263 (or token ids).
        audio_paths: list of audio file paths (one per audio tag position).
        tokenizer: ignored (byte-native path needs no tokenizer).
        audio_tokenizer: ignored (byte-native path needs no audio tokenizer).

    Returns:
        ``(new_input_ids, audios, audio_indices)`` where ``audios`` holds the
        raw byte arrays and ``audio_indices`` holds ``[begin, end]`` canvas
        spans per audio.
    """
    placeholder = AUD_START_ID  # audio tag position = AUD_START marker
    positions = [i for i, x in enumerate(input_ids) if x == placeholder]
    audios: List[List[int]] = []
    audio_indices: List[List[int]] = []
    new_input_ids: List[int] = []
    st = 0
    for aud_idx, aud_path in enumerate(audio_paths):
        if aud_idx >= len(positions):
            break
        pos = positions[aud_idx]
        audio = AudioProcessor()(aud_path)
        audios.append(audio)
        new_input_ids += input_ids[st:pos]
        begin = len(new_input_ids)
        new_input_ids += audio
        end = len(new_input_ids)
        audio_indices.append([begin, end])
        st = pos + 1
    new_input_ids += input_ids[st:]
    return new_input_ids, audios, audio_indices
