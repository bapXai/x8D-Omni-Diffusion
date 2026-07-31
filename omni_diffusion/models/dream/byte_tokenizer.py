# coding=utf-8
"""Raw 8-bit byte tokenizer for x8D-Omni-Diffusion.

Pure Python standard library only. No torch, no transformers, no BPE,
no vocab.json, no merges.txt.

Vocabulary is the 256 unsigned byte states (0x00-0xFF) plus 8 special
tokens:
    MASK=256, PAD=257, BOS=258, EOS=259,
    IMG_START=260, IMG_END=261, AUD_START=262, AUD_END=263

Total vocab size = 264.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Union

VOCAB_SIZE = 264

MASK_TOKEN_ID = 256
PAD_TOKEN_ID = 257
BOS_TOKEN_ID = 258
EOS_TOKEN_ID = 259
IMG_START_TOKEN_ID = 260
IMG_END_TOKEN_ID = 261
AUD_START_TOKEN_ID = 262
AUD_END_TOKEN_ID = 263

SPECIAL_IDS = frozenset(
    {
        MASK_TOKEN_ID,
        PAD_TOKEN_ID,
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
        IMG_START_TOKEN_ID,
        IMG_END_TOKEN_ID,
        AUD_START_TOKEN_ID,
        AUD_END_TOKEN_ID,
    }
)

#: Map a human-friendly name to the special token id.
SPECIAL_TOKEN_ID_MAP: Dict[str, int] = {
    "mask": MASK_TOKEN_ID,
    "pad": PAD_TOKEN_ID,
    "bos": BOS_TOKEN_ID,
    "eos": EOS_TOKEN_ID,
    "img_start": IMG_START_TOKEN_ID,
    "img_end": IMG_END_TOKEN_ID,
    "aud_start": AUD_START_TOKEN_ID,
    "aud_end": AUD_END_TOKEN_ID,
}


def _to_bytes(data: Union[str, bytes, bytearray, Iterable[int]]) -> bytes:
    """Normalize any accepted input type into a bytes object."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(int(b) & 0xFF for b in data)


class ByteTokenizer:
    """Encode/decode arbitrary byte streams with a 264-id vocabulary.

    The tokenizer performs NO vocabulary lookup and NO sub-word merging.
    Text, images, audio, code, and binaries all reduce to the same raw
    byte array: ``list(data_bytes)``.
    """

    vocab_size: int = VOCAB_SIZE

    mask_token_id = MASK_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    bos_token_id = BOS_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID
    img_start_token_id = IMG_START_TOKEN_ID
    img_end_token_id = IMG_END_TOKEN_ID
    aud_start_token_id = AUD_START_TOKEN_ID
    aud_end_token_id = AUD_END_TOKEN_ID

    def encode(
        self,
        data: Union[str, bytes, bytearray, Iterable[int]],
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Convert input to a list of byte ids in [0, 255].

        With ``add_special_tokens=True`` the stream is wrapped as
        ``[BOS] ... [EOS]``.

        Args:
            data: text (UTF-8 encoded), raw bytes, or an iterable of ints.
            add_special_tokens: whether to prepend BOS and append EOS.

        Returns:
            List of integer ids. Content ids are always 0-255.
        """
        ids = list(_to_bytes(data))
        if add_special_tokens:
            return [BOS_TOKEN_ID, *ids, EOS_TOKEN_ID]
        return ids

    def encode_image(
        self,
        image_bytes: Union[bytes, bytearray, Iterable[int]],
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode raw image pixel bytes wrapped in IMG_START/IMG_END."""
        ids = list(_to_bytes(image_bytes))
        wrapped = [IMG_START_TOKEN_ID, *ids, IMG_END_TOKEN_ID]
        if add_special_tokens:
            return [BOS_TOKEN_ID, *wrapped, EOS_TOKEN_ID]
        return wrapped

    def encode_audio(
        self,
        pcm_bytes: Union[bytes, bytearray, Iterable[int]],
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode raw PCM/audio bytes wrapped in AUD_START/AUD_END."""
        ids = list(_to_bytes(pcm_bytes))
        wrapped = [AUD_START_TOKEN_ID, *ids, AUD_END_TOKEN_ID]
        if add_special_tokens:
            return [BOS_TOKEN_ID, *wrapped, EOS_TOKEN_ID]
        return wrapped

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
        as_bytes: bool = True,
    ) -> Union[bytes, str]:
        """Convert ids back into raw bytes or UTF-8 text.

        Content ids (0-255) are mapped directly back to bytes. Special ids
        are dropped when ``skip_special_tokens`` is True (default).

        Args:
            ids: iterable of integer ids.
            skip_special_tokens: drop ids >= 256 when True.
            as_bytes: return ``bytes`` when True, else decode as UTF-8 str.

        Returns:
            The reconstructed byte string, or its UTF-8 text.
        """
        raw = bytearray()
        for i in ids:
            i = int(i) & 0x1FF
            if skip_special_tokens and i in SPECIAL_IDS:
                continue
            if i < 256:
                raw.append(i)
        if as_bytes:
            return bytes(raw)
        return bytes(raw).decode("utf-8", errors="replace")

    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convenience: encode a UTF-8 string."""
        return self.encode(text, add_special_tokens=add_special_tokens)

    def decode_text(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Convenience: decode ids into a UTF-8 string."""
        return self.decode(ids, skip_special_tokens=skip_special_tokens, as_bytes=False)

    def __call__(
        self,
        text: Union[str, bytes, bytearray, Iterable[int]],
        add_special_tokens: bool = True,
    ) -> Dict[str, List[int]]:
        """Tokenize as a dict for drop-in pipeline compatibility.

        Returns ``{"input_ids": [...], "attention_mask": [...]}``.
        """
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    def __len__(self) -> int:
        return VOCAB_SIZE

    def get_vocab(self) -> Dict[int, str]:
        """Return a complete 264-entry id -> label mapping.

        Bytes 0-255 map to a printable hex label; specials map to their
        token name. Exists for tooling/debugging only -- never used for
        encoding (there is no vocabulary lookup in this framework).
        """
        vocab: Dict[int, str] = {}
        for b in range(256):
            vocab[b] = f"byte({b:02X})"
        for name, i in SPECIAL_TOKEN_ID_MAP.items():
            vocab[i] = f"<{name}>"
        return vocab

    def bytes_to_ids(self, data_bytes: bytes) -> List[int]:
        """The fundamental law: ``list(data_bytes)`` with no encoding step."""
        return list(data_bytes)

    def id_batch(self, sequences: Iterable[Iterable[int]]) -> List[List[int]]:
        """Left-pad a batch to equal length with PAD (id 257)."""
        seqs = [list(s) for s in sequences]
        max_len = max(len(s) for s in seqs) if seqs else 0
        return [
            [PAD_TOKEN_ID] * (max_len - len(s)) + list(s)
            for s in seqs
        ]

    @staticmethod
    def mask_canvas(seq_len: int) -> List[int]:
        """Produce a fully-masked diffusion canvas of MASK (id 256)."""
        return [MASK_TOKEN_ID] * seq_len

    @staticmethod
    def renoise_to_random_bytes(positions: Iterable[int]) -> List[int]:
        """Uniform-state re-noise: rejected positions become random bytes."""
        return [b & 0xFF for b in positions]
