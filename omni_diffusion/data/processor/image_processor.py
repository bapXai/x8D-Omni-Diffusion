# coding=utf-8
"""Byte-native image processor — pure binary slicing (no MagViT, no VAE).

The byte-native framework treats images as raw 8-bit byte streams at ids
0-255 — the same vocabulary as text and audio. No image tokenizer exists
(no MagViT-v2, no VAE, no BPE); a pixel file IS its byte array.

    list(open(path, "rb").read())

That array is placed on the diffusion canvas between IMG_START (260) and
IMG_END (261) and the denoiser fills it like any other byte span. The byte
law fixes IMG_START=260 / IMG_END=261 (MASK=256, PAD=257, BOS=258, EOS=259);
the legacy 256/257 ids are rejected.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

#: Byte-law special ids (see AGENTS.md foundational law / DreamConfig).
MASK_ID: int = 256
PAD_ID: int = 257
BOS_ID: int = 258
EOS_ID: int = 259
IMG_START_ID: int = 260
IMG_END_ID: int = 261

#: Legacy ids are forbidden: they collide with MASK/PAD.
_INVALID_START_IDS: frozenset = frozenset((MASK_ID, PAD_ID, BOS_ID, EOS_ID))


class ImageProcessor:
    """Turn image files into raw byte arrays bounded by IMG_START/IMG_END.

    Pure stdlib (no torch, no numpy): the canonical output is a ``list`` of
    ints 0-263. The torch tensor path is available via :meth:`to_tensor` and
    only imports torch lazily (the byte core never requires it).
    """

    def __init__(
        self,
        *args,
        img_start_id: int = IMG_START_ID,
        img_end_id: int = IMG_END_ID,
        **kwargs,
    ) -> None:
        # Legacy kwargs (model_path, process_type, image_size, ...) are
        # accepted and ignored: the byte-native path needs no model, no
        # resize, no norm.
        if img_start_id in _INVALID_START_IDS:
            raise ValueError(
                f"img_start_id {img_start_id} collides with MASK/PAD/BOS/EOS "
                f"({sorted(_INVALID_START_IDS)}); byte law uses IMG_START=260"
            )
        self.start_id = int(img_start_id)
        self.end_id = int(img_end_id)
        self.image_tokenizer = None
        self.patch_size = None

    def load_model(self) -> None:
        """No-op — byte-native images need no model to load."""

    def image_to_bytes(self, path: str) -> List[int]:
        """Read an image file as raw bytes (ids 0-255).

        Args:
            path: path to a pixel file (png/jpg/wav/raw...).

        Returns:
            The file's byte array.
        """
        with open(path, "rb") as f:
            return list(f.read())

    def __call__(self, image_path: str) -> List[int]:
        """Binary-slice a visual file bounded by IMG_START/IMG_END flags.

        Args:
            image_path: path to a pixel file.

        Returns:
            ``[IMG_START] + file_bytes + [IMG_END]`` — the diffusion canvas
            span for one image, ids 0-263.
        """
        raw = self.image_to_bytes(os.fspath(image_path))
        return [self.start_id] + raw + [self.end_id]

    def to_tensor(self, image_path: str):
        """Torch-optional LongTensor of the flag-bounded byte canvas.

        Imports torch lazily; if torch is unavailable, falls back to the
        stdlib ``list`` output (the byte core stays dependency-free).

        Args:
            image_path: path to a pixel file.

        Returns:
            A ``torch.LongTensor`` of ids 0-263, or a plain ``list`` when
            torch is not installed.
        """
        ids = self(image_path)
        try:
            import torch
        except ImportError:
            return ids
        return torch.tensor(ids, dtype=torch.long)

    def process_images(self, img_or_path_list: List[Union[str, object]], image_resolution=None):
        """Byte-native: return one raw byte array per input path.

        Args:
            img_or_path_list: list of file paths (or objects with __fspath__).
            image_resolution: ignored (bytes preserve native resolution).

        Returns:
            list of byte arrays (ids 0-255).
        """
        return [self.image_to_bytes(os.fspath(p)) for p in img_or_path_list]

    def process_images_with_subpatch(self, img_or_path, image_resolution=None):
        """Byte-native single-image helper (legacy subpatch API kept)."""
        return self.image_to_bytes(os.fspath(img_or_path))

    def get_image_token(self, image_bytes):
        """Byte-native identity: the bytes ARE the tokens."""
        return image_bytes


#: Explicit x8D name for the byte-native processor (identical behaviour).
X8DImageProcessor = ImageProcessor
