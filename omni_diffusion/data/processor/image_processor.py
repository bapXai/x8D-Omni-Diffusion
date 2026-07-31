# coding=utf-8
"""Byte-native image processor.

The byte-native framework treats images as raw 8-bit byte streams at ids
0-255 — the same vocabulary as text and audio. No image tokenizer exists
(no MagViT-v2, no VAE, no BPE); a pixel file IS its byte array.

    list(open(path, "rb").read())

That array is placed on the diffusion canvas between IMG_START (260) and
IMG_END (261) and the denoiser fills it like any other byte span.
"""

from __future__ import annotations

import os
from typing import List, Union


class ImageProcessor:
    """Turn image files into raw byte arrays (pure stdlib, no tokenizer)."""

    def __init__(self, *args, **kwargs):  # noqa: D401
        # Legacy kwargs (model_path, process_type, ...) are accepted and
        # ignored: the byte-native path needs no model, no resize, no norm.
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
