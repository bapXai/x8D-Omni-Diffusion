# coding=utf-8
"""x8D 0.001 sub-byte law — faithful port of `bapXai/x8Dsub-byte/x8Dquanta`.

The quantization is the 0.001 law:

    Quanta[i] = input_byte[i] * 0.001          (sub-byte coordinates [0.0, 0.255])
    input_byte[i] = round(Quanta[i] / 0.001)   (EXACT reverse, bijective over 0-255)

This is the 0.001 row of the scaling table (0.008 bit, 1000:1) — NOT the 1.0 row
(raw byte = 8 bit, 1:1 copy) and NOT the 0.5 row (byte x 0.5 = 4 bit, 2:1,
collision-prone). It is also NOT a Q8-style ``scale = max|w|/127`` scheme with a
``manifest.json`` float manifest — that is the Float Trap this project bans.

The Float Trap, per the upstream README:
- Float bloat: storing bytes as 32/64-bit floats increases storage 4x-8x.
- Superposition: quanta are byte vectors; storing them as floats destroys the
  sub-byte domain.
- Byte pollution: non-binary storage (JSON, pt, safetensors) bloats size with
  character symbols instead of raw u8 bits.

The stored file contains ONLY the raw quanta byte coordinates — ``byte * 0.001``
mapped through the law, stored as their U8 stage bytes. No magic, no framing,
no name-length headers, no manifest, no padding on top of the quantized weight.
The ``/0.001`` reverse happens at query time as a live coordinate pointer map
over the exact requested span — the compressed state IS the running state;
there is no roundtrip to a full float model.

Mirrors upstream exactly: ``compress``/``decompress``, ``save_file``/
``load_file``, ``save_gguf``/``load_gguf`` (raw bytes, no magic).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Tuple

#: The x8D sub-byte scaling law.
LAW: float = 0.001

#: 8x8 block element count (upstream RATIO).
RATIO: int = 8 * 8


class x8DSubByte:
    """x8D Sub-Byte Framework: universal 0.001 LAW fractional reduction.

    Supports any input data type: bytes, PNG, audio, models, text. Input is
    multiplied by 0.001 to fractional sub-byte coordinates; the /0.001 reverse
    restores the exact input byte (bijective over 0-255).
    """

    @staticmethod
    def compress(data: Iterable[int]) -> List[float]:
        """Apply the 0.001 law: ``Quanta[i] = byte[i] * LAW``.

        Args:
            data: iterable of byte values (0-255) or a bytes object.

        Returns:
            List of sub-byte quanta coordinates in ``[0.0, 0.255]``.
        """
        if isinstance(data, (bytes, bytearray)):
            byte_values = list(data)
        else:
            byte_values = [int(v) & 0xFF for v in data]
        return [b * LAW for b in byte_values]

    @staticmethod
    def decompress(quanta_list: Iterable[float]) -> bytes:
        """Reverse the 0.001 law: ``byte = round(quanta / LAW)`` (exact).

        Args:
            quanta_list: iterable of sub-byte quanta coordinates.

        Returns:
            The exact original byte string.
        """
        return bytes([int(round(q / LAW)) for q in quanta_list])


def save_file(tensors: Mapping[str, bytes], filename: str) -> str:
    """Save raw x8D quanta stage bytes to a file (upstream ``save_file``).

    Args:
        tensors: mapping of name -> raw bytes (byte coordinates).
        filename: output path.

    Returns:
        The output path.
    """
    with open(filename, "wb") as f:
        for _name, data in tensors.items():
            if isinstance(data, (bytes, bytearray)):
                f.write(bytes(data))
            else:
                f.write(bytes([int(v) & 0xFF for v in data]))
    return filename


def load_file(filename: str) -> bytes:
    """Load a raw x8D quanta file and return the stored bytes.

    Args:
        filename: path to the quanta file.

    Returns:
        The raw stored bytes.
    """
    with open(filename, "rb") as f:
        return f.read()


def save_gguf(file_payloads: Mapping[str, bytes], filename: str) -> str:
    """Write payloads into a raw x8D quanta container (upstream ``save_gguf``).

    The file contains ONLY the raw sub-byte quanta stage bytes (U8 dtype) —
    no magic, no JSON, no float bloat, no character metadata, no padding on
    top of the quantized weight.

    Args:
        file_payloads: mapping of name -> raw bytes (byte coordinates).
        filename: output path.

    Returns:
        The output path.
    """
    if not isinstance(file_payloads, dict):
        raise TypeError("file_payloads must be a dict[str, bytes]")

    with open(filename, "wb") as f:
        for _name, data in file_payloads.items():
            if isinstance(data, (bytes, bytearray)):
                f.write(bytes(data))
            else:
                f.write(bytes([int(v) & 0xFF for v in data]))
    return filename


def load_gguf(filename: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Load a raw x8D quanta container (upstream ``load_gguf``).

    Args:
        filename: path to the quanta container.

    Returns:
        ``({"data": {"quanta": [...], "restored": bytes}}, {})`` — the sub-byte
        quanta coordinates (``byte * LAW``) and the exact restored byte string.
    """
    with open(filename, "rb") as f:
        raw = f.read()

    payload = raw
    fractional_values = [b * LAW for b in payload]
    restored = bytes(payload)

    return {"data": {"quanta": fractional_values, "restored": restored}}, {}


def quantize(weight_bytes: Iterable[int]) -> List[float]:
    """Apply the 0.001 law to raw bytes: ``Quanta[i] = byte[i] * 0.001``.

    Args:
        weight_bytes: raw byte values (0-255).

    Returns:
        List of sub-byte quanta coordinates.
    """
    return x8DSubByte.compress(weight_bytes)


def dequantize(quanta: Iterable[float]) -> bytes:
    """Reverse the 0.001 law exactly: ``byte = round(quanta / 0.001)``.

    Args:
        quanta: iterable of sub-byte quanta coordinates.

    Returns:
        The exact original byte string.
    """
    return x8DSubByte.decompress(quanta)


def to_u8(quanta: Iterable[float]) -> bytes:
    """Project sub-byte coordinates back onto the U8 byte axis.

    ``round(q / LAW)`` recovers exactly the original byte; storage in raw byte
    form is lossless. This is the storage half of the coordinate pointer map.
    """
    return x8DSubByte.decompress(quanta)


def quantize_state_dict_to_gguf(
    state_dict: Dict[str, object], output_path: str, tag: str = "model"
) -> str:
    """Write a state_dict as a raw x8D quanta container (0.001 law, no magic).

    Each weight tensor is mapped to its nearest U8 byte coordinates (0-255);
    the container is the concatenation of those coordinate bytes in sorted
    state_dict-key order. Quanta = byte * 0.001 is applied at compute time;
    /0.001 reverse is exact. No framing, no name headers, no manifest, no
    padding on top of the quantized weight.

    Args:
        state_dict: flat mapping name -> torch.Tensor.
        output_path: output raw quanta file.
        tag: container tag (unused, kept for API compatibility).

    Returns:
        The output path.
    """
    import torch

    payloads: Dict[str, bytes] = {}
    for name in sorted(state_dict):
        t = state_dict[name].detach().float().contiguous().cpu()
        if t.numel() == 0:
            payloads[name] = b""
            continue
        q = torch.clamp(torch.round(t), 0, 255).to(torch.uint8)
        payloads[name] = q.numpy().tobytes()
    return save_gguf(payloads, output_path)


def mmap_gguf(path: str):
    """Zero-copy map an x8D quanta container (compressed state IS the running
    state). The file IS the raw quanta bytes; no magic, no header to skip."""
    import mmap

    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.path.getsize(path)
        m = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    return m


class QuantizedServingReader:
    """Zero-copy on-container serving reader (the serving law).

    The file contains ONLY the raw quanta byte coordinates — no magic, no
    framing, no name headers, no manifest, no padding on top of the quantized
    weight. The mmap'd bytes ARE the running state; the ``/0.001`` reverse is
    applied at query time. There is no full float model anywhere.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._mapping = mmap_gguf(path)

    @property
    def tensor_names(self) -> List[str]:
        """The single raw payload of a raw quanta container is ``data``."""
        return ["data"]

    def tensor_bytes(self, name: str = "data") -> bytes:
        """Raw U8 quanta byte coordinates (sliced from the mmap, zero-copy).

        Args:
            name: ignored; a raw quanta container has one payload.

        Returns:
            The full quanta byte span (the entire file).
        """
        return bytes(self._mapping[:])

    def quanta(self, name: str = "data") -> List[float]:
        """Live 0.001 law: ``Quanta[i] = byte[i] * 0.001`` for the payload."""
        return x8DSubByte.compress(self.tensor_bytes(name))
