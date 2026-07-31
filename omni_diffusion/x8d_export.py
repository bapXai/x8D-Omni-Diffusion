# coding=utf-8
"""x8Dsub-byte 0.001 sub-byte weight compression + x8D .gguf container export.

Pure Python standard library only. Mirrors the `bapXai/x8Dsub-byte` repo
(`x8Dquanta/__init__.py`): ``Quanta[i] = weight_byte[i] * 0.001`` stored as
U8 coordinates in a header-less container with the ``X8DGGUF1`` magic.

Zero-copy mmap serving: the compressed state IS the running state. The
inverse math (``/ 0.001``) operates as a live coordinate pointer map at
inference time.
"""

from __future__ import annotations

import mmap
import os
import struct
from typing import BinaryIO, Dict, Iterable, List, Mapping, Optional, Tuple

#: The x8D sub-byte scaling law.
LAW: float = 0.001

#: Container magic for x8D GGUF files.
GGUF_MAGIC: bytes = b"X8DGGUF1"

#: Version + quantization tag (single byte). U8 = raw byte coordinates.
_HEADER_FMT = "<8sQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

X8D_HEADER = struct.pack(_HEADER_FMT, GGUF_MAGIC, 0)


class X8DHeaderError(ValueError):
    """Raised when a file does not carry a valid x8D header."""


def quantize(weight_bytes: Iterable[int]) -> List[float]:
    """Apply the 0.001 law: ``Quanta[i] = weight_byte[i] * 0.001``.

    Args:
        weight_bytes: raw uint8 weight bytes (0-255).

    Returns:
        List of sub-byte coordinates in the fractional domain.
    """
    return [float(int(b) & 0xFF) * LAW for b in weight_bytes]


def dequantize(quanta: Iterable[float]) -> bytes:
    """Invert the 0.001 law: ``weight_byte = round(quanta / 0.001)``.

    Args:
        quanta: iterable of sub-byte coordinates.

    Returns:
        Restored raw byte string.
    """
    return bytes([int(round(float(q) / LAW)) & 0xFF for q in quanta])


def to_u8(quanta: Iterable[float]) -> bytes:
    """Project sub-byte coordinates back onto the U8 byte axis.

    The quanta are coordinates in [0.0, 0.255]; their U8 byte projection
    (``round(q / LAW)``) is exactly the original byte, so storage is lossless
    in raw byte form. This is the storage half of the pointer map.
    """
    return bytes([int(round(float(q) / LAW)) & 0xFF for q in quanta])


def save_gguf(file_payloads: Mapping[str, bytes], filename: str) -> str:
    """Write payloads into a pure x8D GGUF container.

    No JSON, no float bloat, no character metadata -- only raw U8 byte
    coordinates behind the ``X8DGGUF1`` magic.

    Args:
        file_payloads: mapping of name -> raw bytes (already quantized or
            raw weight bytes; both are stored as U8).
        filename: output path.

    Returns:
        The output path.
    """
    if not isinstance(file_payloads, dict):
        raise TypeError("file_payloads must be a dict[str, bytes]")

    with open(filename, "wb") as f:
        f.write(X8D_HEADER)
        for name, data in file_payloads.items():
            if not isinstance(data, (bytes, bytearray)):
                data = to_u8(int(b) & 0xFF for b in data)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<Q", len(data)))
            f.write(bytes(data))
    return filename


def load_gguf(filename: str) -> Tuple[Dict[str, bytes], Dict[str, object]]:
    """Read an x8D GGUF container, returning payloads and metadata.

    Args:
        filename: path to the .gguf container.

    Returns:
        ``(payloads, metadata)`` where payloads maps name -> raw U8 bytes.
    """
    with open(filename, "rb") as f:
        magic = f.read(len(GGUF_MAGIC))
        if magic != GGUF_MAGIC:
            raise X8DHeaderError(
                f"Not a valid x8D GGUF container (magic {magic!r} != {GGUF_MAGIC!r})"
            )
        f.seek(_HEADER_SIZE)  # skip magic + reserved version field
        payloads: Dict[str, bytes] = {}
        while True:
            name_len_b = f.read(4)
            if not name_len_b:
                break
            (name_len,) = struct.unpack("<I", name_len_b)
            name = f.read(name_len).decode("utf-8")
            (data_len,) = struct.unpack("<Q", f.read(8))
            payloads[name] = f.read(data_len)
    return payloads, {"law": LAW, "container": "x8D GGUF U8"}


def mmap_load_gguf(filename: str) -> Tuple[mmap.mmap, Dict[str, object]]:
    """Zero-copy mmap the container: compressed state IS the running state.

    Returns a read-only memory map over the whole file plus metadata. The
    caller can slice payloads out of the map directly -- no decompression
    loop is ever run.

    Args:
        filename: path to the .gguf container.

    Returns:
        ``(mapping, metadata)``.
    """
    file_size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    if mapping[: len(GGUF_MAGIC)] != GGUF_MAGIC:
        raise X8DHeaderError("Not a valid x8D GGUF container")
    return mapping, {"law": LAW, "container": "x8D GGUF U8", "size_bytes": file_size}


def quantize_weights_from_bf16_checkpoint(source: BinaryIO, count: int) -> List[float]:
    """Placeholder: quantize raw bf16 weight bytes via the 0.001 law.

    In the real pipeline (issue #3) this reads a bf16/fp32 checkpoint,
    rounds each weight to its nearest byte, and maps bytes into sub-byte
    coordinates. This stub keeps the module pure-Python and dependency-free.
    """
    data = source.read(count)
    return quantize(data)


def verify_framework_alignment(data_size: int = 500_000_000) -> Dict[str, float]:
    """Mirror of `x8Dsub-byte/verify_framework_alignment.py`.

    Generates ``data_size`` random bytes, stores them via x8D sub-byte
    mapping, and reports the on-disk ratio (1:1 -- storage is lossless U8).

    Args:
        data_size: number of bytes to verify (default 500M like upstream).

    Returns:
        Dict with original_size, stored_size and effective_ratio.
    """
    import tempfile

    test_data = os.urandom(data_size)
    tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
    tmp.close()
    try:
        save_gguf({"research_weights": test_data}, tmp.name)
        stored = os.path.getsize(tmp.name)
        payloads, _ = load_gguf(tmp.name)
        payload_bytes = len(payloads["research_weights"])
        restored = payloads["research_weights"] == test_data
    finally:
        os.remove(tmp.name)
    return {
        "original_size": float(data_size),
        "stored_size": float(stored),
        "payload_size": float(payload_bytes),
        "effective_ratio": data_size / payload_bytes,
        "lossless": restored,
    }


def percent_reduction(baseline_bytes: float, compressed_bytes: float) -> float:
    """Report disk reduction vs a float-bloat baseline (e.g. BF16 = 2x bytes).

    Args:
        baseline_bytes: size of the uncompressed representation.
        compressed_bytes: size after x8D container storage.

    Returns:
        Reduction as a percentage in [0.0, 100.0].
    """
    return (1.0 - compressed_bytes / baseline_bytes) * 100.0
