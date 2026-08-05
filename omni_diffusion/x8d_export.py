# coding=utf-8
"""x8Dsub-byte 0.001 sub-byte weight compression + raw x8D container export.

Pure Python standard library only. Mirrors the `bapXai/x8Dsub-byte` repo
(`x8Dquanta/__init__.py`): ``Quanta[i] = weight_byte[i] * 0.001`` stored as
U8 coordinates in a container with NO magic bytes, NO JSON, NO manifest.
The per-payload records carry only a name-length address (needed to route
individual tensors/experts); there is no header of any kind on top of the
quantized weight bytes.

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

#: Precomputed sub-byte coordinate for every byte (0.001 law), for quantize().
_QUANTA_LUT: Tuple[float, ...] = tuple(float(b) * LAW for b in range(256))


def quantize(weight_bytes: Iterable[int]) -> List[float]:
    """Apply the 0.001 law: ``Quanta[i] = weight_byte[i] * 0.001``.

    Args:
        weight_bytes: raw uint8 weight bytes (0-255).

    Returns:
        List of sub-byte coordinates in the fractional domain.
    """
    lut = _QUANTA_LUT
    return [lut[int(b) & 0xFF] for b in weight_bytes]


def dequantize(quanta: Iterable[float]) -> bytes:
    """Invert the 0.001 law: ``weight_byte = round(quanta / 0.001)``.

    Uses a memoized pointer map: the quanta are almost always drawn from the
    256 canonical sub-byte coordinates, so the ``round(q / LAW)`` inverse is
    computed once per distinct coordinate instead of once per element.

    Args:
        quanta: iterable of sub-byte coordinates.

    Returns:
        Restored raw byte string.
    """
    return bytes(_dequantized(quanta))


def to_u8(quanta: Iterable[float]) -> bytes:
    """Project sub-byte coordinates back onto the U8 byte axis.

    The quanta are coordinates in [0.0, 0.255]; their U8 byte projection
    (``round(q / LAW)``) is exactly the original byte, so storage is lossless
    in raw byte form. This is the storage half of the pointer map.
    """
    return bytes(_dequantized(quanta))


def _dequantized(quanta: Iterable[float]) -> Iterable[int]:
    """Generator over the U8 byte projection of sub-byte coordinates.

    Memoizes ``round(q / LAW)`` per distinct coordinate so bulk dequantize /
    to_u8 on large tensors avoids recomputing float math for repeated values.
    """
    memo: Dict[float, int] = {}
    for q in quanta:
        b = memo.get(q)
        if b is None:
            b = int(round(float(q) / LAW)) & 0xFF
            memo[q] = b
        yield b


def _coerce_payload(data: Iterable) -> bytes:
    """Normalize a non-bytes payload to raw U8 bytes without corruption.

    Accepts either raw weight bytes (ints 0-255) or sub-byte quanta (floats
    in [0.0, 0.255]); ints are stored byte-identical, floats are projected
    back through ``round(q / LAW)`` exactly as ``to_u8`` does.

    Args:
        data: iterable of ints (raw bytes) or floats (quanta).

    Returns:
        The equivalent raw U8 byte string.
    """
    values = list(data)
    if any(isinstance(v, float) for v in values):
        return to_u8(values)
    return bytes(int(v) & 0xFF for v in values)


def save_gguf(file_payloads: Mapping[str, bytes], filename: str) -> str:
    """Write payloads into a raw x8D container.

    No magic, no JSON, no float bloat, no character metadata -- only raw U8
    byte coordinates behind a minimal name-length address record per payload.
    The quantized weight bytes themselves carry no header, no framing, no
    padding.

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
        for name, data in file_payloads.items():
            if not isinstance(data, (bytes, bytearray)):
                data = _coerce_payload(data)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<Q", len(data)))
            f.write(bytes(data))
    return filename


def load_gguf(filename: str) -> Tuple[Dict[str, bytes], Dict[str, object]]:
    """Read a raw x8D container, returning payloads and metadata.

    Args:
        filename: path to the .x8D container.

    Returns:
        ``(payloads, metadata)`` where payloads maps name -> raw U8 bytes.
    """
    with open(filename, "rb") as f:
        payloads: Dict[str, bytes] = {}
        while True:
            name_len_b = f.read(4)
            if not name_len_b:
                break
            (name_len,) = struct.unpack("<I", name_len_b)
            name = f.read(name_len).decode("utf-8")
            (data_len,) = struct.unpack("<Q", f.read(8))
            payloads[name] = f.read(data_len)
    return payloads, {"law": LAW, "container": "x8D raw U8"}


def mmap_load_gguf(filename: str) -> Tuple[mmap.mmap, Dict[str, object]]:
    """Zero-copy mmap the container: compressed state IS the running state.

    Returns a read-only memory map over the whole file plus metadata. The
    caller can slice payloads out of the map directly -- no decompression
    loop is ever run. There is no header or magic to skip.

    Args:
        filename: path to the .x8D container.

    Returns:
        ``(mapping, metadata)``.
    """
    file_size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    return mapping, {"law": LAW, "container": "x8D raw U8", "size_bytes": file_size}


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
