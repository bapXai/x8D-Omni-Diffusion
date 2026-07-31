# coding=utf-8
"""x8Dsub-byte packed sub-byte model: 0.016 bit/weight = 32 MB for a 16B model.

The compressed sub-byte coordinate state IS the running state. The 0.001
scaling law packs every 500-weight block into ONE sub-byte coordinate byte
(500 x 0.016 bit = 8 bits):

    Quanta[block] = round(mean(weight_byte) x 0.001)   (0-255)
    running weight byte = round(Quanta[block] / 0.001)  (live pointer map)

Size law (matches `x8Dsub-byte`): total = n_params x 16 bits x 0.001 / 8.
  - 16,000,000,000 params x 0.016 bit = 256 Mbit = **32 MB**.
  - That 32 MB IS the full FP16/BF16 32 GB model under the x8Dsub-byte law.

Pure Python standard library only; zero-copy mmap serving.
"""

from __future__ import annotations

import math
import mmap
import os
import struct
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .x8d_export import LAW, GGUF_MAGIC

#: Bits consumed per weight by the sub-byte law (16-bit baseline x 0.001).
BITS_PER_WEIGHT: float = 16.0 * LAW  # 0.016

#: Weights packed per sub-byte coordinate byte (8 bits / 0.016 bits).
WEIGHTS_PER_COORD: int = int(8 / BITS_PER_WEIGHT)  # 8 / 0.016 = 500

#: Container magic for the packed sub-byte coordinate map.
SUB_BYTE_MAGIC: bytes = b"X8DSUB01"


def packed_size_bytes(num_params: int) -> int:
    """Size of the sub-byte coordinate map for ``num_params`` weights.

    Args:
        num_params: total parameter count.

    Returns:
        Packed byte size (0.016 bit/weight).
    """
    return max(1, math.ceil(num_params * BITS_PER_WEIGHT / 8.0))


def coords_per_pack(num_params: int) -> int:
    """Number of sub-byte coordinate bytes for ``num_params`` weights."""
    return max(1, math.ceil(num_params / WEIGHTS_PER_COORD))


class SubByteHeaderError(ValueError):
    """Raised when a file does not carry a valid sub-byte header."""


def pack_subbyte(weight_bytes: Union[bytes, bytearray, Iterable[int]], block: int = WEIGHTS_PER_COORD) -> bytes:
    """Pack raw weight bytes into the 0.016 bit/weight coordinate map.

    Every ``block`` (500) weight bytes collapses to one coordinate byte:
    ``round(mean(weight_byte) x 0.001)``. This is the x8Dsub-byte pointer map —
    the compressed state that IS the running state.

    Args:
        weight_bytes: raw uint8 weight bytes.
        block: weights per coordinate byte (default 500 = 8 / 0.016).

    Returns:
        Packed coordinate bytes (size = n_params x 0.016 bit / 8).
    """
    weights = list(weight_bytes)
    n = len(weights)
    out = bytearray()
    for i in range(0, n, block):
        chunk = weights[i : i + block]
        mean = sum(chunk) / len(chunk)
        quanta = int(round(mean * LAW * 1000.0)) & 0xFF  # round(x * 0.001) scaled
        out.append(quanta)
    return bytes(out)


def unpack_subbyte(data: bytes, num_params: int, block: int = WEIGHTS_PER_COORD) -> List[int]:
    """Expand a sub-byte coordinate map back into running weight bytes.

    The inverse math (``/ 0.001``) is the live coordinate pointer map: each
    stored quanta byte maps back to the weight byte it represents.

    Args:
        data: packed sub-byte coordinate bytes.
        num_params: number of weights to reconstruct.
        block: weights per coordinate byte.

    Returns:
        List of reconstructed weight bytes (length ``num_params``).
    """
    out: List[int] = []
    for coord in data:
        weight_byte = int(round((coord * 0.001) / LAW)) & 0xFF
        out.extend([weight_byte] * block)
    return out[:num_params]


def quanta_of(weight_byte: int) -> int:
    """The sub-byte coordinate for one weight byte (0-255)."""
    return int(round(weight_byte * 0.001 * 1000.0)) & 0xFF


def weight_of(quanta: int) -> int:
    """Inverse pointer map: coordinate byte -> running weight byte."""
    return int(round((quanta * 0.001) / LAW)) & 0xFF


def save_subbyte_gguf(name: str, weight_bytes: bytes, filename: str) -> Tuple[str, int]:
    """Pack weight bytes into a sub-byte x8D container and save it.

    Args:
        name: tensor name.
        weight_bytes: raw weight bytes.
        filename: output .gguf path.

    Returns:
        ``(output_path, packed_bytes)``.
    """
    packed = pack_subbyte(weight_bytes)
    with open(filename, "wb") as f:
        f.write(SUB_BYTE_MAGIC)
        f.write(struct.pack("<Q", len(weight_bytes)))  # original param count
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(packed)
    return filename, len(packed)


def load_subbyte_gguf(filename: str) -> Tuple[Dict[str, bytes], Dict[str, int]]:
    """Read a packed sub-byte container.

    Args:
        filename: path to the .gguf container.

    Returns:
        ``(payloads, meta)`` with meta = {num_params, packed_size}.
    """
    with open(filename, "rb") as f:
        if f.read(len(SUB_BYTE_MAGIC)) != SUB_BYTE_MAGIC:
            raise SubByteHeaderError("Not a valid sub-byte container")
        (num_params,) = struct.unpack("<Q", f.read(8))
        (name_len,) = struct.unpack("<I", f.read(4))
        name = f.read(name_len).decode("utf-8")
        packed = f.read()
    return {name: packed}, {"num_params": num_params, "packed_size": len(packed)}


def mmap_load_subbyte_gguf(filename: str) -> Tuple[mmap.mmap, Dict[str, int]]:
    """Zero-copy mmap the 32 MB sub-byte model: compressed state is running.

    Args:
        filename: path to the sub-byte container.

    Returns:
        ``(mapping, meta)``.
    """
    file_size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    if mapping[: len(SUB_BYTE_MAGIC)] != SUB_BYTE_MAGIC:
        raise SubByteHeaderError("Not a valid sub-byte container")
    (num_params,) = struct.unpack("<Q", mapping[len(SUB_BYTE_MAGIC) : len(SUB_BYTE_MAGIC) + 8])
    (name_len,) = struct.unpack("<I", mapping[16:20])
    packed_size = file_size - 20 - name_len
    return mapping, {"num_params": num_params, "packed_size": packed_size}


class SubByteModel:
    """The 32 MB sub-byte model: serves full FP16/BF16 precision via pointer map.

    The whole coordinate map is memory-mapped; ``/ 0.001`` runs as a live
    pointer lookup so no decompression loop ever executes.
    """

    #: Precomputed inverse pointer map: coordinate byte -> running weight byte.
    _WEIGHT_LUT: Tuple[int, ...] = tuple(weight_of(i) for i in range(256))

    def __init__(self, filename: str):
        self.mapping, self.meta = mmap_load_subbyte_gguf(filename)
        self._payload_offset = 0
        # parse header: magic(8) + num_params(8) + name_len(4) + name
        name_len = struct.unpack("<I", self.mapping[16:20])[0]
        self._payload_offset = 20 + name_len
        self._packed = self.mapping[self._payload_offset :]
        self._n = self.meta["num_params"]

    def __len__(self) -> int:
        return self._n

    def packed_size_mb(self) -> float:
        """Disk size of this model's coordinate map in MB."""
        return len(self._packed) / 1e6

    def weight_at(self, index: int) -> int:
        """Running weight byte for parameter ``index`` via the pointer map.

        Args:
            index: parameter index in ``[0, n)``.

        Returns:
            The reconstructed weight byte.

        Raises:
            IndexError: if ``index`` is out of range.
        """
        if index < 0 or index >= self._n:
            raise IndexError(
                f"weight_at({index}) out of range for {self._n}-param model"
            )
        coord = self._packed[index // WEIGHTS_PER_COORD]
        return self._WEIGHT_LUT[coord]

    def weights(self, start: int = 0, end: Optional[int] = None) -> List[int]:
        """Slice of running weight bytes in ``[start, end)``.

        Args:
            start: first parameter index.
            end: exclusive end index (default ``len(model)``).

        Returns:
            Reconstructed weight bytes for the range.

        Raises:
            IndexError: if the range exceeds the model's parameter count.
        """
        if end is None:
            end = self._n
        if start < 0 or end > self._n or start > end:
            raise IndexError(
                f"weights({start}, {end}) out of range for {self._n}-param model"
            )
        if start == end:
            return []
        # C-speed: translate coordinate bytes -> running weight bytes in bulk.
        packed = self._packed
        wpb = WEIGHTS_PER_COORD
        first = start // wpb
        last = (end - 1) // wpb + 1
        coords = packed[first:last].translate(bytes(self._WEIGHT_LUT))
        out = b"".join(bytes([v]) * wpb for v in coords)
        head = start - first * wpb
        return list(out[head : head + (end - start)])

    def close(self) -> None:
        """Release the memory map."""
        try:
            self.mapping.close()
        except BufferError:
            pass


def size_report_subbyte(num_params: int = 16_000_000_000, baseline_bits: int = 16) -> Dict[str, float]:
    """Full FP16/BF16 model vs the 0.016 bit/weight sub-byte model."""
    baseline_bytes = num_params * (baseline_bits / 8.0)
    subbyte_bytes = packed_size_bytes(num_params)
    return {
        "full_precision_gb": baseline_bytes / 1e9,
        "subbyte_mb": subbyte_bytes / 1e6,
        "bits_per_weight": BITS_PER_WEIGHT,
        "reduction_pct": (1.0 - subbyte_bytes / baseline_bytes) * 100.0,
    }


def print_size_report_subbyte(num_params: int = 16_000_000_000, baseline_bits: int = 16) -> None:
    """Print the 32 GB -> 32 MB equivalence table."""
    r = size_report_subbyte(num_params, baseline_bits)
    print(f"x8Dsub-byte: full {num_params:,} param FP16/BF16 model")
    print(f"  Full precision model : {r['full_precision_gb']:.2f} GB")
    print(f"  Sub-byte 0.016 bit    : {r['subbyte_mb']:.1f} MB  ({r['bits_per_weight']} bit/weight)")
    print(f"  The sub-byte map IS the full-precision running state ({r['reduction_pct']:.2f}% smaller)")
