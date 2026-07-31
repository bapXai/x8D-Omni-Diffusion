# coding=utf-8
"""Zero-copy mmap frame addressing over x8D `.gguf` / `.x8dds.gguf` containers.

Pure Python standard library only. This is the **Colibrì `COLI_MMAP=1` path
re-expressed for the sub-byte container** (issue #41): Colibrì mmaps raw
safetensors shards and lets the kernel page cache serve expert reads; here we
mmap the x8D container (whose U8 coordinates already live at 0.001 scaling) so
the **compressed state IS the running state** — payloads are addressed by an
offset index and sliced straight out of the mapping with zero copying, and the
``/0.001`` inverse operates as a live coordinate pointer map at query time.
"""

from __future__ import annotations

import mmap
import os
import struct
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .x8d_export import GGUF_MAGIC, LAW, X8DHeaderError

#: x8D dataset-stream magic ("X8DDS" + version 0x00 0x01), mirrored from
#: ``x8d_dataset.py`` so the frame walker can validate stream payloads.
X8DDS_MAGIC: bytes = b"X8DDS\x00\x01"

#: DSpark block size: an 8x8 byte block = 64 bytes (mirrored from
#: ``x8d_spec_decode.py``).
BLOCK_SIZE: int = 64

#: ``_HEADER_FMT`` from x8d_export: magic(8) + reserved Q.
_HEADER_SIZE: int = 8 + 8


class X8DMmapError(ValueError):
    """Raised on invalid container layout or impossible offsets."""


def build_payload_index(
    data: bytes, base: int = 0
) -> Dict[str, Tuple[int, int]]:
    """Scan a serialized x8D GGUF body and return ``name -> (offset, length)``.

    Offsets are absolute file positions (``base`` added on top of the header),
    matching the layout written by ``save_gguf``: ``u32 name_len, name,
    u64 data_len, data``. The index lets the mmap reader slice payloads
    straight out of the mapping without reading the whole body.

    Args:
        data: the raw container bytes (header already skipped).
        base: absolute file offset at which ``data`` begins.

    Returns:
        Mapping of payload name to ``(absolute_offset, length)``.

    Raises:
        X8DMmapError: on a truncated or corrupt index record.
    """
    index: Dict[str, Tuple[int, int]] = {}
    pos = 0
    body_len = len(data)
    while pos < body_len:
        if pos + 4 > body_len:
            raise X8DMmapError("truncated name-length field")
        (name_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if pos + name_len > body_len:
            raise X8DMmapError("truncated name bytes")
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        if pos + 8 > body_len:
            raise X8DMmapError("truncated length field")
        (data_len,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        if pos + data_len > body_len:
            raise X8DMmapError(f"payload {name!r} overruns the container")
        index[name] = (base + pos, data_len)
        pos += data_len
    return index


class MappedX8DReader:
    """Zero-copy read-only access to an x8D GGUF container via ``mmap``.

    Mirrors Colibrì's ``COLI_MMAP=1``: the file is mapped with
    ``access=ACCESS_READ`` so the kernel page cache serves reads, payloads are
    sliced out of the mapping in place (no decompression loop), and the
    ``/0.001`` inverse is applied only when a caller asks for a float view.

    Args:
        filename: path to an x8D GGUF container (``.gguf`` or ``.x8dds.gguf``).

    Attributes:
        filename: the mapped path.
        size_bytes: total mapped size.
        index: ``name -> (offset, length)`` payload offset index.
        law: the 0.001 sub-byte scaling law in effect.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.size_bytes = os.path.getsize(filename)
        self._fd = os.open(filename, os.O_RDONLY)
        try:
            self._mapping: Optional[mmap.mmap] = mmap.mmap(
                self._fd, self.size_bytes, access=mmap.ACCESS_READ
            )
        except BaseException:
            os.close(self._fd)
            raise
        if self._mapping[: len(GGUF_MAGIC)] != GGUF_MAGIC:
            self.close()
            raise X8DHeaderError(
                f"Not a valid x8D GGUF container (magic "
                f"{self._mapping[: len(GGUF_MAGIC)]!r})"
            )
        self.index: Dict[str, Tuple[int, int]] = build_payload_index(
            bytes(self._mapping[_HEADER_SIZE:]), base=_HEADER_SIZE
        )
        self.law: float = LAW

    # -- payload addressing -------------------------------------------------

    def names(self) -> List[str]:
        """Return the payload names present in the container, in order."""
        return list(self.index)

    def offsets(self, name: str) -> Tuple[int, int]:
        """Return ``(offset, length)`` for a payload, as file positions.

        Args:
            name: payload name.

        Returns:
            Absolute offset into the mapped file and the payload length.

        Raises:
            KeyError: unknown payload name.
        """
        return self.index[name]

    def load(self, name: str) -> bytes:
        """Slice a payload's raw U8 coordinates out of the mapping.

        The returned bytes are a copy of the underlying map slice; for a
        non-copying window use :meth:`view`.

        Args:
            name: payload name.

        Returns:
            The payload's raw byte string (lossless U8 coordinates).
        """
        offset, length = self.index[name]
        return self._mapping[offset : offset + length]  # type: ignore[index]

    def view(self, name: str) -> memoryview:
        """Return a non-copying ``memoryview`` window over a payload.

        This is the zero-copy serving path: the caller reads bytes directly
        out of the kernel page cache with no intermediate buffer.

        Args:
            name: payload name.

        Returns:
            A ``memoryview`` over the mapped payload.
        """
        offset, length = self.index[name]
        return memoryview(self._mapping)[offset : offset + length]

    def reverse(self, name: str) -> List[float]:
        """Live ``/0.001`` coordinate pointer map over a payload.

        Applies the inverse sub-byte law on demand (query-time only), so the
        compressed state stays the running state on disk and floats are only
        materialized for the specific payload being decoded.

        Args:
            name: payload name.

        Returns:
            List of floats ``byte * 0.001`` for every byte in the payload.
        """
        inv = 1.0 / self.law
        return [b * inv for b in self.load(name)]

    def slice_at(self, offset: int, length: int) -> bytes:
        """Slice arbitrary bytes out of the mapping by file position.

        Args:
            offset: absolute file offset.
            length: number of bytes to read.

        Returns:
            Raw bytes from the mapped file.

        Raises:
            X8DMmapError: if the range falls outside the mapping.
        """
        if offset < 0 or offset + length > self.size_bytes:
            raise X8DMmapError(
                f"range [{offset}, {offset + length}) outside mapping "
                f"of {self.size_bytes} bytes"
            )
        return self._mapping[offset : offset + length]  # type: ignore[index]

    # -- x8D dataset-stream frame walking ----------------------------------

    def stream_header(self, name: str) -> Tuple[int, int]:
        """Parse an ``X8DDS`` stream payload header.

        Args:
            name: payload name of an ``.x8dds`` stream.

        Returns:
            ``(row_count, data_offset)`` where ``data_offset`` is the absolute
            file position of the first stream byte after the magic+row-count
            header (i.e. the first DSpark 8x8 frame boundary).

        Raises:
            X8DMmapError: if the payload is not an ``X8DDS`` stream.
        """
        payload_offset, _ = self.index[name]
        header = self._mapping[payload_offset : payload_offset + len(X8DDS_MAGIC)]
        if header != X8DDS_MAGIC:
            raise X8DMmapError(f"payload {name!r} is not an X8DDS stream")
        (row_count,) = struct.unpack_from(
            "<Q", self._mapping, payload_offset + len(X8DDS_MAGIC)
        )
        data_offset = payload_offset + len(X8DDS_MAGIC) + 8
        return row_count, data_offset

    def frames(
        self, name: str, block_size: int = BLOCK_SIZE
    ) -> Iterator[Tuple[int, memoryview]]:
        """Walk an ``X8DDS`` stream as in-place 8x8 DSpark block frames.

        Yields ``(absolute_offset, memoryview)`` pairs over the mapped data —
        no copies. This is the k-parallel block generator's zero-copy input
        path: blocks are addressed straight out of the page cache.

        Args:
            name: payload name of an ``.x8dds`` stream.
            block_size: frame width in bytes (default 64 = 8x8).

        Yields:
            ``(offset, memoryview)`` per aligned block frame.
        """
        _, data_offset = self.stream_header(name)
        start = data_offset
        while start + block_size <= self.size_bytes:
            yield start, memoryview(self._mapping)[start : start + block_size]
            start += block_size

    def close(self) -> None:
        """Release the memory map and file descriptor."""
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        os.close(self._fd)

    def __enter__(self) -> "MappedX8DReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def reverse_bytes(coords: Iterable[int], law: float = LAW) -> List[float]:
    """Apply the live ``/0.001`` inverse to raw U8 coordinates.

    Args:
        coords: raw byte coordinates (0-255) as read from a mapped payload.
        law: the sub-byte scaling law (default 0.001).

    Returns:
        The coordinate-pointer-map float view of the bytes.
    """
    inv = 1.0 / law
    return [b * inv for b in coords]
