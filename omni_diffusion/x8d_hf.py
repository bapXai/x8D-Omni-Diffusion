# coding=utf-8
"""HF repo -> x8D .gguf converter (issue #9).

Pure Python stdlib. Reads a HuggingFace safetensors shard (from a local file
or a remote Range-fetched span) and packs its tensors into an x8D GGUF
container using the 0.001 sub-byte law.

Design notes
------------
- The x8D container stores raw U8 byte coordinates (X8DGGUF1 magic) behind
  per-tensor name records: ``<u32 name_len><name><u64 data_len><bytes>``.
- Conversion is per-tensor and streaming-friendly: a caller can ask for ONE
  tensor (e.g. a specific MoE expert) and only those bytes are read from disk
  (mmap) or fetched over HTTP Range — the full model is never loaded into RAM.
- ``Quanta[i] = weight_byte[i] * 0.001`` is applied lazily: stored bytes ARE
  the quanta; the ``/0.001`` reverse runs live at query time.

The parser handles the safetensors format: 8-byte LE header length, JSON
header with ``{"tensor_name": {"dtype", "shape", "data_offsets"}}``, then raw
data. No safetensors package required.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Dict, List, Optional, Tuple

from .x8d_export import GGUF_MAGIC, LAW, X8D_HEADER, save_gguf

#: safetensors header: <u64: header_len><json header><data...>
_ST_HEADER = struct.Struct("<Q")


class SafetensorsShard:
    """On-disk reader for a .safetensors shard (no RAM load of data)."""

    def __init__(self, path: str):
        self.path = path
        self._mmap: Optional[mmap.mmap] = None
        self._index: Optional[Dict[str, Dict[str, object]]] = None
        self._data_start = 0

    def _open(self) -> mmap.mmap:
        if self._mmap is None:
            size = os.path.getsize(self.path)
            fd = os.open(self.path, os.O_RDONLY)
            try:
                self._mmap = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
            (header_len,) = _ST_HEADER.unpack_from(self._mmap, 0)
            self._data_start = 8 + header_len
            self._index = json.loads(bytes(self._mmap[8 : 8 + header_len]).decode("utf-8"))
        return self._mmap

    @property
    def index(self) -> Dict[str, Dict[str, object]]:
        """Tensor name -> {dtype, shape, data_offsets[begin,end]}."""
        self._open()
        return self._index  # type: ignore[return-value]

    @property
    def data_start(self) -> int:
        self._open()
        return self._data_start

    def tensor_offsets(self, name: str) -> Optional[Tuple[int, int]]:
        """Absolute file offsets [begin, end) for a tensor, or None."""
        spec = self.index.get(name)
        if spec is None:
            return None
        begin, end = spec["data_offsets"]  # type: ignore[misc]
        ds = self.data_start
        return ds + int(begin), ds + int(end)

    def read_tensor(self, name: str) -> Optional[bytes]:
        """Read one tensor's raw bytes from the mmap'd shard.

        Args:
            name: tensor name (e.g. an MoE expert weight).

        Returns:
            Raw weight bytes, or None if absent.
        """
        offs = self.tensor_offsets(name)
        if offs is None:
            return None
        return bytes(self._mmap[offs[0] : offs[1]])  # type: ignore[index]

    def read_span(self, begin: int, end: int) -> bytes:
        """Read an absolute byte span (for remote Range fetches)."""
        mm = self._open()
        return bytes(mm[begin:end])

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass
            self._mmap = None


def convert_shard_to_gguf(
    shard_path: str,
    output_path: str,
    tensor_names: Optional[List[str]] = None,
    spec_decode: bool = False,
) -> Tuple[str, Dict[str, int]]:
    """Convert safetensors shard tensors into an x8D GGUF container.

    Args:
        shard_path: local path to a .safetensors shard.
        output_path: output .gguf path.
        tensor_names: optional subset; when None, all tensors are converted.
        spec_decode: run the DSpark 8x8 speculative quantizer instead of raw
            byte coordinates.

    Returns:
        ``(output_path, stats)`` with stats = {tensors, bytes_in, bytes_out}.
    """
    shard = SafetensorsShard(shard_path)
    try:
        names = tensor_names or list(shard.index.keys())
        payloads: Dict[str, bytes] = {}
        bytes_in = 0
        for name in names:
            data = shard.read_tensor(name)
            if data is None:
                continue
            bytes_in += len(data)
            if spec_decode:
                from .x8d_spec_decode import speculative_quantize, to_u8

                quanta, _ = speculative_quantize(data)
                payloads[name] = to_u8(quanta)
            else:
                # stored bytes ARE the quanta (0.001 coordinate map)
                payloads[name] = data
        save_gguf(payloads, output_path)
        return output_path, {
            "tensors": len(payloads),
            "bytes_in": bytes_in,
            "bytes_out": sum(len(v) for v in payloads.values()),
        }
    finally:
        shard.close()


def convert_remote_span(
    url: str,
    token: Optional[str],
    shard_header_len: int,
    begin: int,
    end: int,
) -> bytes:
    """Fetch an absolute byte span of a remote safetensors shard (Range).

    Pure stdlib ``urllib`` — lets us grab ONE tensor (e.g. a single MoE
    expert) from a 2.7 TB model without downloading the whole shard.

    Args:
        url: resolve URL for the shard file.
        token: HF access token (Bearer) or None.
        shard_header_len: length of the JSON header (needs one range fetch of
            bytes 0..8+header for parsing; pass the already-read header).
        begin: absolute byte offset of the tensor start.
        end: absolute byte offset of the tensor end.

    Returns:
        The requested byte span.
    """
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={begin}-{end - 1}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return resp.read()


def parse_safetensors_header(header_bytes: bytes) -> Dict[str, Dict[str, object]]:
    """Parse a safetensors JSON header (from the first 8+N bytes)."""
    return json.loads(header_bytes.decode("utf-8"))
