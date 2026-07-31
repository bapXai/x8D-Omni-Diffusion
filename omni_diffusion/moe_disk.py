# coding=utf-8
"""On-disk MoE expert serving from an x8D .gguf container (issue #9).

The serving law: **compressed state IS the running state.** Weights are NEVER
loaded into RAM. An x8D container is memory-mapped, and at query time only the
specific MoE expert needed for the current token is materialized:

    weight_byte = round(quanta / 0.001)      # live /0.001 reverse
    expert_mat = reshape(weight_byte, (in, out))

Pure Python stdlib. The container layout is the x8D GGUF format from
``x8d_export.py``: ``<u64 name_len><name><u64 data_len><raw U8 bytes>`` per
tensor, behind the X8DGGUF1 magic.
"""

from __future__ import annotations

import mmap
import os
import struct
from typing import Dict, List, Optional, Tuple

from .x8d_export import GGUF_MAGIC, LAW, _HEADER_SIZE

#: MoE expert key convention: ``<layer>.<expert_idx>.w{1,2,3}``
_EXP_KEY_FMT = "layers.{layer}.experts.{expert}.w{proj}"

#: Precomputed /0.001 reverse pointer map for every stored byte.
#: ``round((b * LAW) / LAW) & 0xFF`` is byte-exact for every b in 0-255, so
#: the live reverse is a LUT lookup, not per-element float math.
_REVERSE_LUT: Tuple[int, ...] = tuple(int(round((b * LAW) / LAW)) & 0xFF for b in range(256))


def _proj_numeric(proj: str) -> str:
    """Accept 'w1' or '1' and normalize to '1'."""
    return proj[1:] if proj.startswith("w") else proj


class X8DGgufReader:
    """Zero-copy reader over an x8D GGUF container.

    Holds a single memory map; tensor payloads are sliced out on demand.
    """

    def __init__(self, path: str):
        self.path = path
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        try:
            self._mmap = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        if self._mmap[: len(GGUF_MAGIC)] != GGUF_MAGIC:
            raise ValueError(f"Not a valid x8D GGUF container: {path}")
        self._cursor = _HEADER_SIZE
        self._tensors: Dict[str, Tuple[int, int]] = {}  # name -> (off, len)
        self._scan()

    def _scan(self) -> None:
        """Build the tensor index by walking the container once."""
        m = self._mmap
        pos = self._cursor
        while pos < len(m):
            (name_len,) = struct.unpack_from("<I", m, pos)
            pos += 4
            name = bytes(m[pos : pos + name_len]).decode("utf-8")
            pos += name_len
            (data_len,) = struct.unpack_from("<Q", m, pos)
            pos += 8
            self._tensors[name] = (pos, data_len)
            pos += data_len
        self._cursor = pos

    def contains(self, name: str) -> bool:
        return name in self._tensors

    def tensor_bytes(self, name: str) -> Optional[bytes]:
        """Slice one tensor's raw U8 bytes out of the map (no copy of others)."""
        entry = self._tensors.get(name)
        if entry is None:
            return None
        off, length = entry
        return bytes(self._mmap[off : off + length])

    def tensor_size(self, name: str) -> int:
        entry = self._tensors.get(name)
        return entry[1] if entry else 0

    def names(self) -> List[str]:
        return list(self._tensors.keys())

    def close(self) -> None:
        try:
            self._mmap.close()
        except BufferError:
            pass


class MoEOnDisk:
    """On-disk MoE router: resolves and serves a single expert at query time.

    Only the requested expert's byte span is touched. The ``/0.001`` inverse
    math is applied live on that slice; nothing else is materialized.
    """

    def __init__(self, gguf_path: str):
        self.reader = X8DGgufReader(gguf_path)

    # -- expert addressing -------------------------------------------------
    def expert_key(self, layer: int, expert: int, proj: str = "w1") -> str:
        """Canonical container key for an expert weight matrix."""
        return _EXP_KEY_FMT.format(layer=layer, expert=expert, proj=_proj_numeric(proj))

    def expert_available(self, layer: int, expert: int, proj: str = "w1") -> bool:
        return self.reader.contains(self.expert_key(layer, expert, proj))

    # -- query-time materialization ----------------------------------------
    def load_expert(
        self, layer: int, expert: int, proj: str = "w1", shape: Optional[Tuple[int, int]] = None
    ) -> List[int]:
        """Return the running weight bytes for one expert (live /0.001).

        Args:
            layer: MoE layer index.
            expert: expert index within the layer.
            proj: projection name (w1/w2/w3).
            shape: optional (in, out) to validate byte count.

        Returns:
            List of dequantized weight bytes (0-255) for this expert only.
        """
        key = self.expert_key(layer, expert, proj)
        data = self.reader.tensor_bytes(key)
        if data is None:
            raise KeyError(f"expert not on disk: {key}")
        if shape is not None:
            in_d, out_d = shape
            if in_d is not None:
                expected = in_d * (out_d or 1)
                if len(data) % (in_d or 1) != 0:
                    raise ValueError(f"{key}: {len(data)} bytes not divisible by in={in_d}")
                if out_d is not None and len(data) != expected:
                    raise ValueError(
                        f"{key}: stored {len(data)} bytes, shape {shape} needs {expected}"
                    )
        # the ONLY place /0.001 runs: live reverse on this expert's span
        lut = _REVERSE_LUT
        return [lut[b] for b in data]

    def matmul_fp32(self, layer: int, expert: int, hidden: List[float], proj: str = "w1") -> List[float]:
        """CPU-only forward: expert weight (bytes -> [-1,1] scale) @ hidden.

        Demonstrates a real query-time use: route to ONE expert, reverse the
        /0.001 law live, multiply. No model is ever resident in RAM.

        Args:
            layer: MoE layer index.
            expert: expert index.
            hidden: input activations (length = in_features).
            proj: projection name.

        Returns:
            Output vector of length = out_features.
        """
        in_features = len(hidden)
        w = self.load_expert(layer, expert, proj, shape=(in_features, None))  # type: ignore[arg-type]
        out_features = len(w) // in_features
        # bytes are normalized: b/128 - 1 in [-1,1] as the running weight
        out = [0.0] * out_features
        for o in range(out_features):
            acc = 0.0
            base = o * in_features
            for i in range(in_features):
                acc += ((w[base + i] / 128.0) - 1.0) * hidden[i]
            out[o] = acc
        return out

    def size_mb(self) -> float:
        """Disk size of the mmap'd container in MB."""
        return os.path.getsize(self.reader.path) / 1e6

    def disk_size_mb(self) -> float:
        return os.path.getsize(self.path if hasattr(self, "path") else self.reader.path) / 1e6

    def close(self) -> None:
        self.reader.close()
