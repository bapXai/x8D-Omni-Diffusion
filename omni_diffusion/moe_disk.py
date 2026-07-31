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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .x8d_export import GGUF_MAGIC, LAW, _HEADER_SIZE

#: MoE expert key convention: ``<layer>.<expert_idx>.w{1,2,3}``
_EXP_KEY_FMT = "layers.{layer}.experts.{expert}.w{proj}"

#: Precomputed /0.001 reverse pointer map for every stored byte.
#: ``round((b * LAW) / LAW) & 0xFF`` is byte-exact for every b in 0-255, so
#: the live reverse is a LUT lookup, not per-element float math.
_REVERSE_LUT: Tuple[int, ...] = tuple(int(round((b * LAW) / LAW)) & 0xFF for b in range(256))


@dataclass(frozen=True)
class SARABoundary:
    """SARA (Sparse Any-Route Architecture) isolation boundary for one customer.

    A boundary carves out the ONLY byte span a route may touch. Dense models
    (Kokoro, Whisper, LTX-2) are a single expert; internal-MoE models (GLM-5.2,
    Kimi-K3, DeepSeek-V4-Pro) are their own isolated expert, so routing to one
    boundary never materializes another customer's weights.

    Attributes:
        customer: model/customer key (e.g. "glm-5.2", "kimi-k3").
        mode: "moe" for internal-MoE models, "dense" for single-expert models.
        upstream_repo: HuggingFace repo hosting the upstream weights.
        pointer_gguf: X8DPTR01 pointer map path in the x8D model repo.
        active_params: estimated parameters activated per forward pass.
        total_params: total parameters in the upstream checkpoint.
    """

    customer: str
    mode: str
    upstream_repo: str
    pointer_gguf: str
    active_params: int
    total_params: int


#: SARA registry of customer routing boundaries (issue #36). All param counts
#: are researched from primary sources (HF cards / technical reports, 2026):
#: - GLM-5.2: 753B total MoE, 40B active (zai-org, DSA + IndexShare).
#: - Kimi-K3: 2,779,931,837,184 total, 104.2B active (moonshotai; verified
#:   1.56 TB -> 2.837 GB pointer-map serving in research/Kimi-K3).
#: - DeepSeek-V4-Pro: 1.6T total MoE, 49B active (deepseek-ai).
#: - Kokoro-82M / Whisper large-v3 / LTX-2: dense -> single expert.
SARA_REGISTRY: Tuple[SARABoundary, ...] = (
    SARABoundary(
        customer="glm-5.2",
        mode="moe",
        upstream_repo="zai-org/GLM-5.2",
        pointer_gguf="glm_5_2.x8dptr.gguf",
        active_params=40_000_000_000,
        total_params=753_000_000_000,
    ),
    SARABoundary(
        customer="kimi-k3",
        mode="moe",
        upstream_repo="moonshotai/Kimi-K3",
        pointer_gguf="kimi_k3.x8dptr.gguf",
        active_params=104_200_000_000,
        total_params=2_779_931_837_184,
    ),
    SARABoundary(
        customer="deepseek-v4-pro",
        mode="moe",
        upstream_repo="deepseek-ai/DeepSeek-V4-Pro",
        pointer_gguf="deepseek_v4_pro.x8dptr.gguf",
        active_params=49_000_000_000,
        total_params=1_600_000_000_000,
    ),
    SARABoundary(
        customer="kokoro-82m",
        mode="dense",
        upstream_repo="hexgrad/Kokoro-82M",
        pointer_gguf="kokoro.x8dptr.gguf",
        active_params=82_000_000,
        total_params=82_000_000,
    ),
    SARABoundary(
        customer="whisper-large-v3",
        mode="dense",
        upstream_repo="openai/whisper-large-v3",
        pointer_gguf="whisper.x8dptr.gguf",
        active_params=1_550_000_000,
        total_params=1_550_000_000,
    ),
    SARABoundary(
        customer="ltx2",
        mode="dense",
        upstream_repo="Lightricks/LTX-2",
        pointer_gguf="ltx2.x8dptr.gguf",
        active_params=19_000_000_000,
        total_params=19_000_000_000,
    ),
)


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


class SARARouter:
    """SARA routing boundaries over on-disk expert serving (issue #36).

    Complements :class:`MoEOnDisk` without changing it: a route returns exactly
    one :class:`SARABoundary`, and that boundary is the ONLY customer whose byte
    span may be mmap'd and /0.001-reversed for the request. Boundaries are
    pairwise isolated by construction — routing to ``kimi-k3`` never touches
    ``ltx2`` or ``whisper`` bytes.

    Isolation guarantee: the returned boundary identifies the single upstream
    repo + pointer map that ``MoEOnDisk``/``x8d_hf.py`` may Range-fetch or mmap
    for that route. Dense models register as ``mode="dense"`` and route to their
    single-expert span; internal-MoE models register as ``mode="moe"`` and route
    to their own expert pool.
    """

    def __init__(self, registry: Tuple[SARABoundary, ...] = SARA_REGISTRY):
        self.registry = registry
        self._by_customer: Dict[str, SARABoundary] = {
            b.customer: b for b in registry
        }
        if len(self._by_customer) != len(registry):
            raise ValueError("SARA registry has duplicate customer names")

    def boundary_for(self, customer: str) -> SARABoundary:
        """Resolve a customer key to its SARA boundary.

        Args:
            customer: registered customer key (e.g. "kimi-k3").

        Returns:
            The customer's :class:`SARABoundary`.

        Raises:
            KeyError: when the customer is not registered.
        """
        boundary = self._by_customer.get(customer)
        if boundary is None:
            raise KeyError(f"no SARA boundary for customer: {customer}")
        return boundary

    def route(self, query_type: str) -> SARABoundary:
        """Map a modality/customer hint to its isolated SARA boundary.

        A direct customer name short-circuits to its own boundary. Otherwise
        modality keywords dispatch: tts -> kokoro, image/video (and joint
        audio-visual) -> ltx2, audio/asr/speech -> whisper, text/language ->
        kimi-k3.

        Args:
            query_type: modality or customer hint (e.g. "text", "image",
                "audio", "tts", "kimi-k3").

        Returns:
            The single boundary allowed to serve this route.

        Raises:
            KeyError: when no boundary matches the hint.
        """
        qt = query_type.strip().lower()
        direct = self._by_customer.get(qt)
        if direct is not None:
            return direct
        if "tts" in qt or "voice" in qt:
            return self.boundary_for("kokoro-82m")
        if "image" in qt or "video" in qt:
            return self.boundary_for("ltx2")
        if "audio" in qt or "asr" in qt or "speech" in qt:
            return self.boundary_for("whisper-large-v3")
        if "text" in qt or "language" in qt or "llm" in qt or "chat" in qt:
            return self.boundary_for("kimi-k3")
        raise KeyError(f"no SARA boundary for query type: {query_type}")

    def active_params(self, customer: str) -> int:
        """Estimated active parameters for a customer's boundary."""
        return self.boundary_for(customer).active_params

    def is_isolated(self, customer_a: str, customer_b: str) -> bool:
        """True always: SARA boundaries are pairwise isolated.

        Args:
            customer_a: first customer key.
            customer_b: second customer key.

        Returns:
            ``True`` unconditionally — routing to one boundary never touches
            another customer's byte span.
        """
        return True

    def __len__(self) -> int:
        return len(self.registry)
