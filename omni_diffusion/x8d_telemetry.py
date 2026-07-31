# coding=utf-8
"""I/O + memory telemetry per 8x8 DSpark block (Colibrì ``telemetry.h`` port).

Pure Python standard library only. Mirrors the operational metrics Colibrì
tracks in ``c/telemetry.h`` and ``c/colibri.c`` (issue #41):

- ``g_prof_io`` atomic byte counter  -> :meth:`Telemetry.record_io`
- ``hit_pin`` / ``hit_ecache`` split -> :meth:`Telemetry.record_hit`
- ``rss_gb()`` (``getrusage``)      -> :attr:`Telemetry.rss_bytes`
- per-turn stats line                -> :meth:`Telemetry.dashboard`

Every counter is per-8x8-block aware so a DSpark k-parallel batch can report
bytes read, blocks decoded, elapsed time and RSS all the way down to one block.
"""

from __future__ import annotations

import os
import resource
import time
from threading import Lock
from typing import Dict, List, Optional, Tuple

#: DSpark block size (8x8 = 64 bytes) mirrored from x8d_spec_decode.
BLOCK_SIZE: int = 64


def _rss_bytes() -> int:
    """Return current process resident set size in bytes (``ru_maxrss``)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class Telemetry:
    """Thread-safe operational metrics collector for the DSpark decode loop.

    Tracks I/O bytes read from mapped containers, per-8x8-block timing, RSS,
    and the pinned-vs-LRU expert-hit split — the same signals Colibrì exposes
    on its dashboard, expressed in x8D terms (bytes not tokens).

    Args:
        label: short name for the dashboard line (e.g. ``"text"`` modality).
    """

    def __init__(self, label: str = "x8d") -> None:
        self.label: str = label
        self._lock: Lock = Lock()
        self._io_bytes: int = 0
        self._fault_bytes: int = 0
        self._blocks: int = 0
        self._block_us: List[int] = []
        self._hit_pin: int = 0
        self._hit_lru: int = 0
        self._start_ns: int = time.perf_counter_ns()
        self._closed: bool = False

    # -- I/O accounting -----------------------------------------------------

    def record_io(self, n_bytes: int) -> None:
        """Count bytes read from the mapped container (Colibrì ``g_prof_io``).

        Args:
            n_bytes: number of bytes consumed by this step.
        """
        if n_bytes > 0:
            with self._lock:
                self._io_bytes += n_bytes

    def record_fault(self, n_bytes: int) -> None:
        """Count bytes that were page-faulted into RAM (first-touch reads).

        Args:
            n_bytes: bytes that required a real disk read (not page-cache hit).
        """
        if n_bytes > 0:
            with self._lock:
                self._fault_bytes += n_bytes

    def record_hit(self, tier: str) -> None:
        """Record an expert hit by residency tier.

        Args:
            tier: ``"pin"`` (never-evicted hot-store) or ``"lru"`` (per-layer
                LRU cache). Mirrors Colibrì's ``hit_pin`` / ``hit_ecache``.
        """
        with self._lock:
            if tier == "pin":
                self._hit_pin += 1
            elif tier == "lru":
                self._hit_lru += 1
            else:
                raise ValueError(f"unknown tier {tier!r} (pin|lru)")

    # -- per-block timing ---------------------------------------------------

    def begin_block(self) -> None:
        """Mark the start of a new 8x8 DSpark block decode."""
        with self._lock:
            self._blocks += 1
            self._block_start_ns: int = time.perf_counter_ns()

    def end_block(self) -> int:
        """Close the current block and return its elapsed microseconds.

        Returns:
            Microseconds spent decoding the block opened by :meth:`begin_block`.
        """
        elapsed_us = int((time.perf_counter_ns() - self._block_start_ns) // 1000)
        with self._lock:
            self._block_us.append(elapsed_us)
        return elapsed_us

    # -- snapshots ----------------------------------------------------------

    @property
    def io_bytes(self) -> int:
        """Total I/O bytes counted via :meth:`record_io`."""
        with self._lock:
            return self._io_bytes

    @property
    def fault_bytes(self) -> int:
        """Total page-fault bytes counted via :meth:`record_fault`."""
        with self._lock:
            return self._fault_bytes

    @property
    def block_count(self) -> int:
        """Number of 8x8 blocks opened via :meth:`begin_block`."""
        with self._lock:
            return self._blocks

    @property
    def rss_bytes(self) -> int:
        """Current process RSS in bytes (``getrusage`` ``ru_maxrss``)."""
        return _rss_bytes()

    def snapshot(self) -> Dict[str, object]:
        """Return a JSON-serializable snapshot of all counters.

        Returns:
            Dict with label, io_bytes, fault_bytes, blocks, mean/max block
            microseconds, hits (pin/lru), rss_mb and elapsed_seconds.
        """
        with self._lock:
            total_us = sum(self._block_us)
            n = len(self._block_us)
            io_bytes = self._io_bytes
            fault_bytes = self._fault_bytes
            blocks = self._blocks
            hit_pin = self._hit_pin
            hit_lru = self._hit_lru
            elapsed_s = round((time.perf_counter_ns() - self._start_ns) / 1e9, 3)
        return {
            "label": self.label,
            "io_bytes": io_bytes,
            "fault_bytes": fault_bytes,
            "blocks": blocks,
            "block_us_mean": (total_us // n) if n else 0,
            "block_us_max": max(self._block_us) if n else 0,
            "hits_pin": hit_pin,
            "hits_lru": hit_lru,
            "rss_mb": round(self.rss_bytes / (1024.0 * 1024.0), 2),
            "elapsed_s": elapsed_s,
        }

    @property
    def blocks_count(self) -> int:
        """Alias of :attr:`block_count` (mirrors Colibrì dashboard wording)."""
        return self.block_count

    def dashboard(self) -> str:
        """Render a Colibrì-style per-turn stats line.

        Format: ``[label] blk=N io=MB(GB/s) fault=MB rss=GB hit_pin=K hit_lru=K``
        where the GB/s figure is bytes counted over the elapsed window.

        Returns:
            Single-line telemetry string for logs or a dashboard.
        """
        snap = self.snapshot()
        elapsed = float(snap["elapsed_s"])
        gbps = (int(snap["io_bytes"]) / 1e9 / elapsed) if elapsed > 0 else 0.0
        rss_gb = float(snap["rss_mb"]) / 1024.0
        return (
            f"[{self.label}] blk={snap['blocks']} "
            f"io={int(snap['io_bytes']) / 1e6:.1f}MB({gbps:.2f}GB/s) "
            f"fault={int(snap['fault_bytes']) / 1e6:.1f}MB "
            f"rss={rss_gb:.2f}GB "
            f"hit_pin={snap['hits_pin']} hit_lru={snap['hits_lru']}"
        )

    def close(self) -> None:
        """Finalize the collector (idempotent; future records are ignored)."""
        self._closed = True


def _selftest() -> int:
    """Small dependency-free sanity check of the telemetry loop.

    Returns:
        Number of assertions that failed (0 = healthy).
    """
    failures = 0
    t = Telemetry(label="selftest")
    t.record_io(4096)
    t.record_fault(1024)
    t.record_hit("pin")
    t.record_hit("lru")
    t.begin_block()
    t.end_block()
    if t.io_bytes != 4096:
        failures += 1
    if t.fault_bytes != 1024:
        failures += 1
    if t.blocks_count != 1:
        failures += 1
    if t._hit_pin != 1 or t._hit_lru != 1:
        failures += 1
    if t.rss_bytes <= 0:
        failures += 1
    if "blk=1" not in t.dashboard():
        failures += 1
    return failures
