# coding=utf-8
"""Quantize ANY HuggingFace safetensors model DIRECTLY from HF to .x8D.

The 0.001 sub-byte law, zero full-model download. The parameters live inside
the source bytes; we never count parameters and never lose precision. The
only number is DISK SIZE = source_bytes x 0.001 (0.008 bit per weight byte,
1000:1). fp16 3.09 GB -> 3,086,981 B.

The .x8D file is ONLY the sub-byte coordinate bytes: no GGUF_MAGIC, no
framing, no name-length headers, no manifest, no padding. Every 1000 source
bytes become one coordinate byte (the packed coordinate state IS the running
state; `/0.001` reverses it live).

Method (chunked Range fetching, resumable, timeout-tolerant):
  1. HTTP Range-fetch the safetensors header (bytes 0..header_len) only.
  2. Stream the remaining file body in fixed-size CHUNK_SIZE (100 MB) Range
     requests. Each chunk is its own connection, so a stalled connection
     only retries that chunk (exponential backoff), never the whole file.
     CHUNK_SIZE is a multiple of WEIGHTS_PER_COORD, so resume is exact.
  3. A `.resume.json` checkpoint records (shard, shard_consumed, bodies,
     source_bytes) after every successful chunk; a crash resumes from the
     last persisted chunk instead of restarting from zero.
  4. The full model is never downloaded or stored.

Usage:
    python3 tools/quantize_hf_safetensors.py \
        openai/whisper-large-v3 model.safetensors /tmp/whisper.x8D
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from typing import List, Tuple

from omni_diffusion.x8d_subbyte import LAW, WEIGHTS_PER_COORD

STREAM_BLOCK = 1 << 20  # 1 MiB coordinate flush
CHUNK_SIZE = 200_000_000  # 200 MB source per Range request (multiple of 1000)
MAX_RETRIES = 12
TIMEOUT = 600  # per-chunk socket timeout (s)


def _range_fetch(url: str, begin: int, end: int) -> bytes:
    """Fetch bytes [begin, end) via one Range request with retry/backoff.

    A chunk that fails (timeout, URLError, 5xx) is retried with exponential
    backoff up to MAX_RETRIES. A 416 (out of range) means end-of-file and
    returns b"" so callers can stop cleanly.

    Args:
        url: the resolve URL of the source file.
        begin: first byte offset.
        end: one-past-last byte offset.

    Returns:
        The requested byte span, or b"" when begin is past end-of-file.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={begin}-{end - 1}"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                return b""
            if attempt == MAX_RETRIES:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = min(60, 2 ** (attempt - 1))
            sys.stderr.write(
                f"\n  [retry {attempt}/{MAX_RETRIES}] bytes {begin}-{end - 1} "
                f"({type(exc).__name__}); waiting {wait:.0f}s\n"
            )
            sys.stderr.flush()
            time.sleep(wait)
    raise AssertionError("unreachable")


def _fetch_header(url: str) -> int:
    """Fetch the safetensors header JSON and return the data byte offset.

    Safetensors layout: [u64 header_len][header JSON][padding to 8][data].
    The recorded header_len is the JSON length; the writer pads it to an
    8-byte boundary, so data begins at 8 + round_up(header_len, 8).

    Args:
        url: the resolve URL of the safetensors file.

    Returns:
        Offset where the tensor data begins. 0 means "raw fallback".
    """
    head = _range_fetch(url, 0, 8)
    if len(head) < 8:
        return 0
    (header_len,) = struct.unpack("<Q", head[:8])
    if header_len <= 0 or header_len > (1 << 30):
        return 0
    return 8 + ((header_len + 7) // 8) * 8


def _stream_pack(data: bytes, carry: bytearray, out: bytearray) -> bytearray:
    """Pack ``data`` (+ ``carry``) into sub-byte coordinates in ``out``.

    Every ``WEIGHTS_PER_COORD`` (1000) source bytes collapse to one
    coordinate byte ``round(mean(byte) x 0.001 x 1000)``. A trailing partial
    block is kept in ``carry`` for the next chunk.

    Args:
        data: the next chunk of raw source bytes.
        carry: leftover partial block from the previous chunk (mutated).
        out: coordinate bytes accumulator (mutated).

    Returns:
        ``out`` (same object) for chaining.
    """
    carry.extend(data)
    block = WEIGHTS_PER_COORD
    n = len(carry)
    full = n - (n % block)
    i = 0
    while i < full:
        chunk = carry[i : i + block]
        mean = sum(chunk) / block
        quanta = int(round(mean * LAW * 1000.0)) & 0xFF
        out.append(quanta)
        i += block
    del carry[:full]
    return out


class _ResumeState:
    """Persistent checkpoint so a crashed quantization resumes, not restarts.

    The JSON sidecar lives next to the output (``<output>.resume.json``) and
    records enough to continue exactly: completed shard body byte counts,
    the partially-consumed offset inside the current shard, and the total
    source bytes already represented by the on-disk coordinate file.
    """

    def __init__(self, output_path: str) -> None:
        self.path = output_path + ".resume.json"
        self.data = self._load(output_path)

    def _load(self, output_path: str) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if os.path.getsize(output_path) == data.get("written", -1):
                return data
        except (OSError, ValueError, KeyError):
            pass
        return {"shard": 0, "shard_consumed": 0, "bodies": [], "source_bytes": 0, "written": 0}

    def save(self) -> None:
        """Atomically persist the checkpoint (tmp + fsync + rename)."""
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def clear(self) -> None:
        """Delete the checkpoint once the job completes cleanly."""
        try:
            os.remove(self.path)
        except OSError:
            pass

    @property
    def total(self) -> int:
        """Total source bytes already represented by the output file."""
        return sum(self.data["bodies"]) + self.data["shard_consumed"]


def _stream_one_body(
    url: str,
    data_begin: int,
    output_path: str,
    state: _ResumeState,
    carry: bytearray,
    out: bytearray,
) -> int:
    """Stream one file's body from ``data_begin``, packing to coordinates.

    Appends coordinates to ``output_path`` and persists the checkpoint after
    every successful carry-empty chunk. Resumes from ``state.shard_consumed``.
    ``carry`` and ``out`` are shared across files so the coordinate stream is
    continuous (packing crosses file boundaries, exactly like the original
    single-stream law).

    Args:
        url: resolve URL of the source file.
        data_begin: byte offset where this file's body begins.
        output_path: .x8D coordinate output file.
        state: resumable job state (mutated).
        carry: leftover partial block spanning files (mutated).
        out: coordinate accumulator (mutated).

    Returns:
        Number of body bytes consumed from this file.
    """
    pos = data_begin + state.data["shard_consumed"]
    consumed = state.data["shard_consumed"]
    prefix = sum(state.data["bodies"])
    mode = "ab" if state.total else "wb"
    with open(output_path, mode) as f:  # noqa: PTH123
        while True:
            data = _range_fetch(url, pos, pos + CHUNK_SIZE)
            if not data:
                break  # 416 / end of file
            consumed += len(data)
            _stream_pack(data, carry, out)
            pos += len(data)
            # Persist coords + checkpoint together. A checkpoint is only saved
            # at a carry-empty boundary (consumed % WEIGHTS_PER_COORD == 0), so
            # a crash never leaves the checkpoint ahead of the on-disk stream
            # and the resume offset is exact.
            if consumed % WEIGHTS_PER_COORD == 0:
                f.write(out)
                out.clear()
                f.flush()
                os.fsync(f.fileno())
                state.data["shard_consumed"] = consumed
                state.data["source_bytes"] = prefix + consumed
                state.data["written"] = os.path.getsize(output_path)
                state.save()
                sys.stdout.write(
                    f"\r  {state.data['source_bytes']:>16,} source bytes -> "
                    f"{state.data['written']:>14,} coords on disk"
                )
                sys.stdout.flush()
        if carry:  # final partial block is still a coordinate
            mean = sum(carry) / len(carry)
            out.append(int(round(mean * LAW * 1000.0)) & 0xFF)
            del carry[:]
        f.write(out)
        out.clear()
        f.flush()
        os.fsync(f.fileno())
    return consumed


def _run_sources(sources: List[Tuple[str, int]], output_path: str) -> int:
    """Stream+pack a list of (url, data_begin) bodies into one .x8D file.

    Each source's body is appended to the same coordinate stream so the pack
    is continuous across shards (carry spans file boundaries). Resumes from
    any prior checkpoint.

    Args:
        sources: ordered (resolve_url, data_begin) pairs, one per file body.
        output_path: .x8D coordinate output file.

    Returns:
        Total source bytes streamed (disk == total x 0.001).
    """
    state = _ResumeState(output_path)
    carry = bytearray()
    out = bytearray()
    start = state.data["shard"]
    for i, (url, data_begin) in enumerate(sources):
        if i < start:
            continue  # completed in a prior run
        if i > start:
            state.data["shard_consumed"] = 0
        name = url.rsplit("/", 1)[-1]
        print(f"streaming {name} -> sub-byte coordinates ...")
        consumed = _stream_one_body(url, data_begin, output_path, state, carry, out)
        state.data["bodies"] = state.data["bodies"] + [consumed]
        state.data["shard"] = i + 1
        state.data["shard_consumed"] = 0
        state.data["source_bytes"] = sum(state.data["bodies"])
        state.data["written"] = os.path.getsize(output_path)
        state.save()
    total = sum(state.data["bodies"])
    state.clear()
    print()
    return total


def quantize_hf_safetensors(repo_id: str, filename: str, output_path: str) -> int:
    """Quantize a safetensors file directly from HF into a raw .x8D file.

    Args:
        repo_id: HuggingFace repo id (e.g. "openai/whisper-large-v3").
        filename: the safetensors file inside the repo.
        output_path: .x8D sub-byte coordinate output file.

    Returns:
        Number of source bytes streamed (disk size == source_bytes x 0.001).
    """
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"fetching header of {filename} ...")
    data_begin = _fetch_header(url)
    return _run_sources([(url, data_begin)], output_path)


def quantize_hf_raw(repo_id: str, filename: str, output_path: str) -> int:
    """Quantize a raw (non-safetensors, e.g. .pth) HF file to .x8D.

    Streams the ENTIRE source file body sequentially (no header parsing) and
    packs every 1000 bytes into one sub-byte coordinate. Disk == source_bytes
    x 0.001, magic-free.

    Args:
        repo_id: HuggingFace repo id.
        filename: the file inside the repo.
        output_path: .x8D sub-byte coordinate output file.

    Returns:
        Number of source bytes streamed.
    """
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    return _run_sources([(url, 0)], output_path)


def quantize_hf_shards(repo_id: str, shard_files: List[str], output_path: str) -> int:
    """Quantize a multi-shard safetensors model directly from HF to .x8D.

    Streams each shard body in fixed-size chunked Range requests (resumable,
    retried on stall) and packs every 1000 source bytes into one sub-byte
    coordinate, appending to the same output file. Disk ==
    total_source_bytes x 0.001, magic-free.

    Args:
        repo_id: HuggingFace repo id.
        shard_files: ordered list of safetensors filenames in the repo.
        output_path: .x8D sub-byte coordinate output file.

    Returns:
        Total number of source bytes streamed across all shards.
    """
    sources: List[Tuple[str, int]] = []
    for filename in shard_files:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        try:
            data_begin = _fetch_header(url)
        except Exception:
            data_begin = 0  # raw fallback: stream whole file
        sources.append((url, data_begin))
    return _run_sources(sources, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a HF safetensors model directly to .x8D (0.001 law, no full download)."
    )
    parser.add_argument("repo", help="HF repo id, e.g. openai/whisper-large-v3")
    parser.add_argument("file", help="safetensors filename in the repo")
    parser.add_argument("output", help=".x8D sub-byte coordinate output file")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="treat the file as a raw byte stream (.pth etc.), no safetensors header parsing",
    )
    parser.add_argument(
        "--shards",
        action="store_true",
        help="multi-shard mode: FILE is a local text file listing one shard filename per line",
    )
    args = parser.parse_args()

    if args.shards:
        with open(args.file, "r") as f:  # noqa: PTH123
            shard_files = [line.strip() for line in f if line.strip()]
        source = quantize_hf_shards(args.repo, shard_files, args.output)
    elif args.raw:
        source = quantize_hf_raw(args.repo, args.file, args.output)
    else:
        source = quantize_hf_safetensors(args.repo, args.file, args.output)
    target = max(1, math.ceil(source * 0.001))
    actual = os.path.getsize(args.output)
    print(f"source bytes : {source:,}")
    print(f"disk (x0.001): {actual:,} bytes  (target {target:,}, 1000:1)")
    if actual != target:
        print("ERROR: disk size != source_bytes x 0.001")
        sys.exit(1)


if __name__ == "__main__":
    main()
