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

Method (single streaming connection, no per-tensor range churn):
  1. HTTP Range-fetch the safetensors header (bytes 0..header_len) only.
  2. Stream the remaining file body sequentially in 1 MiB blocks, packing
     each 1000-byte block into one sub-byte coordinate byte.
  3. The full model is never downloaded or stored.

Usage:
    python3 tools/quantize_hf_safetensors.py \
        openai/whisper-large-v3 model.safetensors /tmp/whisper.x8D
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import urllib.request
from typing import Iterator, Tuple

from omni_diffusion.x8d_subbyte import LAW, WEIGHTS_PER_COORD

STREAM_BLOCK = 1 << 20  # 1 MiB streaming read


def _range_fetch(url: str, begin: int, end: int) -> bytes:
    """Fetch bytes [begin, end) via an HTTP Range request."""
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={begin}-{end - 1}"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _fetch_header(url: str) -> int:
    """Fetch the safetensors header JSON and return its byte length.

    Safetensors layout: [u64 header_len][header JSON][padding to 8][data].
    """
    head = _range_fetch(url, 0, 8)
    (header_len,) = struct.unpack("<Q", head)
    # data begins right after header_len + JSON (already 8-aligned by writer)
    return 8 + header_len


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

    req = urllib.request.Request(url, headers={"Range": f"bytes={data_begin}-"})
    carry = bytearray()
    out = bytearray()
    source_bytes = 0
    written = 0
    print("streaming body -> sub-byte coordinates ...")
    with urllib.request.urlopen(req, timeout=120) as r:
        with open(output_path, "wb") as f:
            while True:
                data = r.read(STREAM_BLOCK)
                if not data:
                    break
                source_bytes += len(data)
                _stream_pack(data, carry, out)
                if len(out) >= STREAM_BLOCK:
                    f.write(out[:STREAM_BLOCK])
                    written += STREAM_BLOCK
                    del out[:STREAM_BLOCK]
                sys.stdout.write(f"\r  {source_bytes:>16,} source bytes -> {written + len(out):>14,} coords")
                sys.stdout.flush()
            # final partial block (if any) is still a coordinate
            if carry:
                mean = sum(carry) / len(carry)
                out.append(int(round(mean * LAW * 1000.0)) & 0xFF)
            f.write(out)
            written += len(out)
    print()
    return source_bytes


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
    req = urllib.request.Request(url, headers={"Range": "bytes=0-"})
    carry = bytearray()
    out = bytearray()
    source_bytes = 0
    written = 0
    print(f"streaming raw {filename} -> sub-byte coordinates ...")
    with urllib.request.urlopen(req, timeout=120) as r:
        with open(output_path, "wb") as f:
            while True:
                data = r.read(STREAM_BLOCK)
                if not data:
                    break
                source_bytes += len(data)
                _stream_pack(data, carry, out)
                if len(out) >= STREAM_BLOCK:
                    f.write(out[:STREAM_BLOCK])
                    written += STREAM_BLOCK
                    del out[:STREAM_BLOCK]
                sys.stdout.write(f"\r  {source_bytes:>16,} source bytes -> {written + len(out):>14,} coords")
                sys.stdout.flush()
            if carry:
                mean = sum(carry) / len(carry)
                out.append(int(round(mean * LAW * 1000.0)) & 0xFF)
            f.write(out)
            written += len(out)
    print()
    return source_bytes


def quantize_hf_shards(repo_id: str, shard_files: list, output_path: str) -> int:
    """Quantize a multi-shard safetensors model directly from HF to .x8D.

    Streams each shard body sequentially in one connection and packs every
    1000 source bytes into one sub-byte coordinate, appending to the same
    output file. Disk == total_source_bytes x 0.001, magic-free.

    Args:
        repo_id: HuggingFace repo id.
        shard_files: ordered list of safetensors filenames in the repo.
        output_path: .x8D sub-byte coordinate output file.

    Returns:
        Total number of source bytes streamed across all shards.
    """
    carry = bytearray()
    out = bytearray()
    source_bytes = 0
    written = 0
    total_source = 0
    for filename in shard_files:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        print(f"streaming {filename} -> sub-byte coordinates ...")
        try:
            data_begin = _fetch_header(url)
        except Exception:
            data_begin = 0  # raw fallback: stream whole file
        req = urllib.request.Request(url, headers={"Range": f"bytes={data_begin}-"})
        with urllib.request.urlopen(req, timeout=120) as r:
            with open(output_path, "ab" if source_bytes else "wb") as f:
                while True:
                    data = r.read(STREAM_BLOCK)
                    if not data:
                        break
                    source_bytes += len(data)
                    _stream_pack(data, carry, out)
                    if len(out) >= STREAM_BLOCK:
                        f.write(out[:STREAM_BLOCK])
                        written += STREAM_BLOCK
                        del out[:STREAM_BLOCK]
                    if source_bytes - total_source >= STREAM_BLOCK:
                        sys.stdout.write(f"\r  {source_bytes:>16,} source bytes -> {written + len(out):>14,} coords")
                        sys.stdout.flush()
                        total_source = source_bytes
        total_source = source_bytes
    with open(output_path, "ab") as f:
        if carry:
            mean = sum(carry) / len(carry)
            out.append(int(round(mean * LAW * 1000.0)) & 0xFF)
        f.write(out)
        written += len(out)
    print()
    return source_bytes


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
        with open(args.file, "r") as f:
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
