# coding=utf-8
"""Byte-native HF dataset import CLI (no tokenizer).

This is the byte-native ``load_dataset()`` equivalent: every dataset field
becomes raw bytes at ids 0-255. No tokenizer, no vocabulary lookup --
``datasets-server`` rows are flattened to ``name -> bytes`` leaf fields,
framed into a reversible byte stream, and stored losslessly through the
DSpark 8x8 speculative-decode quantizer into an x8D GGUF container.

Two ingestion paths:

1. **Live HF datasets** (``--dataset``): fetch rows via the datasets-server
   HTTP API and block-compress them into ``<name>.x8dds.gguf``.
2. **Local JSONL shards** (``--jsonl``): zero-copy ``mmap`` a local JSONL
   file (e.g. ai4bharat/sangraha or nvidia/Open-SWE-Traces shards) and
   compile its ``text``/``code`` fields into the same lossless U8 container.

The 0.001 law is applied at storage as **raw U8 coordinates only** -- never
as packed floats. ``/0.001`` is the live inverse at compute time; storing
``struct.pack("f", byte * 0.001)`` would quadruple the footprint (float
bloat) and is therefore rejected by the byte law.

Examples:
    python3 tools/import_hf_dataset.py --dataset sarvamai/indivibe \
        --config chat --split test --length 5 --out /tmp/x8d_ds_test

    python3 tools/import_hf_dataset.py \
        --jsonl sangraha/verified-shard.jsonl \
        --out /tmp/x8d_sangraha --name sangraha_verified
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_dataset import (  # noqa: E402
    X8DDatasetError,
    _write_manifest,
    block_compress_dataset,
    resolve_hf_dataset,
    rows_to_byte_stream,
)
from omni_diffusion.x8d_export import save_gguf, X8D_HEADER  # noqa: E402
from omni_diffusion.x8d_mmap import MappedX8DReader  # noqa: E402

#: Fields concatenated for the JSONL pipeline (byte-native text/code arrays).
_TEXT_FIELDS = ("text", "code", "content", "transcript", "problem", "solution")


def _default_name(dataset_id: str) -> str:
    """File-safe base name for a dataset id (slash -> underscore)."""
    return dataset_id.replace("/", "_")


def _yield_jsonl_lines(path: str) -> Iterable[bytes]:
    """Memory-map a JSONL shard and yield raw line bytes (zero-copy).

    The kernel page cache serves the mapped file; each line is sliced out of
    the mapping as bytes and decoded only at ``json.loads`` time. No
    ``read()`` of the whole file, no copy of the payload.

    Args:
        path: path to the JSONL shard.

    Yields:
        Raw line bytes from the mapping.
    """
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        with mmap.mmap(fd, size, access=mmap.ACCESS_READ) as mapping:
            start = 0
            while start < size:
                n = mapping.find(b"\n", start)
                if n == -1:
                    n = size
                line = mapping[start:n].strip()
                start = n + 1
                if line:
                    yield line
    finally:
        os.close(fd)


def jsonl_to_rows(path: str, limit: int = 0) -> List[Dict]:
    """Compile a JSONL shard into byte-native row dicts.

    Args:
        path: path to the JSONL shard.
        limit: max rows (0 = all).

    Returns:
        List of rows; each row is ``{"text": <concatenated field bytes>}``.
    """
    rows: List[Dict] = []
    for line in _yield_jsonl_lines(path):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        parts = []
        for key in _TEXT_FIELDS:
            value = data.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list) and all(
                isinstance(v, str) for v in value
            ):
                parts.extend(value)
        if parts:
            rows.append({"text": "".join(parts)})
        if limit and len(rows) >= limit:
            break
    return rows


def convert_jsonl_to_x8d(input_jsonl: str, output_bin: str, name: str) -> Dict:
    """Compile a JSONL shard into a lossless U8 x8D GGUF container.

    Zero-copy: the input is memory-mapped (``_yield_jsonl_lines``), rows are
    flattened to ``text`` bytes, framed into an ``X8DDS`` byte stream and
    stored behind the ``X8DGGUF1`` magic with raw U8 coordinates -- the
    0.001 scaling lives at compute time only, exactly as the byte law
    requires (no float32 packing, no wrapper pollution).

    Args:
        input_jsonl: path to the JSONL shard.
        output_bin: output ``.x8dds.gguf`` path.
        name: payload name.

    Returns:
        Manifest dict with stream_bytes / gguf_bytes / rows / lossless.

    Raises:
        X8DDatasetError: if the lossless roundtrip verification fails.
    """
    if not os.path.exists(input_jsonl):
        raise X8DDatasetError(f"input file not found: {input_jsonl}")

    rows = jsonl_to_rows(input_jsonl)
    if not rows:
        raise X8DDatasetError(f"no byte-native rows parsed from {input_jsonl}")

    stream = rows_to_byte_stream(rows)
    payloads = {name: stream}
    save_gguf(payloads, output_bin)

    # Zero-copy read-back verification: the compressed state IS the running
    # state -- we slice the payload straight out of the mmap and compare.
    with MappedX8DReader(output_bin) as reader:
        restored = reader.load(name)
    lossless = restored == stream
    if not lossless:
        raise X8DDatasetError(
            f"lossless roundtrip verification FAILED for {input_jsonl}"
        )

    manifest = {
        "dataset": os.path.basename(input_jsonl),
        "method": "x8d-jsonl-mmap-u8",
        "rows_count": len(rows),
        "stream_bytes": len(stream),
        "gguf_bytes": os.path.getsize(output_bin),
        "threshold": 0.001,
        "roundtrip_lossless": True,
    }
    manifest_dir = os.path.dirname(os.path.abspath(output_bin))
    manifest_path = os.path.join(manifest_dir, "manifest.json")
    _write_manifest(manifest, manifest_path)
    return manifest


def _run_jsonl_pipeline(args: argparse.Namespace) -> int:
    """Local JSONL -> x8D GGUF pipeline (mmap, zero-copy, lossless)."""
    import time

    name = args.name or _default_name(os.path.basename(args.jsonl))
    out_gguf = os.path.join(args.out, f"{name}.x8dds.gguf")
    os.makedirs(args.out, exist_ok=True)
    manifest = convert_jsonl_to_x8d(args.jsonl, out_gguf, name)
    print(
        f"jsonl={args.jsonl} rows={manifest['rows_count']} "
        f"stream_bytes={manifest['stream_bytes']} "
        f"gguf_bytes={manifest['gguf_bytes']} "
        f"lossless={manifest['roundtrip_lossless']}"
    )
    print(f"wrote {out_gguf} + manifest.json (U8, 0.001 law at compute)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Import an HF dataset or local JSONL as a byte-native "
        "x8D GGUF container"
    )
    ap.add_argument("--dataset", default=None, help="HF dataset id (live path)")
    ap.add_argument("--jsonl", default=None, help="local JSONL shard (mmap path)")
    ap.add_argument("--config", default=None, help="dataset config (default: first available)")
    ap.add_argument("--split", default="train", help="split (default: train; falls back to first)")
    ap.add_argument("--offset", type=int, default=0, help="starting row index")
    ap.add_argument("--length", type=int, default=100, help="rows to fetch (max 1000)")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--name", default=None, help="file base name (default: dataset id with / -> _)")
    ap.add_argument("--timeout", type=float, default=30, help="socket timeout in seconds")
    ap.add_argument("--heavy-load", action="store_true", help="clip verification length")
    args = ap.parse_args()

    if not args.dataset and not args.jsonl:
        ap.error("provide --dataset (live HF) or --jsonl (local shard)")
    if args.dataset and args.jsonl:
        ap.error("provide --dataset or --jsonl, not both")

    try:
        if args.jsonl:
            return _run_jsonl_pipeline(args)

        if not 0 < args.length <= 1000:
            ap.error("--length must be in [1, 1000]")

        resolved = resolve_hf_dataset(
            args.dataset,
            config=args.config,
            split=args.split,
            offset=args.offset,
            length=args.length,
            timeout=args.timeout,
        )
        rows = resolved["rows"]
        name = args.name or _default_name(args.dataset)
        manifest = block_compress_dataset(
            rows,
            args.out,
            name,
            seed=0,
            heavy_load=args.heavy_load,
        )
        manifest["dataset"] = resolved["dataset"]
        manifest["config"] = resolved["config"]
        manifest["split"] = resolved["split"]
        manifest_path = os.path.join(args.out, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        ratio = manifest["stream_bytes"] / manifest["gguf_bytes"]
        print(
            f"dataset={manifest['dataset']} config={manifest['config']} "
            f"split={manifest['split']}"
        )
        print(
            f"rows fetched={len(rows)} num_rows_total={resolved['num_rows_total']} "
            f"stream_bytes={manifest['stream_bytes']} gguf_bytes={manifest['gguf_bytes']} "
            f"ratio={ratio:.2f}"
        )
        print(
            f"wrote {os.path.join(args.out, manifest['gguf'])} + manifest.json (lossless)"
        )
        return 0
    except X8DDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
