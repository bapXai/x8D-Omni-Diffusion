# coding=utf-8
"""Byte-native HF dataset import CLI (no tokenizer).

This is the byte-native ``load_dataset()`` equivalent: every dataset field
becomes raw bytes at ids 0-255. No tokenizer, no vocabulary lookup --
``datasets-server`` rows are flattened to ``name -> bytes`` leaf fields,
framed into a reversible byte stream, and stored losslessly through the
DSpark 8x8 speculative-decode quantizer into an x8D GGUF container.

Examples:
    python3 tools/import_hf_dataset.py --dataset sarvamai/indivibe \
        --config chat --split test --length 5 --out /tmp/x8d_ds_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_dataset import (  # noqa: E402
    X8DDatasetError,
    block_compress_dataset,
    resolve_hf_dataset,
)


def _default_name(dataset_id: str) -> str:
    """File-safe base name for a dataset id (slash -> underscore)."""
    return dataset_id.replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Import an HF dataset as a byte-native x8D GGUF container"
    )
    ap.add_argument("--dataset", required=True, help="HF dataset id")
    ap.add_argument("--config", default=None, help="dataset config (default: first available)")
    ap.add_argument("--split", default="train", help="split (default: train; falls back to first)")
    ap.add_argument("--offset", type=int, default=0, help="starting row index")
    ap.add_argument("--length", type=int, default=100, help="rows to fetch (max 1000)")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--name", default=None, help="file base name (default: dataset id with / -> _)")
    ap.add_argument("--timeout", type=float, default=30, help="socket timeout in seconds")
    ap.add_argument("--heavy-load", action="store_true", help="clip verification length")
    args = ap.parse_args()

    if not 0 < args.length <= 1000:
        ap.error("--length must be in [1, 1000]")

    try:
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
