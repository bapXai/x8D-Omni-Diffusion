# coding=utf-8
"""Generic x8D pointer quantizer for any HF model (index.json or single shard).

Handles both multi-shard models (via model.safetensors.index.json) and
single-file safetensors models (header fetched directly, no index needed).
All weights stay on the upstream HF disk; only the X8DPTR01 pointer map is
stored. /0.001 reverse happens live at query time on the fetched span.

Examples:
    python3 tools/quantize_hf.py --repo Lightricks/LTX-2 \
        --file ltx-2-19b-dev-fp4.safetensors --out ltx2.x8dptr.gguf
    python3 tools/quantize_hf.py --repo openai/whisper-large-v3 \
        --index model.safetensors.index.json --out whisper.x8dptr.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from quantize_kimi_k3 import (  # noqa: E402
    PTR_MAGIC,
    load_pointer_gguf,
    save_pointer_gguf,
    serve_expert_from_pointer,
)


def _resolve_url(repo: str, shard: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{shard}"


def _range_fetch(url: str, begin: int, end: int, token: Optional[str]) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={begin}-{end - 1}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return resp.read()


def build_single_file_pointers(
    repo: str, shard: str, token: Optional[str] = None, max_header_bytes: int = 64 * 1024 * 1024
) -> Dict[str, dict]:
    """Build pointer records for a single-file safetensors model.

    Fetches only the safetensors JSON header (a few MB) via HTTP Range; the
    weight bytes stay on the upstream disk.

    Args:
        repo: HF repo id.
        shard: safetensors file name in the repo.
        token: HF bearer token.
        max_header_bytes: header fetch cap.

    Returns:
        name -> pointer record (repo, shard, name, begin, end, dtype, shape).
    """
    from omni_diffusion.x8d_hf import parse_safetensors_header

    url = _resolve_url(repo, shard)
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes=0-{max_header_bytes - 1}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        head = resp.read()
    (header_len,) = struct.unpack("<Q", head[:8])
    if header_len > len(head) - 8:
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes=0-{8 + header_len - 1}")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp2:  # noqa: S310
            head = resp2.read()
    header = parse_safetensors_header(head[8 : 8 + header_len])
    data_start = 8 + header_len
    pointers: Dict[str, dict] = {}
    for name, spec in header.items():
        if name == "__metadata__" or "data_offsets" not in spec:
            continue
        begin, end = spec["data_offsets"]
        pointers[name] = {
            "repo": repo,
            "shard": shard,
            "name": name,
            "begin": data_start + begin,
            "end": data_start + end,
            "dtype": spec["dtype"],
            "shape": spec["shape"],
        }
    return pointers


def build_index_pointers(
    repo: str, index_url: str, token: Optional[str] = None, max_header_bytes: int = 16 * 1024 * 1024
) -> Dict[str, dict]:
    """Build pointer records for a multi-shard model via its index.json.

    Args:
        repo: HF repo id.
        index_url: full URL to model.safetensors.index.json.
        token: HF bearer token.
        max_header_bytes: per-shard header fetch cap.

    Returns:
        name -> pointer record.
    """
    from quantize_kimi_k3 import build_pointer_map, resolve_remote_offsets

    tmp = "/tmp/_x8d_index.json"
    req = urllib.request.Request(index_url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        open(tmp, "wb").write(resp.read())
    pointers = build_pointer_map(tmp, repo)
    resolve_remote_offsets(pointers, token=token, max_header_bytes=max_header_bytes)
    os.remove(tmp)
    return pointers


def main() -> None:
    ap = argparse.ArgumentParser(description="x8D pointer-quantize any HF model")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--index", default=None, help="model.safetensors.index.json path or URL")
    ap.add_argument("--file", default=None, help="single safetensors file name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    if args.file:
        pointers = build_single_file_pointers(args.repo, args.file, token=args.token)
    elif args.index:
        url = args.index if args.index.startswith("http") else _resolve_url(args.repo, args.index)
        pointers = build_index_pointers(args.repo, url, token=args.token)
    else:
        raise SystemExit("need --file or --index")

    ptr_bytes = save_pointer_gguf(pointers, args.out)
    print(f"{args.repo}: {len(pointers)} tensors -> {ptr_bytes} bytes")
    # verify: real invariants, not a tautology
    from omni_diffusion.x8d_hf import expected_span_length

    sample = min(pointers.items(), key=lambda kv: kv[1]["end"] - kv[1]["begin"])[1]
    raw = serve_expert_from_pointer(sample, token=args.token)
    span_ok = len(raw) == sample["end"] - sample["begin"]
    exp = expected_span_length(sample["shape"], sample["dtype"])
    shape_ok = exp is None or exp == sample["end"] - sample["begin"]
    print(
        f"  verify {sample['name']} ({len(raw)} B): "
        f"span[{sample['begin']},{sample['end']}) len_ok={span_ok} shape*{sample['dtype']}==span={shape_ok}"
    )
    if not (span_ok and shape_ok):
        raise SystemExit("pointer verification FAILED")


if __name__ == "__main__":
    main()
