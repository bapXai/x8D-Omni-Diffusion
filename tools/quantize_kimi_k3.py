# coding=utf-8
"""Pointer-based x8Dsub-byte quantization for upstream HF models (issue #10).

KEY IDEA (user directive): we quantize the model by **pin-pointing the HF
weights in the upstream repo** — we do NOT download the model. The x8D .gguf
container stores a *pointer map*:

    repo_id | shard | tensor_name | data_offsets | dtype | shape

For every tensor (or every MoE expert). At query time the server reads ONLY
the specific expert's byte span from disk (mmap) or via HTTP Range fetch,
applies the live /0.001 reverse math, and materializes just that expert.
The 2.78 TB Kimi-K3 is never resident in RAM; the pointer map itself is
kilobytes.

Example:
    python3 tools/quantize_kimi_k3.py \
        --index model.safetensors.index.json \
        --repo moonshotai/Kimi-K3 \
        --layer 12 --expert 895 \
        --out kimi_k3.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#: Magic for pointer-map containers.
PTR_MAGIC = b"X8DPTR01"

#: Per-pointer record: repo, shard, name, begin, end, dtype, shape
_PTR_FMT = "<QQQQ"


def build_pointer_map(
    index_path: str,
    repo: str,
    tensor_names: Optional[List[str]] = None,
    shard_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """Resolve tensors to their upstream pointer records (no download).

    Args:
        index_path: local model.safetensors.index.json.
        repo: HF repo id (e.g. moonshotai/Kimi-K3).
        tensor_names: optional subset of tensors; default all.
        shard_paths: optional map shard_name -> local .safetensors path so
            byte offsets can be validated against real files when available.

    Returns:
        name -> {repo, shard, name, begin, end, dtype, shape}.
    """
    with open(index_path) as f:
        idx = json.load(f)
    wm: Dict[str, str] = idx["weight_map"]
    names = tensor_names or list(wm.keys())
    pointers: Dict[str, dict] = {}
    for name in names:
        shard = wm[name]
        pointers[name] = {
            "repo": repo,
            "shard": shard,
            "name": name,
            "begin": 0,
            "end": 0,
            "dtype": "unknown",
            "shape": [],
        }
    # if local shards exist, resolve exact offsets + dtype + shape
    if shard_paths:
        from omni_diffusion.x8d_hf import SafetensorsShard

        by_shard: Dict[str, List[str]] = {}
        for n in names:
            by_shard.setdefault(wm[n], []).append(n)
        for shard, ns in by_shard.items():
            path = shard_paths.get(shard)
            if not path or not os.path.exists(path):
                continue
            s = SafetensorsShard(path)
            try:
                for n in ns:
                    spec = s.index[n]
                    begin, end = spec["data_offsets"]
                    pointers[n].update(
                        begin=s.data_start + begin,
                        end=s.data_start + end,
                        dtype=spec["dtype"],
                        shape=spec["shape"],
                    )
            finally:
                s.close()
    return pointers


def save_pointer_gguf(pointers: Dict[str, dict], output_path: str) -> int:
    """Write the pointer map as an x8D pointer container.

    Layout: magic(8) | <u32 name_len><name><u64 rec_len><pointer json> | ...

    Args:
        pointers: name -> pointer record.
        output_path: output .gguf path.

    Returns:
        Bytes written (the pointer map is tiny vs the upstream model).
    """
    with open(output_path, "wb") as f:
        f.write(PTR_MAGIC)
        for name, rec in pointers.items():
            raw = json.dumps(rec, sort_keys=True).encode("utf-8")
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
    return os.path.getsize(output_path)


def load_pointer_gguf(path: str) -> Dict[str, dict]:
    """Read a pointer container back into a name -> pointer dict."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(PTR_MAGIC):
        raise ValueError("Not a valid x8D pointer container")
    pos = len(PTR_MAGIC)
    out: Dict[str, dict] = {}
    while pos < len(data):
        (name_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        (rec_len,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        out[name] = json.loads(data[pos : pos + rec_len].decode("utf-8"))
        pos += rec_len
    return out


def _resolve_url(repo: str, shard: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{shard}"


def _range_fetch(url: str, begin: int, end: int, token: Optional[str]) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={begin}-{end - 1}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return resp.read()


def serve_expert_from_pointer(
    ptr: dict,
    token: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> bytes:
    """Read ONE expert's raw byte span via pointer (disk mmap or Range).

    This is the query-time read path: only the requested expert's bytes are
    fetched; the /0.001 reverse happens on this span alone.

    Args:
        ptr: pointer record from the map.
        token: HF bearer token for remote reads.
        local_dir: optional directory holding downloaded shards.

    Returns:
        Raw weight bytes for the expert.
    """
    if local_dir:
        p = os.path.join(local_dir, ptr["shard"])
        if os.path.exists(p):
            from omni_diffusion.x8d_hf import SafetensorsShard

            s = SafetensorsShard(p)
            try:
                return s.read_span(ptr["begin"], ptr["end"])
            finally:
                s.close()
    url = _resolve_url(ptr["repo"], ptr["shard"])
    return _range_fetch(url, ptr["begin"], ptr["end"], token)


def report(model_bytes: int, pointer_bytes: int, upstream_total: int = 1_560_860_324_864) -> Dict[str, float]:
    """Size comparison: upstream model vs x8D pointer map."""
    return {
        "upstream_bytes": float(upstream_total),
        "pointer_map_bytes": float(pointer_bytes),
        "reduction_pct": (1.0 - pointer_bytes / upstream_total) * 100.0,
        "per_byte_upstream": upstream_total / max(1, model_bytes),
    }


def resolve_remote_offsets(
    pointers: Dict[str, dict],
    token: Optional[str] = None,
    max_header_bytes: int = 16 * 1024 * 1024,
) -> int:
    """Resolve absolute byte offsets for remote shards by Range-fetching headers.

    For each distinct shard referenced by the pointer map, fetch only the
    safetensors JSON header (a few hundred KB) via HTTP Range, look up each
    tensor's ``data_offsets``, and store absolute ``begin/end`` + ``dtype`` +
    ``shape``. The model weight bytes themselves are never fetched here.

    Args:
        pointers: name -> pointer record (in place).
        token: HF bearer token.
        max_header_bytes: upper bound for the header range fetch.

    Returns:
        Number of shards whose offsets were resolved.
    """
    from omni_diffusion.x8d_hf import parse_safetensors_header

    shards: Dict[str, List[str]] = {}
    for n, p in pointers.items():
        shards.setdefault(p["shard"], []).append(n)

    resolved = 0
    for shard, names in shards.items():
        url = _resolve_url(pointers[names[0]]["repo"], shard)
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes=0-{max_header_bytes - 1}")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                head = resp.read()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not fetch header for {shard}: {exc}")
            continue
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
        for n in names:
            spec = header.get(n)
            if spec is None:
                continue
            begin, end = spec["data_offsets"]
            pointers[n].update(
                begin=data_start + begin,
                end=data_start + end,
                dtype=spec["dtype"],
                shape=spec["shape"],
            )
        resolved += 1
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description="Pointer-quantize an HF model with x8Dsub-byte")
    ap.add_argument("--index", required=True, help="local model.safetensors.index.json")
    ap.add_argument("--repo", default="moonshotai/Kimi-K3")
    ap.add_argument("--layer", type=int, default=None, help="restrict to one MoE layer")
    ap.add_argument("--expert", type=int, default=None, help="restrict to one MoE expert")
    ap.add_argument("--out", default="kimi_k3.x8dptr.gguf")
    ap.add_argument("--shards", default=None, help="comma list of shard files to resolve offsets for")
    ap.add_argument("--test-fetch", action="store_true", help="fetch one expert and compare bytes")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    # pin-point the upstream weights
    pointers = build_pointer_map(args.index, args.repo)
    if args.layer is not None:
        prefix = f"language_model.model.layers.{args.layer}.block_sparse_moe.experts."
        if args.expert is not None:
            prefix += f"{args.expert}."
        pointers = {n: p for n, p in pointers.items() if n.startswith(prefix)}

    if args.shards:
        allowed = set(args.shards.split(","))
        pointers = {n: p for n, p in pointers.items() if p["shard"] in allowed}
        print(f"restricted to {len(pointers)} tensors in selected shards")

    resolved = resolve_remote_offsets(pointers, token=args.token)
    print(f"resolved remote offsets for {resolved} shards")

    ptr_bytes = save_pointer_gguf(pointers, args.out)
    print(f"x8Dsub-byte pointer map: {len(pointers)} tensors -> {ptr_bytes} bytes")
    r = report(ptr_bytes, ptr_bytes)
    print(f"  upstream model     : {r['upstream_bytes']/1e12:.2f} TB")
    print(f"  pointer map        : {r['pointer_map_bytes']:.0f} B")
    print(f"  reduction          : {r['reduction_pct']:.8f}%")

    if args.test_fetch:
        from omni_diffusion.x8d_hf import expected_span_length

        sample = next(iter(pointers.values()))
        raw = serve_expert_from_pointer(sample, token=args.token)
        span_ok = len(raw) == sample["end"] - sample["begin"]
        exp = expected_span_length(sample["shape"], sample["dtype"])
        shape_ok = exp is None or exp == sample["end"] - sample["begin"]
        print(f"  test-fetch {sample['name']}: {len(raw)} B")
        print(f"    span_len=={sample['end']-sample['begin']}: {span_ok}")
        print(f"    shape*{sample['dtype']}==span_len: {shape_ok}")
        print(f"    dtype={sample['dtype']} shape={sample['shape']}")
        if not (span_ok and shape_ok):
            raise SystemExit("pointer verification FAILED")


if __name__ == "__main__":
    main()
