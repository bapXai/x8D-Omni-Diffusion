# coding=utf-8
"""Byte-native HF dataset import + 8x8 block compression (no tokenizer).

This module is the byte-native ``load_dataset()`` equivalent: instead of a
tokenizer or vocabulary, every dataset field becomes raw bytes living at
ids 0-255. Pure Python standard library only (``urllib.request``, ``json``,
``struct``, ``time``, ``os``) -- no ``datasets``, ``torch``,
``transformers`` or ``requests``.

Two halves:

1. **Import** -- ``resolve_hf_dataset`` talks to the HuggingFace
   ``datasets-server`` HTTP API (``parquet`` + ``rows`` endpoints) and
   returns rows as plain dicts. Requested config/split are resolved against
   the dataset's parquet file index (first available fallbacks when the
   requested value is absent).
2. **Compression** -- each row is flattened into dotted ``name -> bytes``
   leaf fields, framed into a reversible byte stream
   (``rows_to_byte_stream`` / ``byte_stream_to_rows``), then stored through
   the DSpark 8x8 speculative-decode quantizer into an x8D GGUF container
   via ``block_compress_dataset``. The 0.001 sub-byte law makes the
   container payload a LOSSLESS copy of the original byte stream.

Every field is a raw 8-bit byte sequence; there is no encoding step and no
vocabulary lookup anywhere in the pipeline.
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

from .x8d_export import LAW, load_gguf

#: datasets-server base URL.
DATASETS_SERVER: str = "https://datasets-server.huggingface.co"

#: Row-stream magic: "X8DDS" + version 0x00 0x01.
MAGIC: bytes = b"X8DDS\x00\x01"

#: HTTP User-Agent sent on all datasets-server requests.
USER_AGENT: str = "x8d/1.0"

#: Type tags embedded as the first byte of each framed value so the byte
#: stream is genuinely reversible (e.g. "" vs b"" stay distinct).
_TAG_STR: int = 0
_TAG_BOOL: int = 1
_TAG_INT: int = 2
_TAG_FLOAT: int = 3
_TAG_BYTES: int = 4


class X8DDatasetError(ValueError):
    """Raised on datasets-server HTTP/JSON failures and stream corruption."""


def field_to_bytes(value: object) -> Optional[bytes]:
    """Convert a scalar field value into raw 8-bit bytes, or None.

    Args:
        value: scalar value (str, bool, int, float, bytes, bytearray).
            Containers (list/tuple/dict) and None return None; they are
            expanded by ``flatten_fields`` instead.

    Returns:
        Raw bytes encoding of the scalar, or None if unencodable/nested.

    Raises:
        X8DDatasetError: int not representable as unsigned 64-bit.
    """
    enc = _encode_scalar(value)
    return None if enc is None else enc[1]


def _encode_scalar(value: object) -> Optional[Tuple[int, bytes]]:
    """Return ``(type_tag, payload)`` for a scalar, or None if unencodable.

    Mirrors ``field_to_bytes`` but keeps the type tag so the row-stream
    framing can rebuild the exact Python scalar on the way back out.
    """
    if isinstance(value, str):
        return (_TAG_STR, value.encode("utf-8"))
    if isinstance(value, bool):
        return (_TAG_BOOL, b"\x01" if value else b"\x00")
    if isinstance(value, int):
        if not 0 <= value < 2**64:
            raise X8DDatasetError(f"int {value} out of unsigned 64-bit range")
        return (_TAG_INT, struct.pack("<Q", value))
    if isinstance(value, float):
        return (_TAG_FLOAT, struct.pack("<d", value))
    if isinstance(value, (bytes, bytearray)):
        return (_TAG_BYTES, bytes(value))
    return None


def _walk_leaves(obj: object, path: str) -> List[Tuple[str, object]]:
    """Recursively collect ``(dotted_path, scalar)`` leaves of a row.

    Dict keys join with '.', list items appear as ``[i]`` (e.g.
    ``messages[0].role``). Containers recurse; scalars are emitted with
    their original Python type intact.
    """
    out: List[Tuple[str, object]] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            child = str(key)
            child_path = f"{path}.{child}" if path else child
            out.extend(_walk_leaves(val, child_path))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            out.extend(_walk_leaves(val, f"{path}[{i}]"))
    else:
        out.append((path, obj))
    return out


def flatten_fields(row: Dict) -> List[Tuple[str, bytes]]:
    """Recursively flatten a row into dotted ``path -> bytes`` leaf fields.

    Args:
        row: dataset row (nested dicts/lists of scalars).

    Returns:
        List of ``(dotted_path, bytes)`` pairs. Dict keys join with '.',
        list items appear as ``[i]``. None and unencodable leaves are
        skipped.
    """
    out: List[Tuple[str, bytes]] = []
    for path, value in _walk_leaves(row, ""):
        data = field_to_bytes(value)
        if data is not None:
            out.append((path, data))
    return out


def rows_to_byte_stream(rows: List[Dict]) -> bytes:
    """Frame rows into a reversible byte stream.

    Layout (all little-endian): MAGIC + u64 row_count, then per row: u64
    row_len, then row payload = u16 nfields + per field: u16 name_len, name
    bytes, u64 val_len, value bytes. Each ``value`` begins with a u8 type
    tag so scalar types survive the roundtrip (str vs bytes, int vs float).

    Args:
        rows: list of row dicts (nested structures are flattened).

    Returns:
        The framed byte stream.
    """
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<Q", len(rows))
    for row in rows:
        payload = bytearray()
        fields: List[Tuple[str, int, bytes]] = []
        for path, value in _walk_leaves(row, ""):
            enc = _encode_scalar(value)
            if enc is None:
                continue
            fields.append((path, enc[0], enc[1]))
        payload += struct.pack("<H", len(fields))
        for path, tag, data in fields:
            name_b = path.encode("utf-8")
            payload += struct.pack("<H", len(name_b))
            payload += name_b
            encoded = bytes((tag,)) + data
            payload += struct.pack("<Q", len(encoded))
            payload += encoded
        buf += struct.pack("<Q", len(payload))
        buf += payload
    return bytes(buf)


def byte_stream_to_rows(stream: bytes) -> List[Dict]:
    """Inverse of ``rows_to_byte_stream``.

    Args:
        stream: framed byte stream produced by ``rows_to_byte_stream``.

    Returns:
        Rebuilt row dicts; dotted paths are re-nested into dicts/lists and
        scalar values are restored to their original Python types.

    Raises:
        X8DDatasetError: bad magic or truncated stream.
    """
    if not stream.startswith(MAGIC):
        raise X8DDatasetError(f"bad magic {stream[:len(MAGIC)]!r} != {MAGIC!r}")
    pos = len(MAGIC)
    if pos + 8 > len(stream):
        raise X8DDatasetError("truncated row count")
    (row_count,) = struct.unpack_from("<Q", stream, pos)
    pos += 8
    rows: List[Dict] = []
    for _ in range(row_count):
        if pos + 8 > len(stream):
            raise X8DDatasetError("truncated row length")
        (row_len,) = struct.unpack_from("<Q", stream, pos)
        pos += 8
        if pos + row_len > len(stream):
            raise X8DDatasetError("truncated row payload")
        payload = stream[pos : pos + row_len]
        pos += row_len
        if len(payload) < 2:
            raise X8DDatasetError("truncated field count")
        (nfields,) = struct.unpack_from("<H", payload)
        p = 2
        fields: List[Tuple[str, object]] = []
        for _ in range(nfields):
            if p + 2 > len(payload):
                raise X8DDatasetError("truncated field name length")
            (name_len,) = struct.unpack_from("<H", payload, p)
            p += 2
            if p + name_len > len(payload):
                raise X8DDatasetError("truncated field name")
            name = payload[p : p + name_len].decode("utf-8")
            p += name_len
            if p + 8 > len(payload):
                raise X8DDatasetError("truncated field value length")
            (val_len,) = struct.unpack_from("<Q", payload, p)
            p += 8
            if p + val_len > len(payload):
                raise X8DDatasetError("truncated field value")
            value = payload[p : p + val_len]
            p += val_len
            fields.append((name, _decode_scalar(value)))
        rows.append(_rebuild_row(fields))
    return rows


def _decode_scalar(value: bytes) -> object:
    """Invert ``_encode_scalar``: unpack ``(tag, payload)`` into a scalar.

    Args:
        value: framed value bytes starting with a u8 type tag.

    Returns:
        The reconstructed Python scalar.

    Raises:
        X8DDatasetError: unknown tag or missing tag byte.
    """
    if not value:
        raise X8DDatasetError("empty framed value (missing type tag)")
    tag = value[0]
    data = value[1:]
    if tag == _TAG_STR:
        return data.decode("utf-8")
    if tag == _TAG_BOOL:
        return data == b"\x01"
    if tag == _TAG_INT:
        (v,) = struct.unpack("<Q", data)
        return v
    if tag == _TAG_FLOAT:
        (v,) = struct.unpack("<d", data)
        return v
    if tag == _TAG_BYTES:
        return data
    raise X8DDatasetError(f"unknown type tag {tag}")


def _split_path(name: str) -> List[str]:
    """Split a dotted path like ``messages[0].role`` into parts.

    Args:
        name: dotted path with optional ``[i]`` list indices.

    Returns:
        Parts, e.g. ``["messages", "[0]", "role"]``.
    """
    parts: List[str] = []
    for chunk in name.split("."):
        rest = chunk
        while "[" in rest:
            idx = rest.index("[")
            if idx > 0:
                parts.append(rest[:idx])
            end = rest.find("]", idx)
            if end == -1:
                parts.append(rest[idx:])
                rest = ""
                break
            parts.append(rest[idx : end + 1])
            rest = rest[end + 1 :]
        if rest:
            parts.append(rest)
    return parts


def _rebuild_row(fields: List[Tuple[str, object]]) -> Dict:
    """Rebuild a nested dict/list row from flattened dotted paths.

    List indices ``[i]`` grow lists in index order; missing indices are
    back-filled with None so later siblings land at the right position.

    Args:
        fields: (dotted_path, scalar) pairs in flattened order.

    Returns:
        The nested row structure.

    Raises:
        X8DDatasetError: path structure is inconsistent (e.g. a list index
            into a dict, or an empty path).
    """
    root: Dict = {}
    for name, value in fields:
        parts = _split_path(name)
        if not parts:
            raise X8DDatasetError(f"empty field path {name!r}")
        node: object = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            is_list = part.startswith("[") and part.endswith("]")
            if is_list:
                idx = int(part[1:-1])
                if not isinstance(node, list):
                    raise X8DDatasetError(f"path {name!r} indexes a non-list")
                while len(node) <= idx:
                    node.append(None)
                if last:
                    node[idx] = value
                else:
                    nxt = parts[i + 1]
                    nxt_list = nxt.startswith("[") and nxt.endswith("]")
                    if node[idx] is None:
                        node[idx] = [] if nxt_list else {}
                    node = node[idx]
            else:
                if not isinstance(node, dict):
                    raise X8DDatasetError(f"path {name!r} traverses a non-dict")
                if last:
                    node[part] = value
                else:
                    nxt = parts[i + 1]
                    nxt_list = nxt.startswith("[") and nxt.endswith("]")
                    if part not in node or node[part] is None:
                        node[part] = [] if nxt_list else {}
                    node = node[part]
    return root


def _http_get_json(url: str, timeout: float) -> Dict:
    """GET ``url`` with the x8d User-Agent and parse the JSON reply.

    Args:
        url: datasets-server endpoint URL.
        timeout: socket timeout in seconds.

    Returns:
        Parsed JSON object (dict).

    Raises:
        X8DDatasetError: HTTP error (with status + reason), network error,
            or invalid JSON.
    """
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise X8DDatasetError(f"HTTP {exc.code} {exc.reason}: {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise X8DDatasetError(f"request failed ({exc}): {url}") from exc
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise X8DDatasetError(f"invalid JSON from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise X8DDatasetError(f"non-object JSON from {url}: {payload!r}")
    return payload


def resolve_hf_dataset(
    dataset_id: str,
    config: Optional[str] = None,
    split: str = "train",
    offset: int = 0,
    length: int = 100,
    timeout: float = 30,
) -> Dict:
    """Resolve config/split and fetch rows from the HF datasets-server.

    Args:
        dataset_id: HF dataset id (e.g. "sarvamai/indivibe").
        config: requested config name; None picks the first available.
        split: requested split; falls back to the first split of the chosen
            config when the requested split is absent.
        offset: starting row index.
        length: number of rows to fetch.
        timeout: socket timeout in seconds.

    Returns:
        Dict with keys: dataset, config, split, rows (list of row dicts),
        num_rows_total (int).

    Raises:
        X8DDatasetError: on HTTP/JSON failure (status + reason included).
    """
    parquet_url = f"{DATASETS_SERVER}/parquet?{urlencode({'dataset': dataset_id})}"
    parquet = _http_get_json(parquet_url, timeout)
    files = parquet.get("parquet_files")
    if not isinstance(files, list) or not files:
        raise X8DDatasetError(
            f"no parquet_files in datasets-server reply for {dataset_id}"
        )

    splits_by_config: Dict[str, List[str]] = {}
    for f in files:
        if not isinstance(f, dict):
            continue
        cfg = f.get("config")
        spl = f.get("split")
        if isinstance(cfg, str):
            bucket = splits_by_config.setdefault(cfg, [])
            if isinstance(spl, str) and spl not in bucket:
                bucket.append(spl)

    if config is None:
        config = next(iter(splits_by_config))
    if config not in splits_by_config:
        raise X8DDatasetError(
            f"config {config!r} not found for {dataset_id} "
            f"(available: {sorted(splits_by_config)})"
        )

    splits = splits_by_config[config]
    if split not in splits:
        if not splits:
            raise X8DDatasetError(f"config {config!r} has no splits for {dataset_id}")
        split = splits[0]

    rows_url = (
        f"{DATASETS_SERVER}/rows?"
        f"{urlencode({'dataset': dataset_id, 'config': config, 'split': split, 'offset': offset, 'length': length})}"
    )
    rows_payload = _http_get_json(rows_url, timeout)
    rows_data = rows_payload.get("rows")
    if not isinstance(rows_data, list):
        raise X8DDatasetError(
            f"rows endpoint returned no rows list for {dataset_id}/{config}/{split}"
        )
    rows: List[Dict] = []
    for entry in rows_data:
        if isinstance(entry, dict) and isinstance(entry.get("row"), dict):
            rows.append(entry["row"])
    num_total = rows_payload.get("num_rows_total")
    num_rows_total = int(num_total) if num_total is not None else len(rows)
    return {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "rows": rows,
        "num_rows_total": num_rows_total,
    }


def block_compress_dataset(
    rows: List[Dict],
    out_dir: str,
    name: str,
    max_steps: int = 16,
    seed: int = 0,
    heavy_load: bool = False,
) -> Dict:
    """Compress rows through the DSpark 8x8 spec-decode quantizer.

    Frames the rows into a byte stream (``rows_to_byte_stream``), stores it
    via ``speculative_save_gguf`` into ``out_dir/<name>.x8dds.gguf``, and
    verifies the container payload is byte-identical to the stream (the
    roundtrip through the 0.001 law is lossless).

    Args:
        rows: list of row dicts.
        out_dir: output directory (created if missing).
        name: dataset/tensor name (file base and container payload key).
        max_steps: regeneration budget per 8x8 block.
        seed: RNG seed.
        heavy_load: clip verification length.

    Returns:
        Manifest dict (also written to ``out_dir/manifest.json``).

    Raises:
        X8DDatasetError: if the lossless roundtrip verification fails.
    """
    from .x8d_spec_decode import speculative_save_gguf

    stream = rows_to_byte_stream(rows)
    os.makedirs(out_dir, exist_ok=True)
    gguf_name = f"{name}.x8dds.gguf"
    gguf_path = os.path.join(out_dir, gguf_name)
    speculative_save_gguf(
        name, stream, gguf_path, max_steps=max_steps, seed=seed, heavy_load=heavy_load
    )
    payloads, _ = load_gguf(gguf_path)
    if payloads.get(name) != stream:
        raise X8DDatasetError("lossless roundtrip verification FAILED")
    manifest = {
        "dataset": None,
        "config": None,
        "split": None,
        "rows_count": len(rows),
        "stream_bytes": len(stream),
        "gguf": gguf_name,
        "gguf_bytes": os.path.getsize(gguf_path),
        "method": "x8d-spec-8x8",
        "threshold": LAW,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roundtrip_lossless": True,
    }
    _write_manifest(manifest, os.path.join(out_dir, "manifest.json"))
    return manifest


def read_manifest(out_dir: str) -> Dict:
    """Read the manifest written by ``block_compress_dataset``.

    Args:
        out_dir: output directory.

    Returns:
        Manifest dict.

    Raises:
        X8DDatasetError: missing or invalid manifest.json.
    """
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        raise X8DDatasetError(f"no manifest.json in {out_dir}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise X8DDatasetError(f"invalid manifest.json in {out_dir}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise X8DDatasetError(f"manifest.json in {out_dir} is not an object")
    return manifest


def _write_manifest(manifest: Dict, path: str) -> None:
    """Write a manifest dict as indented JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
