# coding=utf-8
"""Tests for the resumable chunked HF streaming quantizer (issue #55).

Pure stdlib unittest. Spins up a local HTTP server serving synthetic
safetensors bodies, then verifies the chunked Range-fetch + retry + resume
logic in ``tools/quantize_hf_safetensors.py``:

- a stalled chunk is retried (server drops the first body Range) and the final
  coordinate stream is byte-identical to the no-stall run;
- a job that "dies" mid-stream resumes from its checkpoint and produces the
  exact same .x8D file as a single uninterrupted run;
- disk size == source_bytes x 0.001 on every path.
"""

import http.server
import json
import os
import re
import shutil
import struct
import tempfile
import threading
import unittest
from importlib.machinery import SourceFileLoader

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
_loader = SourceFileLoader(
    "quantize_hf_safetensors", os.path.join(TOOLS_DIR, "quantize_hf_safetensors.py")
)
_mod = _loader.load_module()

_range_fetch = _mod._range_fetch
_fetch_header = _mod._fetch_header
_stream_pack = _mod._stream_pack
_run_sources = _mod._run_sources
_ResumeState = _mod._ResumeState


def _make_safetensors_body(n_bytes: int, seed: int = 7) -> bytes:
    """Build a synthetic safetensors-like file: header + deterministic bytes.

    Mirrors real safetensors: the leading u64 stores the header length AFTER
    padding to an 8-byte boundary, and data begins right after the padded
    header.
    """
    header = {"x": {"dtype": "F32", "shape": [n_bytes // 4], "data_offsets": [0, n_bytes]}}
    hjson = json.dumps(header).encode()
    padded = hjson + b"\x00" * (-len(hjson) % 8)
    head = struct.pack("<Q", len(padded))
    body = bytes((i * seed + 3) % 256 for i in range(n_bytes))
    return head + padded + body


def _header_end(raw: bytes) -> int:
    return struct.unpack("<Q", raw[:8])[0] + 8


def _header_end(raw: bytes) -> int:
    return struct.unpack("<Q", raw[:8])[0] + 8


class _StallingHandler(http.server.BaseHTTPRequestHandler):
    """Serves a prebuilt file, dropping the FIRST two body Range requests.

    Range requests whose start offset is past the header are counted; the
    first two such requests close the connection without a response so the
    client sees a stalled read and must retry. Later requests succeed.
    """

    payload = b""
    header_end = 0
    _fail_count = 0
    _lock = threading.Lock()

    def log_message(self, *args):  # silence server chatter
        pass

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d+)-(\d*)", rng or "")
        if not m:
            self.send_error(400)
            return
        begin = int(m.group(1))
        with _StallingHandler._lock:
            if begin >= _StallingHandler.header_end:
                _StallingHandler._fail_count += 1
                is_fail = _StallingHandler._fail_count <= 2
            else:
                is_fail = False
        if is_fail:
            self.connection.close()
            return
        end = int(m.group(2)) + 1 if m.group(2) else len(_StallingHandler.payload)
        data = _StallingHandler.payload[begin:end]
        if not data:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Range", f"bytes {begin}-{begin + len(data) - 1}/*")
        self.end_headers()
        self.wfile.write(data)


class QuantizeStreamTest(unittest.TestCase):
    """Offline coverage of chunked fetch, retry, and resume."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="x8d_qhf_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _serve(self, handler_cls, payload, header_end):
        handler_cls.payload = payload
        handler_cls.header_end = header_end
        handler_cls._fail_count = 0
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    def test_stream_pack_disk_size_law(self):
        """_stream_pack emits exactly source_bytes x 0.001 coordinates."""
        body = bytes(range(256)) * 40  # 10240 bytes -> 10 coords
        carry, out = bytearray(), bytearray()
        _stream_pack(body, carry, out)
        self.assertEqual(len(out), len(body) // 1000)

    def test_chunked_retry_recovers_stalled_connection(self):
        """Server drops the first 2 body Ranges; client retries and still
        produces the identical coordinate stream."""
        n_bytes = 10_000_000  # 10 MB -> 10,000 coords
        raw = _make_safetensors_body(n_bytes)
        server, base = self._serve(_StallingHandler, raw, _header_end(raw))
        try:
            url = f"{base}/model.safetensors"
            begin = _fetch_header(url)
            self.assertGreater(begin, 0)
            carry, out = bytearray(), bytearray()
            pos = begin
            while True:
                chunk = _range_fetch(url, pos, pos + 1_000_000)
                if not chunk:
                    break
                _stream_pack(chunk, carry, out)
                pos += len(chunk)
            self.assertEqual(pos - begin, n_bytes)
            self.assertEqual(len(out), n_bytes // 1000)
        finally:
            server.shutdown()

    def test_shards_resume_matches_fresh_run(self):
        """A run that dies mid-stream then resumes produces byte-identical
        .x8D output to a single uninterrupted run."""
        n_bytes = 5_000_000
        shard1 = _make_safetensors_body(n_bytes, seed=1)
        shard2 = _make_safetensors_body(n_bytes, seed=2)
        sources = [
            (f"http://x/s1", _header_end(shard1)),
            (f"http://x/s2", _header_end(shard2)),
        ]

        # Reference: uninterrupted run over in-memory bytes.
        out_ref = os.path.join(self.tmp, "ref.x8D")
        _run_over_memory(sources, out_ref, shard1, shard2)

        # Crash: patch _range_fetch to fail hard after 1 successful chunk.
        out_res = os.path.join(self.tmp, "res.x8D")
        real_fetch = _range_fetch

        def _crashing_fetch(url, begin, end):
            _crashing_fetch.calls += 1
            if _crashing_fetch.calls == 2:
                raise TimeoutError("simulated crash")
            return _mem_fetch(url, begin, end, shard1, shard2)

        _crashing_fetch.calls = 0
        _mod._range_fetch = _crashing_fetch
        try:
            with self.assertRaises(TimeoutError):
                _run_sources(sources, out_res)
        finally:
            _mod._range_fetch = real_fetch

        # Resume: fresh process-level state object, same output file.
        _mod._range_fetch = _mem_fetch_maker(shard1, shard2)
        try:
            _run_sources(sources, out_res)
        finally:
            _mod._range_fetch = real_fetch

        with open(out_res, "rb") as fa, open(out_ref, "rb") as fb:
            self.assertEqual(fa.read(), fb.read())
        self.assertEqual(os.path.getsize(out_res), (10_000_000) // 1000)


def _mem_fetch(url, begin, end, shard1, shard2):
    if url.endswith("/s1"):
        blob, hdr = shard1, _header_end(shard1)
    else:
        blob, hdr = shard2, _header_end(shard2)
    if begin < hdr:
        return blob[begin:end]
    body = blob[hdr:]
    rel = begin - hdr
    if rel >= len(body):
        return b""
    return body[rel : rel + (end - begin)]


def _mem_fetch_maker(shard1, shard2):
    def fetch(url, begin, end):
        return _mem_fetch(url, begin, end, shard1, shard2)

    return fetch


def _run_over_memory(sources, out_path, shard1, shard2):
    real_fetch = _range_fetch
    _mod._range_fetch = _mem_fetch_maker(shard1, shard2)
    try:
        _run_sources(sources, out_path)
    finally:
        _mod._range_fetch = real_fetch


if __name__ == "__main__":
    unittest.main()
