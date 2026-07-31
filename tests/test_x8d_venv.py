# coding=utf-8
"""Tests for the x8D SandboxComput.bin compressed venv (issue #28)."""

import importlib
import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_venv import (  # noqa: E402
    LAW,
    VENV_MAGIC,
    VENV_VERSION,
    SandboxComput,
    SandboxComputError,
    compress_venv,
    install_venv_hook,
    uninstall_venv_hook,
)


def _write(src_dir, rel_path, data):
    full = os.path.join(src_dir, *rel_path.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)


class CompressRoundtripTest(unittest.TestCase):
    def _tmpdir(self):
        path = tempfile.mkdtemp(prefix="x8dvenv_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def _compress(self, src_dir, include=None):
        out = os.path.join(src_dir, "SandboxComput.bin")
        manifest = compress_venv(src_dir, out, include=include)
        self.addCleanup(os.remove, out)
        return out, manifest

    def test_roundtrip_byte_exact(self):
        src = self._tmpdir()
        files = {
            "pkg/sub/empty.bin": b"",
            "pkg/allbytes.bin": bytes(range(256)),
            "pkg/sub/utf8.txt": "नमस्ते".encode("utf-8"),
            "pkg/random.bin": os.urandom(4096),
            "pkg/random_big.bin": os.urandom(65536),
            "top.py": b"x = 1\n",
        }
        for rel, data in files.items():
            _write(src, rel, data)
        out, _ = self._compress(src)
        box = SandboxComput(out)
        try:
            self.assertEqual(sorted(box.paths()), sorted(files))
            for rel, data in files.items():
                self.assertEqual(box.read_file(rel), data)
        finally:
            box.close()

    def test_missing_file_raises(self):
        src = self._tmpdir()
        _write(src, "a.py", b"x = 1\n")
        out, _ = self._compress(src)
        box = SandboxComput(out)
        try:
            with self.assertRaises(SandboxComputError):
                box.read_file("not-there.py")
        finally:
            box.close()

    def test_quanta_for_applies_law_at_read_time(self):
        src = self._tmpdir()
        data = bytes(range(256))
        _write(src, "w.bin", data)
        out, _ = self._compress(src)
        box = SandboxComput(out)
        try:
            self.assertEqual(box.quanta_for("w.bin"), [b * LAW for b in data])
        finally:
            box.close()

    def test_manifest_keys(self):
        src = self._tmpdir()
        _write(src, "a.py", b"a" * 100)
        _write(src, "b.py", b"b" * 200)
        out, manifest = self._compress(src)
        for key in ("files", "total_bytes", "compressed_bytes", "file_count"):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["file_count"], 2)
        self.assertEqual(manifest["total_bytes"], 300)
        self.assertEqual(manifest["compressed_bytes"], os.path.getsize(out))
        self.assertTrue(manifest["lossless"])
        self.assertEqual(manifest["container"], "SandboxComput")

    def test_header_magic_and_version(self):
        src = self._tmpdir()
        _write(src, "a.py", b"x")
        out, _ = self._compress(src)
        with open(out, "rb") as f:
            header = f.read(struct.calcsize("<8sBQ"))
        magic, version, count = struct.unpack("<8sBQ", header)
        self.assertEqual(magic, VENV_MAGIC)
        self.assertEqual(version, VENV_VERSION)
        self.assertEqual(count, 1)

    def test_empty_dir_compresses(self):
        src = self._tmpdir()
        out, manifest = self._compress(src)
        self.assertEqual(manifest["file_count"], 0)
        box = SandboxComput(out)
        try:
            self.assertEqual(box.paths(), [])
        finally:
            box.close()

    def test_include_filter(self):
        src = self._tmpdir()
        _write(src, "keep.py", b"k")
        _write(src, "skip.py", b"s")
        _write(src, "pkg/keep.py", b"k")
        out, manifest = self._compress(src, include=["keep.py", "pkg/*"])
        self.assertEqual(
            manifest["files"], ["keep.py", "pkg/keep.py"]
        )

    def test_bad_magic_raises(self):
        src = self._tmpdir()
        bad = os.path.join(src, "bad.bin")
        with open(bad, "wb") as f:
            f.write(b"NOTX8DVENV")
        with self.assertRaises(SandboxComputError):
            SandboxComput(bad)


class SandboxHookTest(unittest.TestCase):
    def _tmpdir(self):
        path = tempfile.mkdtemp(prefix="x8dvenv_hook_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def test_import_module_from_sandbox(self):
        src = self._tmpdir()
        _write(src, "sandbox/__init__.py", b'"""sandbox package"""\n')
        _write(src, "sandbox/pkg/__init__.py", b'"""pkg"""\n')
        _write(
            src,
            "sandbox/pkg/mathutil.py",
            b"def add(a, b):\n    return a + b\n\n"
            b"SCALE = 0.001\n",
        )
        out = os.path.join(src, "SandboxComput.bin")
        compress_venv(src, out)
        box = install_venv_hook(out, prefix="sandbox")
        try:
            mod = importlib.import_module("sandbox.pkg.mathutil")
            self.assertEqual(mod.add(2, 3), 5)
            self.assertEqual(mod.SCALE, 0.001)
            self.assertIn("sandbox.pkg.mathutil", sys.modules)
        finally:
            uninstall_venv_hook()
        self.assertNotIn(box, (None,))

    def test_uninstall_removes_hook(self):
        src = self._tmpdir()
        _write(src, "sandbox/__init__.py", b"")
        _write(src, "sandbox/only.py", b"VALUE = 42\n")
        out = os.path.join(src, "SandboxComput.bin")
        compress_venv(src, out)
        install_venv_hook(out, prefix="sandbox")
        try:
            self.assertEqual(importlib.import_module("sandbox.only").VALUE, 42)
        finally:
            uninstall_venv_hook()
        for key in ("sandbox", "sandbox.only"):
            sys.modules.pop(key, None)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("sandbox.only")

    def test_uninstall_is_idempotent(self):
        uninstall_venv_hook()
        uninstall_venv_hook()

    def test_import_missing_module_fails(self):
        src = self._tmpdir()
        _write(src, "sandbox/__init__.py", b"")
        out = os.path.join(src, "SandboxComput.bin")
        compress_venv(src, out)
        install_venv_hook(out, prefix="sandbox")
        try:
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module("sandbox.does_not_exist")
        finally:
            uninstall_venv_hook()


if __name__ == "__main__":
    unittest.main()
