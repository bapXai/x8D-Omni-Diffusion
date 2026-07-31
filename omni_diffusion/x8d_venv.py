# coding=utf-8
"""x8D-compressed Python environment: SandboxComput.bin (issue #28).

Compresses a directory tree (typically a venv's ``site-packages``) into a
single lossless ``SandboxComput.bin`` container, then serves it back through
an mmap at import time -- the "compressed state IS the running state" law.

The byte law is respected end to end:

* **Storage** -- file bytes are written to the container **as-is**, never
  rounded and never scaled. ``SandboxComput.bin`` is byte-lossless.
* **Compute** -- the 0.001 sub-byte reversal happens only at read time:
  ``quanta_for()`` maps a stored file's bytes into sub-byte coordinates
  (``byte * 0.001``) on demand, the same live inverse used by
  ``x8d_export.quantize``.

Container layout (single file):

* Header: ``<8sBQ`` -- magic ``X8DVENV1``, version byte, u64 file count.
* Index (one entry per file): u16 path_len, UTF-8 relative path,
  u64 offset into the blob, u64 length.
* Blob: the concatenated raw bytes of every file, stored byte-for-byte.

Pure Python standard library only (``mmap``, ``os``, ``struct``,
``importlib``, ``fnmatch``) -- no torch, no virtualenv, no requests.
"""

from __future__ import annotations

import fnmatch
import importlib.abc
import importlib.util
import mmap
import os
import shutil
import struct
import sys
from typing import Dict, List, Optional, Tuple

from .x8d_export import LAW, quantize

#: Container magic (8 bytes) + version byte = "X8DVENV1" + b"\\x01".
VENV_MAGIC: bytes = b"X8DVENV1"

#: Container version byte.
VENV_VERSION: int = 1

#: Header: magic (8s) + version (B) + file count (Q).
_HEADER_FMT = "<8sBQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class SandboxComputError(ValueError):
    """Raised when a file is missing from, or a path is not a valid,
    SandboxComput container."""


def _included(rel_path: str, patterns: Optional[List[str]]) -> bool:
    """True when ``rel_path`` matches one of the fnmatch patterns.

    An empty/None pattern list includes everything. Exact relative paths are
    also accepted alongside glob patterns.
    """
    if not patterns:
        return True
    return any(rel_path == p or fnmatch.fnmatch(rel_path, p) for p in patterns)


def _collect_files(
    src_dir: str,
    include: Optional[List[str]],
    follow_symlinks: bool,
) -> List[str]:
    """Collect deterministic, sorted relative paths of files under ``src_dir``.

    Symlinked files/dirs are skipped unless ``follow_symlinks`` is True.
    Paths use ``/`` separators and are sorted for reproducible containers.
    """
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(src_dir, followlinks=follow_symlinks):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full) and not follow_symlinks:
                continue
            rel = os.path.relpath(full, src_dir).replace(os.sep, "/")
            if _included(rel, include):
                out.append(rel)
    return sorted(out)


def _path_join(src_dir: str, rel_path: str) -> str:
    """Join a ``/``-separated relative path onto a base dir portably."""
    return os.path.join(src_dir, *rel_path.split("/"))


def compress_venv(
    src_dir: str,
    out_bin: str,
    include: Optional[List[str]] = None,
    follow_symlinks: bool = False,
) -> Dict:
    """Compress a directory tree into a single lossless SandboxComput.bin.

    Args:
        src_dir: directory to walk (e.g. a venv ``site-packages``).
        out_bin: output path for the ``.bin`` container.
        include: optional fnmatch patterns on relative paths; None = all.
        follow_symlinks: descend into symlinked dirs / include symlinked files.

    Returns:
        Manifest dict with ``files``, ``total_bytes``, ``compressed_bytes``,
        ``file_count``, ``container``, ``magic`` and ``version`` keys.

    Raises:
        SandboxComputError: ``src_dir`` does not exist or is not a directory.
    """
    if not os.path.isdir(src_dir):
        raise SandboxComputError(f"not a directory: {src_dir!r}")
    paths = _collect_files(src_dir, include, follow_symlinks)
    sizes = {p: os.path.getsize(_path_join(src_dir, p)) for p in paths}
    total_bytes = sum(sizes.values())

    offset = 0
    entries: List[Tuple[str, int, int]] = []
    for p in paths:
        entries.append((p, offset, sizes[p]))
        offset += sizes[p]

    with open(out_bin, "wb") as f:
        f.write(struct.pack(_HEADER_FMT, VENV_MAGIC, VENV_VERSION, len(entries)))
        for p, off, length in entries:
            pb = p.encode("utf-8")
            f.write(struct.pack("<H", len(pb)))
            f.write(pb)
            f.write(struct.pack("<QQ", off, length))
        for p, _off, _length in entries:
            with open(_path_join(src_dir, p), "rb") as src:
                shutil.copyfileobj(src, f)

    return {
        "container": "SandboxComput",
        "magic": VENV_MAGIC.decode("ascii"),
        "version": VENV_VERSION,
        "files": paths,
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "compressed_bytes": os.path.getsize(out_bin),
        "lossless": True,
    }


class SandboxComput:
    """mmap-based reader over a SandboxComput.bin container.

    The index is parsed once at open time; file bytes are pulled from the
    memory map only when ``read_file`` (or ``quanta_for``) is called. This is
    the byte-law serving behavior: the compressed state IS the running state
    and the 0.001 reversal happens at read time, never at storage time.
    """

    def __init__(self, bin_path: str):
        self.path = bin_path
        size = os.path.getsize(bin_path)
        fd = os.open(bin_path, os.O_RDONLY)
        try:
            self._mmap: Optional[mmap.mmap] = mmap.mmap(
                fd, size, access=mmap.ACCESS_READ
            )
        finally:
            os.close(fd)
        mapping = self._mmap
        if mapping[: len(VENV_MAGIC)] != VENV_MAGIC:
            raise SandboxComputError(
                f"not a SandboxComput container (magic {mapping[:8]!r})"
            )
        _magic, self.version, count = struct.unpack_from(_HEADER_FMT, mapping, 0)
        pos = _HEADER_SIZE
        self._index: Dict[str, Tuple[int, int]] = {}
        for _ in range(count):
            (path_len,) = struct.unpack_from("<H", mapping, pos)
            pos += 2
            path = bytes(mapping[pos : pos + path_len]).decode("utf-8")
            pos += path_len
            off, length = struct.unpack_from("<QQ", mapping, pos)
            pos += 16
            self._index[path] = (off, length)
        self._blob_start = pos
        self._paths = sorted(self._index)

    def paths(self) -> List[str]:
        """Sorted list of relative file paths stored in the container."""
        return list(self._paths)

    def has_file(self, path: str) -> bool:
        """True if ``path`` is stored in the container."""
        return path in self._index

    def __contains__(self, path: str) -> bool:
        return self.has_file(path)

    def read_file(self, path: str) -> bytes:
        """Slice one file's raw bytes out of the memory map (lazy).

        Args:
            path: relative path exactly as stored in the container.

        Returns:
            The file's bytes, byte-for-byte identical to the source file.

        Raises:
            SandboxComputError: the path is not in the container.
        """
        entry = self._index.get(path)
        if entry is None:
            raise SandboxComputError(f"file not in sandbox: {path!r}")
        off, length = entry
        mapping = self._mmap
        if mapping is None:
            raise SandboxComputError(f"sandbox closed: {path!r}")
        start = self._blob_start + off
        return bytes(mapping[start : start + length])

    def quanta_for(self, path: str) -> List[float]:
        """Compute-time /0.001 reversal: sub-byte coordinates for a file.

        ``quanta_for(p) == [b * 0.001 for b in read_file(p)]`` -- the bytes
        stay raw in the container; only this read-time access applies the law.
        """
        return quantize(self.read_file(path))

    def close(self) -> None:
        """Release the memory map. Safe to call more than once."""
        mapping = self._mmap
        if mapping is not None:
            try:
                mapping.close()
            except (BufferError, ValueError):
                pass
            self._mmap = None


class _SandboxLoader(importlib.abc.Loader):
    """Loader that execs a module's source straight off the mmap."""

    def __init__(self, box: SandboxComput, bin_path: str, is_package: bool):
        self._box = box
        self._bin_path = bin_path
        self._is_package = is_package

    def create_module(self, spec) -> None:
        return None

    def exec_module(self, module) -> None:
        source = self._box.read_file(self._bin_path)
        code = compile(source, self._bin_path, "exec")
        module.__file__ = f"<{self._box.path}>:{self._bin_path}"
        exec(code, module.__dict__)


def _bin_candidates(fullname: str, prefix: str) -> List[str]:
    """Container paths that could back the dotted module ``fullname``."""
    if fullname == prefix:
        return [f"{prefix}/__init__.py"]
    if fullname.startswith(prefix + "."):
        rel = fullname[len(prefix) + 1 :].replace(".", "/")
        return [f"{prefix}/{rel}.py", f"{prefix}/{rel}/__init__.py"]
    return []


class _SandboxFinder(importlib.abc.MetaPathFinder):
    """sys.meta_path finder serving ``<prefix>/<dotted>.py`` from the .bin."""

    def __init__(self, box: SandboxComput, prefix: str):
        self._box = box
        self._prefix = prefix

    def find_spec(self, fullname, path=None, target=None):
        for bin_path in _bin_candidates(fullname, self._prefix):
            if self._box.has_file(bin_path):
                is_pkg = bin_path.endswith("/__init__.py")
                loader = _SandboxLoader(self._box, bin_path, is_pkg)
                return importlib.util.spec_from_loader(
                    fullname, loader, is_package=is_pkg
                )
        return None


#: Module-level state so ``uninstall_venv_hook`` can restore sys.meta_path.
_HOOK_BOX: Optional[SandboxComput] = None
_HOOK_FINDER: Optional[_SandboxFinder] = None
_HOOK_PREFIX: Optional[str] = None


def install_venv_hook(bin_path: str, prefix: str = "sandbox") -> SandboxComput:
    """Install a sys.meta_path importer backed by a SandboxComput.bin.

    Modules whose dotted paths exist as ``<prefix>/<dotted>.py`` in the
    container become importable from the compressed environment; their source
    is served byte-lazily from the mmap at import time.

    Args:
        bin_path: path to a SandboxComput.bin container.
        prefix: import prefix (default ``sandbox``), also the top-level
            directory name used inside the container.

    Returns:
        The backing :class:`SandboxComput` reader (mmap left open until
        ``uninstall_venv_hook`` or ``close``).
    """
    global _HOOK_BOX, _HOOK_FINDER, _HOOK_PREFIX
    uninstall_venv_hook()
    box = SandboxComput(bin_path)
    finder = _SandboxFinder(box, prefix)
    sys.meta_path.insert(0, finder)
    _HOOK_BOX, _HOOK_FINDER, _HOOK_PREFIX = box, finder, prefix
    return box


def uninstall_venv_hook() -> None:
    """Remove the venv importer from sys.meta_path and close its mmap."""
    global _HOOK_BOX, _HOOK_FINDER, _HOOK_PREFIX
    finder = _HOOK_FINDER
    if finder is not None and finder in sys.meta_path:
        sys.meta_path.remove(finder)
    if _HOOK_BOX is not None:
        _HOOK_BOX.close()
    _HOOK_BOX, _HOOK_FINDER, _HOOK_PREFIX = None, None, None
