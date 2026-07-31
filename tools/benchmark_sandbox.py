#!/usr/bin/env python3
# coding=utf-8
"""Benchmark: regular venv import vs x8D SandboxComput.bin import (issue #28).

Builds a synthetic venv of pure-Python stdlib modules, then compares:

* **(a) Regular venv** -- the temp dir is put on ``sys.path`` and the modules
  are imported from the filesystem.
* **(b) SandboxComput.bin** -- the same dir is compressed into one container
  and imported through the ``install_venv_hook`` importer, serving bytes from
  the mmap at import time.

Both phases import the SAME dotted names (``sandbox.<mod>``); only the backing
store differs. Pure stdlib (no torch, no network). Run:

    python3 tools/benchmark_sandbox.py [--iterations N] [--modules a,b,...]
"""

import argparse
import importlib
import os
import resource
import shutil
import sys
import sysconfig
import tempfile
import time
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_venv import (  # noqa: E402
    LAW,
    compress_venv,
    install_venv_hook,
    uninstall_venv_hook,
)

PREFIX = "sandbox"

DEFAULT_MODULES = [
    "fractions",
    "heapq",
    "statistics",
    "random",
    "base64",
    "decimal",
    "calendar",
    "uuid",
    "datetime",
    "textwrap",
]


def _compute(mod, name: str):
    """Small per-module computation to exercise the freshly imported module."""
    if name == "fractions":
        total = mod.Fraction(0, 1)
        for i in range(2000):
            total += mod.Fraction(i, 97)
        return total
    if name == "heapq":
        data = [(i * 7919) % 100003 for i in range(20000)]
        mod.heapify(data)
        return [mod.heappop(data) for _ in range(100)][-1]
    if name == "statistics":
        xs = [(i % 997) * 1.5 for i in range(20000)]
        return (mod.mean(xs), mod.variance(xs), mod.pstdev(xs))
    if name == "random":
        return mod.Random(1234).randrange(1000000)
    if name == "base64":
        return mod.b64encode(b"x8d benchmark payload " * 20)
    if name == "decimal":
        mod.getcontext().prec = 40
        total = mod.Decimal(0)
        for i in range(1, 400):
            total += mod.Decimal(1) / mod.Decimal(i)
        return total
    if name == "calendar":
        return mod.calendar(2026)
    if name == "uuid":
        return mod.uuid4()
    if name == "datetime":
        return mod.datetime(2026, 7, 31, 12, 30, 45)
    if name == "textwrap":
        return mod.wrap("The quick brown fox jumps over the lazy dog. " * 8, width=30)
    raise ValueError(f"no computation for module {name!r}")


def rss_mb() -> float:
    """Rough peak RSS in MB (ru_maxrss is a high-water mark)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def _clear_module(prefix: str, name: str) -> None:
    """Drop a module from sys.modules so the next import is a fresh cold load."""
    sys.modules.pop(prefix, None)
    sys.modules.pop(f"{prefix}.{name}", None)


def measure_import(
    prefix: str, name: str
) -> Tuple[float, object]:
    """Time a fresh import + computation; returns (ms, result)."""
    _clear_module(prefix, name)
    t0 = time.perf_counter()
    mod = importlib.import_module(f"{prefix}.{name}")
    result = _compute(mod, name)
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, result


def run_phase(
    prefix: str, modules: List[str], iterations: int
) -> Tuple[Dict[str, List[float]], float]:
    """Run one phase (N iterations) and return per-module ms lists + RSS after."""
    importlib.import_module(prefix)
    times: Dict[str, List[float]] = {m: [] for m in modules}
    for _ in range(iterations):
        for m in modules:
            dt, _result = measure_import(prefix, m)
            times[m].append(dt)
    return times, rss_mb()


def build_venv(tmp: str, modules: List[str]) -> str:
    """Copy pure-Python stdlib modules under ``<tmp>/<prefix>/``."""
    stdlib = sysconfig.get_paths()["stdlib"]
    pkg_dir = os.path.join(tmp, PREFIX)
    os.makedirs(pkg_dir)
    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write('"""Synthetic byte-native venv package for the sandbox benchmark."""\n')
    for m in modules:
        src = os.path.join(stdlib, f"{m}.py")
        shutil.copyfile(src, os.path.join(pkg_dir, f"{m}.py"))
    return pkg_dir


def format_table(
    modules: List[str],
    reg_times: Dict[str, List[float]],
    sand_times: Dict[str, List[float]],
    reg_rss_delta: float,
    sand_rss_delta: float,
) -> str:
    rows = []
    reg_total = 0.0
    sand_total = 0.0
    rows.append(f"{'module':<14} {'regular ms':>12} {'sandbox ms':>12} {'delta ms':>10}")
    rows.append("-" * 50)
    for m in modules:
        r_avg = sum(reg_times[m]) / len(reg_times[m])
        s_avg = sum(sand_times[m]) / len(sand_times[m])
        reg_total += r_avg
        sand_total += s_avg
        rows.append(
            f"{m:<14} {r_avg:>12.3f} {s_avg:>12.3f} {s_avg - r_avg:>+10.3f}"
        )
    rows.append("-" * 50)
    rows.append(f"{'total':<14} {reg_total:>12.3f} {sand_total:>12.3f} {sand_total - reg_total:>+10.3f}")
    rows.append("")
    rows.append(f"peak RSS delta (vs prior phase): regular +{reg_rss_delta:.2f} MB, "
                f"sandbox +{sand_rss_delta:.2f} MB")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="SandboxComput.bin import benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="iterations per phase")
    parser.add_argument(
        "--modules",
        type=str,
        default=",".join(DEFAULT_MODULES),
        help="comma-separated stdlib module names",
    )
    args = parser.parse_args()
    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    iterations = max(1, args.iterations)

    print(f"x8D SandboxComput.bin vs regular venv import benchmark (issue #28)")
    print(f"prefix={PREFIX!r} modules={len(modules)} iterations={iterations}")
    print(f"byte law: Quanta[i] = weight_byte[i] * {LAW}")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="sandboxbench_")
    try:
        build_venv(tmp, modules)

        baseline = rss_mb()
        sys.path.insert(0, tmp)
        reg_times, rss_after_reg = run_phase(PREFIX, modules, iterations)
        sys.path.remove(tmp)
        reg_rss_delta = rss_after_reg - baseline

        bin_path = os.path.join(tmp, "SandboxComput.bin")
        manifest = compress_venv(tmp, bin_path, include=[f"{PREFIX}/*.py"])
        print(f"\ncompressed {manifest['file_count']} files: "
              f"{manifest['total_bytes']} B -> SandboxComput.bin "
              f"({manifest['compressed_bytes']} B, lossless={manifest['lossless']})")

        box = install_venv_hook(bin_path, prefix=PREFIX)
        try:
            sand_times, rss_after_sand = run_phase(PREFIX, modules, iterations)
        finally:
            uninstall_venv_hook()
        sand_rss_delta = rss_after_sand - rss_after_reg

        print("\n" + format_table(modules, reg_times, sand_times, reg_rss_delta, sand_rss_delta))
        print(f"\nbaseline peak RSS: {baseline:.2f} MB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
