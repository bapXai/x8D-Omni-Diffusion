# coding=utf-8
"""Micro-benchmarks for the byte-native core hot paths (issue #14).

Pure stdlib (no torch). Run:  python3 tools/bench_byte_core.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_subbyte import pack_subbyte, unpack_subbyte, save_subbyte_gguf
from omni_diffusion.x8d_export import LAW
from omni_diffusion.x8d_spec_decode import speculative_quantize

MB = 1_000_000


def bench(name, fn, n=3):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    print(f"  {name:<55} {best*1000:9.2f} ms")
    return best


def main():
    print("x8D byte-core benchmarks (pure stdlib, macOS)")
    print("=" * 72)

    # 1 MB of weight bytes
    weights = bytes((i * 31 + 7) & 0xFF for i in range(MB))

    print("\n[pack_subbyte] 1 MB -> 2 KB coordinate map (500 w/byte)")
    bench("pack 1 MB weights", lambda: pack_subbyte(weights))

    packed = pack_subbyte(weights)
    print("\n[unpack_subbyte] 2 KB -> 1 MB running weights")
    bench("unpack 1 MB (500-block repeat)", lambda: unpack_subbyte(packed, len(weights)))

    print("\n[speculative_quantize] 1 MB weights")
    bench("spec quantize 1 MB", lambda: speculative_quantize(weights))

    print("\n[SubByteModel] mmap + live /0.001 weight lookups")
    tmp = "/tmp/bench_subbyte.gguf"
    save_subbyte_gguf("bench", weights, tmp)
    from omni_diffusion.x8d_subbyte import SubByteModel

    m = SubByteModel(tmp)

    def slice_read():
        for _ in range(100):
            m.weights(0, 100000)

    bench("100x slice of 100k weights", slice_read)
    m.close()

    # 16B model scaling numbers
    print("\n[scaling] 16B-param model sizes")
    from omni_diffusion.x8d_subbyte import size_report_subbyte
    r = size_report_subbyte()
    print(f"  FP16 full : {r['full_precision_gb']:.2f} GB")
    print(f"  x8D sub-byte: {r['subbyte_mb']:.1f} MB ({r['bits_per_weight']} bit/weight)")


if __name__ == "__main__":
    main()
