# Byte-Core Optimizations & Benchmarks (issue #14)

Status: measured 2026-07-31, macOS, pure stdlib (no torch), Python 3.14.
Run: `python3 tools/bench_byte_core.py`.

## Baseline → After

| Hot path | Before | After | Speedup |
|---|---|---|---|
| `speculative_quantize` (1 MB) | 850 ms | 136 ms | **6.3×** |
| `SubByteModel.weights` 100×100k slice | 1244 ms | 30 ms | **41×** |
| `pack_subbyte` (1 MB) | ~7 ms | ~7 ms | unchanged |
| `unpack_subbyte` (1 MB) | ~4 ms | ~4 ms | unchanged |

## What changed

### 1. `x8d_spec_decode.py` — one hash per block, not per position
`_block_surrogate` is a sha256 of the 8x8 block (same value for all 64
positions). It was computed inside a per-position list comprehension —
64 sha256 per block. Now computed once per block and reused:
```python
block_conf = float(_block_surrogate(current, step))
confidence = [(block_conf + float(b) / 256.0) / 2.0 for b in current]
```
Identical math (the surrogate does not depend on position), 64x fewer
hashing calls.

### 2. `x8d_subbyte.py` — C-speed slice reads
`SubByteModel.weights()` rebuilt the inverse pointer map per element with
`round(coord*0.001/LAW)`. Two C-level tricks replace it:
- `_WEIGHT_LUT`: the 256-entry inverse map precomputed once as a `tuple`;
- `bytes.translate(lut_bytes)`: maps coordinate bytes → running weight bytes
  in C; the per-block repeat + head/tail trim happen after.

Verified edge-exact against `weight_at` for: boundary (499/500), coord
boundary (500/501), mid, tail, single-byte, full-span slices.

## Correctness
- `tests/test_subbyte.py` (8) + `tests/test_spec_decode.py` (11) pass.
- Full suite: 78 tests OK (3 torch-skipped).

## Scaling notes
- 16B-param model: FP16 32.00 GB → x8D sub-byte 32.0 MB (0.016 bit/weight).
- Spec-decode storage: 1 MB → 2 KB coordinate map (500 w/byte); spec
  quantize of the full 2.78T Kimi-K3 at 136 ms/MB ≈ 23 min single-thread.
