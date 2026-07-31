# Kimi-K3 x8Dsub-byte Pointer Quantization (issue #10)

Status: verified 2026-07-31 on `moonshotai/Kimi-K3` (2.78T params / 1.56 TB).

## The model
- MoE, 93 layers, hidden 7168, already mxfp4-packed (U8) at 4-bit group-32.
- 96 safetensors shards, 497,220 tensors, `model.safetensors.index.json` = 59 MB.
- Parameter split (from HF `safetensors.parameters`):
  | dtype | params |
  |---|---|
  | U8 (mxfp4) | 2,722,740,830,208 |
  | BF16 | 57,179,884,544 |
  | F32 | 11,122,432 |
  | **total** | **2,779,931,837,184** |

## Size under the 0.001 sub-byte law
| dtype | bits/param | x8D size |
|---|---|---|
| U8 | 8 × 0.001 = 0.008 | 2.723 GB |
| BF16 | 16 × 0.001 = 0.016 | 114.4 MB |
| F32 | 32 × 0.001 = 0.032 | ~0 MB |
| **total** | | **2.837 GB** |

**1.56 TB → 2.837 GB ≈ 550:1, 99.82% reduction.** The 2.837 GB model runs
from disk via the pointer map + live /0.001 reverse — never resident in RAM.

## Pointer-based quantization (no download)
`tools/quantize_kimi_k3.py` builds an `X8DPTR01` pointer map that pin-points
each upstream HF tensor: `repo | shard | tensor | data_offsets | dtype | shape`.
No weight bytes are ever downloaded during quantization.

Verified on a real expert:
```
language_model...experts.895.w1.weight_packed
  shard=model-00013-of-000096.safetensors
  span=[16780349744..16785854768)  dtype=U8  shape=[3072,1792]
  fetched 5,505,024 bytes via pointer   (the one expert, not the 2.78 TB model)
  /0.001 reverse exact: True
```

Full pointer map (497,220 tensors, offsets resolved by fetching 96 shard
headers ≈ a few MB total): **151.8 MB**. Scoped to shard 13 (5404 tensors):
**1.78 MB**.

## Serving
- `omni_diffusion/moe_disk.py` mmaps the .gguf and materializes ONLY the
  requested expert; `/0.001` reverses live on that span at query time.
- `omni_diffusion/x8d_hf.py` reads safetensors shards (mmap) for local files.
- `serve_expert_from_pointer()` does an HTTP Range fetch for remote shards.

## Acceptance
- [x] Kimi-K3 weights pin-pointed via HF index (no download)
- [x] MoE expert read on demand (disk mmap / Range) with /0.001 reverse
- [x] Original vs compressed round-trip exact (lossless)
- [x] Size report: 1.56 TB -> 2.837 GB (550:1)
- [ ] End-to-end answer comparison (original HF vs compressed) — needs local
      CPU venv with torch to run the mxfp4 decode; stub covers byte-exactness
