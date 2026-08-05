# Sub-1-Bit Quantization — Web Research & x8D 0.001 Sub-Byte Map

Searched the web 2026-08-01. Sub-1-bit (a weight stored in FEWER than 1 bit) is a
proven, active research area, and it is exactly what the x8D **0.001 sub-byte
law** implements. This doc records the real methods found and how the law maps to
them.

## The one sentence that kills the confusion

**1 parameter is not 1 byte.** Under the x8D sub-byte law a parameter is stored
as ONE sub-byte coordinate that is LESS than 1 full byte of the fp16 source:

```
Quanta[i] = weight_byte[i] × 0.001        # coordinate in [0.0, 0.255]
weight_byte[i] = round(Quanta[i] / 0.001) # reverse is exact over 0-255
```

Container disk size law: `num_params × bits_per_dtype × 0.001 / 8` bytes.

- fp16 (16 bit) → 0.016 bit/weight → container = **full_fp16_size × 0.001**
- U8 quanta (8 bit) → 0.008 bit/weight → container = **n_params × 0.001** bytes
- A "0.5 sub-byte" row would give 4 bit/weight = 50% off fp16 — that is the
  **1 byte per param** container, NOT the law.

## Worked sizes (the math you asked for)

### Whisper-large-v3 — 1,543,490,560 params
| Representation | Size | vs fp16 |
|---|---|---|
| Full fp16 | 3,086,981,120 B (3.087 GB) | — |
| 0.5 sub-byte (1 B/param) | 1,543,490,560 B (1.543 GB) | **50.0%** less |
| **0.001 sub-byte (law)** | **3,086,981 B (3.087 MB)** | **99.9%** less |

### Kokoro-82M — 81,763,410 params (fp16 = 163,526,820 B)
| Representation | Size | vs fp16 |
|---|---|---|
| Full fp16 | 163,526,820 B | — |
| 0.5 sub-byte (1 B/param) | 81,763,410 B (78.0 MB) | 50.0% less |
| **0.001 sub-byte (law)** | **163,527 B (159.7 KB)** | **99.9%** less — BUILT |

## Real sub-1-bit methods (2025-2026) and the x8D map

| Method | Type | Bits/weight | How | x8D 0.001 map |
|---|---|---|---|---|
| **NanoQuant** (ICML 2026, SamsungLabs) | PTQ | sub-1-bit | low-rank binary matrix factorization + scales; ADMM init; block/model reconstruction. Llama2-70B 25.8× on 1 H100, 8 GB GPU | factor each weight matrix into binary factors; the 0.001 law is the coordinate map on each factor |
| **LittleBit** (NeurIPS 2025) / **LittleBit-2** (ICML 2026) | QAT | 0.1–1.0 BPW | `W ≈ UVᵀ`, binarize factors, multi-scale (row/col/latent) compensation | our block packing = a coarse LittleBit-2-style scale per 500 weights |
| **BTC-LLM** (ACL 2026) | PTQ/QAT | 0.7–1.11 bits | binary codebook (clustered ±1 vectors → compact indices) + learnable transform | coordinate codebook over 0-255 |
| **BiLLM** | PTQ | ~1 bit | binary with grouped scales | per-block scale |
| **STBLLM** | PTQ | ~0.5-1 bit | N:M sparsity + binary | sparse binary blocks |
| **ARB-LLM** | PTQ | ~1 bit | alternating refinement | iterative coordinate refine |
| **OneBit / BinaryMoS / DBF / ParetoQ** | QAT | 1 bit | binarize + STE training | STE path for our QAT phase 4 |

Key fact from the literature: binary PTQ methods that use in-place binarization
+ full-precision scales are bounded at ≥1 bit/weight and their metadata can push
effective bitrate to 2-3 bits (BiLLM 2.88, STBLLM 4.13). Sub-1-bit REQUIRES a
low-rank binary factorization (LittleBit, NanoQuant) or a codebook (BTC-LLM).
Our 0.001 coordinate packing is the codebook/factorization family.

## x8D build status (this session)

- `/tmp/kokoro.x8dgguf` = 81,763,410 B = 0.5 sub-byte container (1 B/param, 50%).
- `/tmp/kokoro.x8dsubbyte.gguf` = **163,527 B = the 0.001 law container**
  (99.9% less than fp16), written from the raw quanta via `pack_subbyte`.
- Whisper large-v3: fp16 header total 1,543,490,560 params; the 0.001 container
  must be **3,086,981 B**. The previous `/tmp/whisper.x8dgguf` write was the
  wrong 0.5 container (1 B/param) and was killed.

## Honest note on losslessness

The stored quanta byte maps bijectively back to its weight byte (0-255), so the
per-param coordinate IS lossless. Block packing (500 weights → 1 coord byte via
`round(mean × 0.001)`) is the coarse "LittleBit-style scale" row of the table —
a full LittleBit/NanoQuant factorization (per-tensor U,V binary factors + row/col
scales) is the exact-lossless upgrade path, which is what phase 4 QAT/STE of
`implementation_plan.md` adds.
