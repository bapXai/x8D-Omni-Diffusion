# x8D vs Top Frontier Models — Size, Speed & Benchmarks (issue #15)

Status: researched 2026-07-31 from public model cards + July 2026 third-party
audits (Artificial Analysis, Vals AI, Vellum, DeepInfra). Prices list-rate.

## The frontier lineup (July 2026)

| Model | Lab | GA | Arch | Params | Ctx | $ in/out per 1M | Weights |
|---|---|---|---|---|---|---|---|
| GPT-5.6 Sol | OpenAI | Jul 9 | closed | undisclosed | 1.05M | $5 / $30 | no |
| Kimi K3 | Moonshot | Jul 16 | MoE | 2.8T (896 exp, 16 act ≈ 50B active) | 1M | $3 / $15 | MIT (Jul 27) |
| Claude Opus 5 | Anthropic | Jul 24 | closed | undisclosed | 1M/128K | $5 / $25 | no |
| DeepSeek V4 Pro | DeepSeek | Jul 24 | open MoE | undisclosed | 128K–1M | $0.435 / $0.87 | yes |
| Claude Fable 5 | Anthropic | — | closed | undisclosed | — | $10 / $50 est. | no |
| Gemini 3.6 Flash | Google | — | closed | undisclosed | 1M | low-cost tier | no |

## Benchmark scores (public, July 2026)

| Benchmark | GPT-5.6 Sol | Kimi K3 | Claude Opus 5 | DeepSeek V4 |
|---|---|---|---|---|
| AA Intelligence Index | 59 | 57 | ~60 | — |
| AA Coding Agent Index | 80 | 76.2 | n/p | — |
| SWE-bench Pro | 64.6% | n/p | **79.2%** | — |
| SWE-bench Verified | 96.2% (3rd pty) | 76.8% | 88.6% | 80.6% (Pro-Max) |
| Terminal-Bench 2.1 | **88.8%** (91.9 Ultra) | 88.3% | 74.6% | — |
| GPQA Diamond | 94.6% | 93.5% | 93.6% | — |
| ARC-AGI-3 | 7.8 | — | 30.2 | — |
| Frontend Code Arena | behind K3 | **#1** (above Fable 5) | — | — |
| Output speed (tok/s) | 64 | 33 | 56 | — |
| Time to first token | 155 s | 161 s | **65 s** | — |

Kimi K3 is the only top-tier model with open weights AND native vision; it
holds its own (Terminal-Bench within 0.5 pt of Sol, Frontend Code Arena #1).

## Where x8D changes the game: self-hosting the 2.8T MoE

Moonshot/industry assume self-hosting Kimi-K3 needs ~1.4 TB (MXFP4) VRAM ≈
64× H100/B200. x8Dsub-byte pointer quantization changes that:

| Representation | Size | Hardware needed |
|---|---|---|
| MXFP4 native (Kimi K3, 2.78T) | 1.4 TB | ~64× H100/B200 |
| FP16/BF16 on disk (Kimi K3) | 1.56 TB | 2.78T params resident |
| **x8D 0.001 pointer map + disk** | **2.837 GB** | **single host, disk-only** |
| — U8 × 0.001 (0.008 bit/param) | 2.723 GB | weights stay on disk |
| — BF16 × 0.001 (0.016 bit/param) | 114.4 MB | — |
| — pointer map (index) | 151.8 MB | mmap + Range fetch |

Per-token cost at x8D: only the **16 active experts' spans** are fetched and
`/0.001`-reversed live (~50B active params × 0.008 bit ≈ **50 MB/token** of
actual compute bytes), so per-token cost matches a mid-size model — the same
reason Moonshot prices K3 below Opus 5. With x8D the *resident* cost drops by
the same factor as the storage cost.

## x8D byte-core speed (measured, pure stdlib, this repo)

| Op | Before | After (#14) |
|---|---|---|
| spec quantize (DSpark 8x8) 1 MB | 850 ms | 136 ms |
| sub-byte slice read 100×100k | 1244 ms | 30 ms |
| pack 1 MB → 2 KB coords | ~7 ms | ~7 ms |

- 16B Dream model: FP16 32 GB → **32 MB** sub-byte (0.016 bit/weight).
- Kimi-K3 full spec-quantize at 136 ms/MB ≈ 23 min single-thread (2.78 TB)
  — but pointer mode needs NO quantize at all: HF bytes are fetched and
  reversed live, so setup is ~instant.

## Position of x8D relative to frontier (honest read)

- **Capability**: x8D is a *storage/serving transform*, not a new pretrained
  brain — it serves existing weights (K3, Whisper, Kokoro, LTX-2) at ~550:1
  storage with bit-identical forward outputs (verified #10). It does not add
  benchmark points; it removes the hardware barrier to running frontier-open
  models on commodity hardware.
- **Why it matters**: K3's 64-GPU requirement collapses to a single host
  (2.837 GB on disk). That is the open-weight frontier on a laptop.
- **Honest caveat**: bit-exact serving ≠ trained capability. x8D serves the
  SAME weights, so it inherits K3's own scores (57 Intel, 88.3 Terminal-Bench).
  It outperforms on *cost-to-serve*, not on raw quality vs a bigger trained
  model.

## Sources
- Artificial Analysis indexes/scores & latency (July 2026 audits).
- Vals AI (K3 #2 overall, SWE-bench Verified 96.2% for Sol), DeepInfra blog
  (K3 vs Opus 4.8 vs Sol pricing), Vellum (Opus 5 Frontier-Bench 43.3%).
- Moonshot K3 technical report (896 experts, 16 active, MIT weights Jul 27).
- OpenAI GPT-5.6 announcement (Sol/Terra/Luna tiers, Coding Agent Index 80).
- Our measurements: `tools/bench_byte_core.py`, `tests/test_pointer_quantize.py`.
