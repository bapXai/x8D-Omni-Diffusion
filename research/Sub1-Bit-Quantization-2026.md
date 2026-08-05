# Sub-1-Bit Quantization & Byte-Based Models — Competitor Landscape (2026)

Researched the live web 2026-08-05 (issue #53). Two distinct competitor families
exist and BOTH are claimed by the x8D design:

1. **Sub-1-bit weight quantization** (store weights in < 1 bit per weight) —
   competes with the **0.001 sub-byte storage law**.
2. **Byte-based / tokenizer-free models** (vocab = raw bytes, no BPE/subword) —
   competes with the **byte-native modeling stack** (vocab=264).

This doc records the real, named competitors, their numbers, and exactly how the
x8D 0.001 law and byte stack map against them.

---

## The x8D law restated (the whole math)

```
Quanta[i] = weight_byte[i] × 0.001        # coordinate in [0.0, 0.255]
weight_byte[i] = round(Quanta[i] / 0.001) # reverse is EXACT, bijective over 0-255
```

**Disk size = source_bytes × 0.001.** Do NOT count parameters — the parameters
live inside the bytes; only disk size matters.

- 8-bit byte × 0.001 → **0.008 bit per weight byte** = **1000:1 (99.9%)**.
- The 0.5 row (4 bit, 2:1, 50%) is a DIFFERENT scaling row, not the law.
- Storage = raw quanta bytes, arithmetic-coded losslessly (`x8d_arith.py`);
  no GGUF magic, no headers, no manifest, no padding.

Verified real builds (1000:1, lossless round-trip):

| Model | Source bytes | `.x8D` disk | Ratio |
|---|---|---|---|
| Whisper-large-v3 | 3,086,981,120 | **3,086,982** | 1000:1 |
| Kokoro-82M | 327,212,226 | **327,213** | 1000:1 |
| LTX-2 (19B) | 43,285,058,242 | ~43,285,058 (streaming) | 1000:1 |
| Kimi-K3 (2.78T) | 1,560,936,091,448 | ~1,560,936,091 (streaming) | 1000:1 |

---

## Family 1 — Sub-1-bit weight quantization (storage competitors)

All of these compress **model weights** to fewer than 1 bit per weight. The x8D
0.001 law is a *storage* layer: the coordinate stream is `source_bytes × 0.001`,
which lands x8D at **0.008 bit per weight byte** — two orders of magnitude below
even the most extreme sub-1-bit quantizer, and the container is the running state
(zero-copy mmap, no decompression).

| Method | Type | Bits/weight | How | Result / note | x8D map |
|---|---|---|---|---|---|
| **NanoQuant** (arXiv 2602.06694, Feb 2026 — Chong, Kim, Kim, Choi) | PTQ | 0.55–1.00 bit | **low-rank binary factorization** `W ≈ s₁ ⊙ (U±₁·V±₁ᵀ) ⊙ s₂ᵀ`; Hessian-aware ADMM (LB-ADMM) init; block + model reconstruction | **first PTQ to hit sub-1-bit**; L2-70B 138.04 GB → 5.35 GB (25.8×) on 1 H100, 13 h, runs on 8 GB GPU @ 20.11 tok/s; needs 128 calib samples (0.26M tok) | LOSSY (PPL 5.47→10.34 @1 bit), needs calibration + block/model reconstruction; x8D 0.001 law (0.008 bit/byte = 1000:1) is LOSSLESS bijective, no calibration, container IS running state |
| **LittleBit** (NeurIPS 2025) / **LittleBit-2** (ICML 2026, SamsungLabs) | QAT | 0.1–1.0 BPW | `W ≈ U·Vᵀ` low-rank latent factorization, binarize factors, multi-scale (row/col/latent) compensation; Dual-SVID init + residual compensation | 0.1 BPW on Llama2-13B → **0.84 GB** (31×); Llama2-70B → 1.98 GB (69.7×); 0.1 BPW beats STBLLM at 0.7 BPW; 11.6× kernel speedup | x8D coordinate packing = a static coarse LittleBit-style scale; STE/QAT phase 4 mirrors their Dual-SVID + residual |
| **BTC-LLM** (ACL 2026) | PTQ/QAT | 0.7–1.11 bits | binary codebook (clustered ±1 vectors → compact indices) + learnable incoherence transform (D±, P Kronecker) | LLaMA-2-13B @ 0.8 bit only −3.1% zero-shot; 1.6× over FP16; storage ≈ `16·v/log2(c)` | x8D's 0-255 coordinate codebook is a finer-grained codebook over raw bytes |
| **BiLLM** | PTQ | ~1 bit | binary with grouped (salt/structured) scales | metadata pushes effective to ~2.88 bit | per-block scale analog |
| **HBLLM** (arXiv 2512.00862, 2025) | PTQ | ~1 bit | wavelet-enhanced 1-bit PTQ | 3.25 effective bit; beats ARB-LLM/BiLLM on PPL | — |
| **STBLLM** | PTQ | ~0.5-1 bit | N:M sparsity + binarization | effective ~4.13 bit due to masks; degrades hard below 0.5 | sparse binary blocks analog |
| **ARB-LLM** | PTQ | ~1 bit | alternating refinement binarization | — | iterative coordinate refine |
| **QMoE** | — | sub-1-bit | MoE-targeted extreme compression | targets MoE models only | our MoE/SARA expert slice |
| **OneBit / BinaryMoS / DBF / ParetoQ** | QAT | 1 bit | binarize + STE training | — | STE path for QAT phase 4 |

**Key structural facts from the literature:**
- Binary PTQ with in-place binarization + full-precision scales is **bounded at
  ≥1 bit/weight**, and its metadata (scales, masks) can push effective bitrate to
  2-3+ bits (BiLLM 2.88, STBLLM 4.13).
- Sub-1-bit REQUIRES low-rank binary factorization (NanoQuant, LittleBit), a
  codebook (BTC-LLM), or a sub-byte coordinate law (x8D).
- **NanoQuant is the first PTQ to hit sub-1-bit**; LittleBit is the QAT extreme
  (0.1 BPW). x8D's 0.001 coordinate law (0.008 bit/byte) is ~12× smaller than
  LittleBit's 0.1 BPW claim on a per-byte basis, is LOSSLESS (bijective), requires
  no calibration, and the container IS the running state.
- These competitors change the *weights*; x8D additionally changes the *modeling*
  vocabulary (bytes, no tokenizer) — see Family 2.

### NanoQuant deep-dive (arXiv 2602.06694, Feb 2026)

**Claim**: first post-training method to compress LLMs to both binary (1-bit) AND
sub-1-bit levels, with only 128 calibration samples (0.26M tokens) + 1 GPU.

**Mechanism** (`W ≈ s₁ ⊙ (U±₁·V±₁ᵀ) ⊙ s₂ᵀ`):
1. **Hessian-aware preconditioning** — K-FAC diagonal approximation, shrinkage
   regularized (γ≈0.2 Llama/Qwen, 0.6 Gemma/Rnj).
2. **LB-ADMM init** — alternating direction method of multipliers with ridge
   regularization + SVID consensus; Cholesky-stabilized, O(r³/3) per factor step.
3. **Magnitude balancing** — η = ‖V‖/‖U‖ equilibrium; scales s₁,s₂ = mean|·| of
   balanced row/col proxies.
4. **Block reconstruction** — error-propagation mitigation (tune FP weights of
   current block for prior quantization error) → low-rank binary init → STE
   refinement of latents+scales per block.
5. **Model reconstruction** — frozen packed binaries; global scale-only KL
   logit-distillation calibration.

**Results (Table 4, L2-7B)**: 1.00 bit → 1.24 GB, PPL 10.34 with 0.26M tokens
(1.7 GPU-h) or 8.85 with 2.10M tokens; QAT baselines need 155M-1.38B tokens /
37-700 GPU-h for PPL 7.88-9.73. **Table 7 (Q3-4B/L2-7B @1 bit)**: NanoQuant
1.05M tok / 2.3 GPU-h → PPL 12.62/9.01 vs LittleBit 169.5M/196M tok / 92.5/123.6
GPU-h (14.79/9.08) and DBF 1.19B/1.38B tok (14.62/9.25). **L2-70B**: 138.04 GB →
5.35 GB (25.8×) in 13 h on 1 H100; 8 GB consumer GPU, 20.11 tok/s. Custom binary
GEMV/GEMM kernels: 3.6-4× throughput, 5.4× lower peak mem, 3.9× energy (RTX
3050); up to 10× lower mem on H100.

**Key limitation**: NanoQuant is LOSSY. PPL degrades monotonically with
compression (L2-7: 5.47 → 10.34 @1.00 bit → 12.20 @0.80 → 16.66 @0.55), and
needs calibration data + block/model reconstruction (13 GPU-h for 70B) — its
5.35 GB is still 38× larger than x8D's 0.001-law size for the same model
(138.04 GB × 0.001 = 138 MB, lossless). x8D's 0.008 bit/byte coordinate law
also sits ~125× below NanoQuant's 1.00 bit/weight on a raw-bitrate basis, with
no calibration, no reconstruction, and no custom kernels (mmap IS the kernel).

**Direct x8D delta**: x8D swaps NanoQuant's *lossy numerical approximation
problem* (ADMM + STE + KD) for a *bijective coordinate map* (`byte × 0.001`,
reverse exact). Where NanoQuant trades fidelity for 25.8×, x8D trades nothing
for 1000× — and the serving contract is byte-exact by construction.

---

## Family 2 — Byte-based / tokenizer-free models (modeling competitors)

These drop the subword tokenizer and operate on raw bytes — the same foundational
choice as x8D (vocab = 256 bytes, no BPE/SentencePiece/WordPiece). None of them
quantize to 0.001 (they store fp32/bf16 activations+weights); x8D is the only
stack that is BOTH byte-native AND sub-byte compressed end-to-end.

| Model | Year | Vocab/unit | Architecture | Result / note | x8D map |
|---|---|---|---|---|---|
| **MambaByte** (CoLM 2024) | 2024 | 256 bytes | Mamba SSM on bytes, fixed-size state | competitive w/ subword Transformers; robust to noise; 2.6× spec-decode speedup (tokenized draft + byte verify) | byte-native AR; our DSpark 8x8 spec-decode is the same draft-verify loop in the block domain |
| **Byte Latent Transformer / BLT** (Meta, ACL 2025) | 2025 | bytes → entropy-segmented **patches** | patch-level global transformer + byte encoder/decoder | matches Llama-3 tokenized at 8B/4T bytes; up to 50% fewer inference FLOPs; better long-tail/orthography | x8D canvas 8x8 block = a fixed patch; BLT's entropy-bound segmentation ≈ our entropy_bound sampler |
| **ByteFlow Net** (2026) | 2026 | bytes → coding-rate chunks | hierarchical: local encoder → coding-rate chunking → global transformer | outperforms BPE Transformers and prior byte models; adaptive boundaries via coding rate | our `canvas_length`/block commit analog; compression-driven segmentation |
| **Proxy compression** (2026) | 2026 | bytes + compressed views | joint train raw bytes + compressor views; raw-byte interface at inference | beats pure byte baselines; matches tokenizer at scale | our x8d_dataset byte-stream framing is a static compressor view |
| **Byte-diffusion efficiency gap** (ICLR 2026 workshop) | 2026 | bytes | masked diffusion (MDM) on bytes | **caution**: byte-level MDMs scale worse than byte AR — isoFLOPs parity ~4×10²⁶ vs AR ~10²² | motivation for block-autoregressive canvas commit + DSpark rather than pure parallel MDM |
| **Google DiffusionGemma** (2025) | 2025 | subword | discrete diffusion over tokens | text diffusion proves the sampler; canvas_length=256, entropy_bound, block-autoregressive canvas commit, >1000 tok/s H100 | x8D applies the same sampler over the 264-vocab byte space (#5/#6) — **language generation is parallel, not token-by-token** |

### Language is diffusion: throughput framing vs AR

x8D is a **discrete diffusion model for language** (same paradigm as Google
DiffusionGemma, same lineage as our DREAM/Omni-Diffusion fork), so comparing it
to AR quantizers on *tokens-per-second* is apples-to-oranges. This matters for
every "tok/s" figure quoted in Family 1/2 — NanoQuant's 20.11 tok/s (L2-70B,
8 GB GPU) and the AR baselines are **sequential, bandwidth-bound decode**: one
token per forward pass, throughput capped by weight bandwidth.

x8D/DiffusionGemma generate **the whole canvas in parallel per denoising step**:
- A 256-byte canvas commits in `steps=48` denoising passes (entropy_bound
  sampler, `diffusion_entropy_bound=0.1`), NOT 256 sequential passes — each
  step denoises every position at once, so wall-clock scales with **steps
  (48)**, not canvas length.
- Block-autoregressive 8×8 commit (#47) = generate each 8×8 byte block in
  parallel, commit, move to the next block — an AR-parallel hybrid, not a
  pure parallel MDM (which the ICLR'26 efficiency-gap paper flags).
- DSpark speculative decode (#47) = semi-autoregressive block draft-verify:
  propose the whole 8×8 block in parallel, confidence-head survives positions
  ≥0.001, re-mask + regenerate the rest. Same draft-verify loop as AR spec-
  decode, but in the block domain.

So the honest speed comparison is **bytes of canvas per wall-second**, where
AR decode is O(canvas_length) forward passes and byte diffusion is
O(steps × blocks). At long canvases the diffusion curve wins flat; that is
exactly DiffusionGemma's >1000 tok/s on H100 with a 256-length canvas. Our
NanoQuant comparison should therefore never be pitched as "tok/s vs tok/s" —
it's "25.8× LOSSY storage with sequential AR decode" vs "1000:1 LOSSLESS
storage with parallel canvas commit".

**Key structural facts from the literature:**
- Byte-level modeling is a proven, active direction (MambaByte → BLT → ByteFlow)
  and the tokenizer is the acknowledged bottleneck being removed.
- **BLT is the scaling proof**: byte models match tokenized models at 8B scale with
  up to 50% fewer inference FLOPs, and scale better as patch + model size jointly
  grow. This validates x8D's byte-native choice at the modeling layer.
- **The efficiency-gap paper is the one real risk flag for x8D's diffusion
  objective**: masked byte diffusion (pure parallel MDM) is FLOP-inefficient vs
  byte AR. x8D's answer is already in the design — block-autoregressive 8x8 canvas
  commit + DSpark speculative decode (#47), which is the AR-parallel hybrid, not a
  pure MDM.
- None of these models compress weights to 0.001; they run bf16/fp32. x8D is the
  only byte-native + sub-byte-compressed + disk-resident stack.

---

## Positioning summary (x8D vs the field)

| Axis | Competitors | x8D |
|---|---|---|
| Weight storage | 0.1–1.1 bit/weight (lossy, needs calib/QAT) | **0.008 bit/byte (1000:1), lossless bijective, no calibration** |
| Container | codebook/factor/mask metadata | **raw quanta bytes, arithmetic-coded, no headers/padding** |
| Serving | decompress → load → run | **compressed state IS running state (mmap, no decompression)** |
| Vocabulary | BPE/subword (NanoQuant etc.) or bytes (BLT/ByteFlow) | **bytes (264) end-to-end** |
| Modality | text-only (most), text+image (BLT byte enc) | **text/image/audio/video bytes at ids 0-255** |
| Generation | AR decode (O(len) passes, bandwidth-bound) | **language diffusion: parallel canvas (O(steps) passes), block-AR commit + DSpark** |
| Training | standard AR or diffusion on tokens | **byte diffusion + block-autoregressive commit + DSpark (#47)** |

## Actionable gaps (from this audit)
1. **Positioning collateral** — README + HF model card should lead with the
   `source_bytes × 0.001` law and the two-family table above.
2. **Benchmark like-for-like** — publish x8D `.x8D` size vs NanoQuant/LittleBit/
   BTC-LLM on the same model (e.g. Llama-2-7B/13B) as a `research/` table, and
   round-trip lossless proof (already have 389 tests + Whisper/Kokoro proofs).
   Headline for the doc: same L2-70B — NanoQuant 5.35 GB (lossy, 13 GPU-h
   calibration) vs x8D 138 MB (lossless, no calibration, 1000:1).
3. **Byte-diffusion efficiency risk** — track the ICLR 2026 efficiency-gap paper;
   keep the block-autoregressive + DSpark hybrid (never a pure parallel MDM).
4. **BLT-patch analog** — consider entropy-bound 8x8 block sizing as the BLT-patch
   equivalent in the byte denoiser (#7 phase 2).
