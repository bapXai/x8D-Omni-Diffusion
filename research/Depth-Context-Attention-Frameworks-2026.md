# 2026 Depth & Context Attention Frameworks — AttnRes, KDA, mHC, Engram (issue #24)

Status: researched 2026-07-31 from primary sources (HF papers pages, arxiv
HTML). Two papers fetched and verified directly; mHC/V4/Engram marked
secondary (corroborated by the AttnRes paper's own mHC comparison).

Goal: map the 2026 architecture-paper wave that shapes Kimi K3 / DeepSeek V4,
and decide what matters for x8Dsub-byte (storage/serving) + the Dream byte
denoiser (training). Two axes, same medicine:
- **depth** = look-back across LAYERS (residuals) — AttnRes, mHC, CLVR.
- **context** = look-back across TOKENS (sequence) — KDA, Gated DeltaNet-2, MLA.

## 1. Attention Residuals (AttnRes) — arXiv:2603.15031 (Kimi Team)

**Problem — PreNorm dilution.** Standard residuals accumulate every layer
output with fixed unit weight: `h_l = h_{l-1} + f(h_{l-1})`. Unrolled, each
layer sees a uniformly-weighted sum of ALL prior outputs. PreNorm makes
hidden-state magnitudes grow O(L) with depth, diluting early-layer info and
destabilizing gradients (explode early / vanish deep). ~fraction of layers
prunable with little loss.

**Fix — softmax attention over depth.** Replace fixed accumulation with
learned, input-dependent weights:

```
h_l = alpha_{0->l}·h_1 + sum_{i=1..l-1} alpha_{i->l}·f_i(h_i)
alpha_{i->l} = softmax over depth of exp(q_l^T RMSNorm(k_i))
q_l = w_l  (learned pseudo-query, d-vector, INITIALIZED TO ZERO)
k_i = v_i  (layer outputs)
```

Zero init is critical: the first step is an equal-weight average of the
baseline, so training starts safe. One RMSNorm + one w_l per layer ≈
negligible parameters.

**Block AttnRes (the scalable form).** L layers → N blocks (N≈8 empirically);
intra-block outputs summed to one representation `b_n`, inter-block attention
over the N block summaries + embedding. Memory/comm drop O(Ld) → O(Nd); per
token per layer I/O = **5.5d** (Full 24d, mHC 34d @ m=4 streams).

**Results.** Block AttnRes ≈ baseline trained with **1.25× more compute**;
Full 1.737 vs mHC 1.747 vs DenseFormer 1.767 (no gain — proves input-dependence
matters) vs Baseline 1.766 (16-layer ablation). Pre-trained into Kimi Linear
48B/3B (54 layers, 6/block → 9 blocks + embedding = 10 sources). Downstream:
GPQA-Diamond **+7.5**, Math +3.6, HumanEval +3.1, MMLU +1.1. Multihead depth
attention HURTS (1.752) — the depth mixture is uniform across channels.

## 2. Kimi Delta Attention (KDA) / Kimi Linear — arXiv:2510.26692

**Problem.** Full attention is O(T²) memory/time; KV cache explodes with
context (K3 targets 1M tokens).

**Fix — recurrent linear memory with channel-wise forgetting.** KDA is a
Gated-DeltaNet-style associative memory W ∈ R^{dv×dφ}. Per token:

```
prediction:     vbar = W D_alpha kappa
residual:       r = v - vbar
memory update:  W = W D_alpha + beta·r ⊗ kappa     (D_alpha = Diag(vector decay gate))
output:         y = W·phi(q)
```

The vector decay gate α ∈ (0,1)^dφ decays different key-feature channels at
different rates (vs Gated DeltaNet's single scalar). Write strength β is a
scalar. Constant-size memory → O(1) per-token inference, linear training.

**Gated DeltaNet-2** (SDSC/Samba, per ETH paper) decouples β into channel-wise
erase gate b (key-side) + write gate w (value-side): the most granular memory
control of the family. Collapses to KDA when b = w = β·1.

**Kimi K3 integration.** K3 interleaves **3 KDA layers per 1 MLA layer** —
KDA tracks/compresses long history, MLA routes final output. Cuts KV cache up
to 75%, ~6× decode throughput. Combined with AttnRes for depth.

## 3. ETH linear-attention comparison — arXiv:2607.07953

350M-param / 15B-token matched sweep (DeltaNet, Gated DeltaNet, KDA, GDN-2 +
softmax baseline):

| Finding | Detail |
|---|---|
| Best final loss | **KDA + Muon (hybrid)** 2.273 |
| Optimizer | Muon lowers final loss for EVERY family vs AdamW (lr ~3e-4 vs 1e-3) |
| Hybrid vs pure | Hybrid (e.g. 2:1 or 3:1 linear:softmax) improves loss at a throughput cost |
| Iteration time 4k→32k | softmax +192%, GDN hybrid +65%, pure GDN +8% |
| Fastest (pure GDN, AdamW) | 100% throughput but loss 2.433 |

**Cross-Layer Value Routing (CLVR).** New: route a lower delta-rule layer's
internal WRITE VALUE (not its error) into the shared residual stream through a
zero-initialized projection `h += P_l·v_l`. Modest validation-loss gain for
DeltaNet/Gated DeltaNet; preserves linear-time structure. The write-error
variant (CLER) does NOT help — only the aligned hidden-stream value injection
works. This is a cheap depth pathway that complements AttnRes for recurrent
stacks.

## 4. DeepSeek: mHC, Engram, V4 (secondary; mHC corroborated by AttnRes)

| Mechanism | What | Source confidence |
|---|---|---|
| **mHC** (Manifold-constrained Hyper-Connections, Jan 2026) | Rethinks ResNet residuals: m parallel streams with learned mixing matrices A_l, α_l, β_l; re-projects across the network graph to stabilize deep stacks | Independent numbers in AttnRes paper: 34d I/O/layer @ m=4; 16-layer ablation 1.747 |
| **Engram** (conditional memory / scalable lookup) | Hardware-efficient conditional dictionary lookup as external neural scratchpad; 97%+ retrieval accuracy, active KV cache → ~10% of V3 | User/secondary (2026 DeepSeek materials) |
| **V4-Pro** (Apr 2026) | 1.6T/49B active MoE, hybrid CSA/HCA (Compressed Sparse Attention) + mHC, auxiliary-loss-free load balance, **MTP** (multi-token-prediction modules = built-in spec-decode drafters), **DSpeak** native voice | User/secondary |

Note: MTP is the one that aligns most with x8D — it makes the model its own
speculative decoder (draft + verify), the same DSpark idea our
`x8d_spec_decode.py` applies to weight compression.

## 5. Comparison: depth-lookback mechanisms

| Mechanism | Depth access | Weighting | Memory | I/O/layer | 16L val loss |
|---|---|---|---|---|---|
| Standard residual | h_{l-1} only | fixed = 1 | — | 3d | 1.766 |
| DenseFormer | all previous | fixed scalars | O(Ld) | — | 1.767 |
| SWA (W=8) | last 8 | fixed | O(8d) | — | 1.764 |
| mHC (m=4) | prev stream + mixes | learned streams | O(md) | 34d | 1.747 |
| AttnRes Full | all previous | softmax (input-dep) | O(Ld) | 24d | **1.737** |
| AttnRes Block (N=8) | N block summaries | softmax (input-dep) | O(Nd) | **5.5d** | 1.746 |
| CLVR (on delta-rule) | nearest lower layer value | zero-init learned proj | O(d) | ~3d | modest gain (DeltaNet/GDN) |

Takeaway: AttnRes (esp. Block) dominates on both loss AND memory I/O; mHC is
competitive on loss at 6× the I/O; DenseFormer/SWA prove fixed-weight depth
mixing does not help.

## 6. What x8D-Omni-Diffusion should optimize (decision)

1. **Serving (pointer map / moe_disk) — no change needed.** AttnRes adds ~1
   RMSNorm + 1 d-vector per layer (< 0.01% params) → nothing to quantize.
   KDA/MLA are already in K3's weight layout; our pointer map serves any span.
   The real serving win is already captured: KDA = smaller KV cache = fewer
   bytes fetched per token.
2. **Spec-decode at inference — adopt MTP idea.** MTP modules turn the model
   into its own drafter. Our `_block_surrogate` in `x8d_spec_decode.py` is a
   stand-in confidence head; replace it with a real MTP-style head when torch
   lands (same as the planned #7 confidence-head upgrade).
3. **Dream denoiser depth — adopt Block AttnRes.** Our byte-denoiser suffers
   the same depth dilution as any deep stack; Block AttnRes (N=8) is the
   drop-in residual replacement with 5.5d I/O. Fold into issue #7's planned
   `kda_attention.py` work as a phase: KDA (context) + Block AttnRes (depth).
4. **Cross-layer routing — CLVR is the cheap test.** For a recurrent-memory
   (KDA) Dream variant, route each layer's write value into the residual
   stream via zero-init projection. Zero-init = drop-in safe, byte-law neutral.
5. **Training optimizer — Muon.** ETH paper: Muon beats AdamW for every
   recurrent/linear family at matched scale. Our Dream training plan
   (Training-Dataset-and-Quantization-Plan.md) should default Muon for any
   KDA-style variant.

## Sources
- Attention Residuals (Kimi Team), arXiv:2603.15031; github.com/MoonshotAI/Attention-Residuals.
- Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer
  Routing (ETH Zurich), arXiv:2607.07953.
- Kimi Linear (Moonshot), arXiv:2510.26692; github.com/MoonshotAI/Kimi-Linear.
- mHC numbers cross-checked against AttnRes paper (Tables 2/4, Table 1 I/O).
- DeepSeek mHC/Engram/V4: user-provided secondary summary (2026 materials).
