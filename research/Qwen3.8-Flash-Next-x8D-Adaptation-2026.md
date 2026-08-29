# Qwen3.8-Flash-Next -> x8D adaptation (2026)

## Scope

This note maps Qwen3.8-Flash-Next architecture ideas into x8D-Omni-Diffusion without importing Qwen's tokenizer or GPU-resident execution assumptions.

x8D invariant: the model-visible domain is exactly raw unsigned bytes `0..255`. There are no MASK/PAD/BOS/EOS/modality IDs above 255. Text is UTF-8 bytes; images/audio/video/PDF/code/binaries use their real serialized bytes and native headers. Diffusion corruption/re-noising also remains within `0..255`.

Primary serving target: **zero VRAM requirement**. The compressed state remains disk-resident and is accessed through NVMe/mmap/page cache with bounded CPU working state. Accelerators may be optional compute devices but must not become required model-storage tiers.

## Primary Qwen3.8 sources

- Official Qwen release/blog: https://qwen.ai/blog?id=qwen3.8-flash-next
- Official Qwen repository: https://github.com/QwenLM/Qwen3.8-Flash-Next
- Qwen technical report PDF: https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf

Qwen3.8-Flash-Next introduces four architecture directions relevant to x8D: Qwen Sparse Attention (QSA) combined with Gated DeltaNet (GDN), Gated Residual, deterministic N-gram embedding memory that can be offloaded to host memory, and Muon-centered optimization with parameter-specific optimizer assignment.

## Relevant upstream research/implementation repos

- NVIDIA Gated DeltaNet reference: https://github.com/NVlabs/GatedDeltaNet
- Flash Linear Attention (optimized GDN/KDA/AttnRes implementations): https://github.com/fla-org/flash-linear-attention
- Muon optimizer reference: https://github.com/KellerJordan/Muon
- Kimi Linear / KDA + MLA baseline already relevant to x8D issue #7: https://github.com/MoonshotAI/Kimi-Linear

## Adaptation 1: exact Byte N-gram Memory

Qwen's N-gram embedding is particularly compatible with an exact 256-byte domain.

- unigram states: `256`
- bigram addresses: `256^2 = 65,536`
- trigram addresses: `256^3 = 16,777,216`

A trigram can be addressed without hashing or collisions:

```text
addr = (b0 << 16) | (b1 << 8) | b2
```

This should be implemented as disk-backed/mmap lookup memory. The complete table must not require RAM or VRAM residency; only requested pages/ranges should enter the working set.

## Adaptation 2: Byte-QSA

Translate Qwen Sparse Attention from token micro-blocks to raw-byte blocks:

```text
byte canvas -> block summaries/index -> select relevant byte blocks -> sparse global retrieval
```

Requirements:

- no full-context GPU KV cache;
- bounded index state;
- mmap/NVMe-backed context data where practical;
- telemetry for page faults, NVMe bytes read, RSS, latency and selected blocks;
- benchmark against existing KDA/MLA and DSpark paths on long byte streams.

## Adaptation 3: GDN as bounded bidirectional byte memory

Qwen's GDN is causal. x8D diffusion is not inherently causal, so do not copy it literally. Prototype forward+reverse fixed-state delta memory or another bidirectional bounded-state variant and compare against KDA/MLA.

The goal is long-range memory without a large resident KV cache.

## Adaptation 4: Gated Residual

Prototype Qwen-style multi-branch gated residual streams inside the byte-diffusion backbone. Branch count and low-rank gate dimensions must be selected by ablation rather than copied from Qwen.

Measure convergence, quality, RSS, memory traffic and latency in the zero-VRAM runtime.

## Adaptation 5: sparse disk-backed MoE

Extend existing x8D MoE/SARA work rather than building another router.

Target runtime:

```text
request/context
 -> router
 -> selected expert ids
 -> mmap selected compressed expert ranges
 -> bounded working state
 -> compute
 -> evict/release unused pages
```

Requirements:

- configurable expert count/top-k/shared expert;
- route-first expert selection;
- async NVMe/page-cache prefetch;
- no full expert set in RAM or VRAM;
- active-expert and I/O telemetry.

Qwen's expert counts are research reference points, not x8D defaults.

## Adaptation 6: Muon + AdamW under x8D QAT

Evaluate Muon on appropriate 2-D transformation matrices (attention/GDN/MoE experts) while retaining AdamW-like optimization for embeddings, routing/gating and other non-Muon parameters, following the Qwen and Muon parameter-allocation principle.

This is an ablation under existing x8D QAT/STE, not a new invariant until measured.

## Adaptation 7: MTP concept -> multi-step diffusion prediction

Do not copy autoregressive next-token MTP semantics. Translate it to diffusion-state/block prediction:

```text
state t -> predict t-1 and t-2 + confidence/verification
```

Integrate with DSpark block-parallel prediction and verification so the benefit is fewer denoising iterations/blocks.

## Explicit rejects

- Qwen tokenizer/vocabulary.
- Any model-visible state outside `0..255`.
- MASK/PAD/BOS/EOS or synthetic image/audio framing states.
- Mandatory VRAM residency for weights, experts, n-gram memory or context cache.
- Copying Qwen dimensions/expert counts without x8D measurements.
- Replacing diffusion with causal next-token generation.

## Issue mapping

- #2: exact 256-byte domain and removal of special states.
- #5/#36: sparse disk-backed MoE and route-first expert prefetch.
- #6: Muon/AdamW vs current optimizer under QAT/STE.
- #7/#24: benchmark KDA/MLA/AttnRes against GDN+QSA+Gated Residual.
- #37/#4: MTP-derived multi-step block/diffusion prediction.
- #45: authoritative zero-VRAM serving requirement.
- #54/#56: remove MASK=256 training/sampling assumptions and keep corruption in 0..255.

## Required measurements

Every architecture experiment must report at least:

- quality/task metric;
- model/compressed disk size;
- RSS / peak RAM;
- VRAM (primary target = 0 required);
- NVMe bytes read;
- page faults/cache hit behavior where available;
- first-byte latency;
- steady-state throughput;
- active expert/block count.
