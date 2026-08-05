#!/usr/bin/env python
# coding=utf-8
"""x8D QAT fine-tuning scaffold over quantized ``.x8D`` weights.

Loads a quantized ``.x8D`` container via ``omni_diffusion/x8d_quanta.py``
(``QuantizedServingReader`` — zero-copy mmap + ``tensor_names``/``tensor_bytes``),
builds initial float weights from the stored U8 byte coordinates, and runs a
pure-stdlib byte-diffusion QAT loop:

1. ``split_canvas_blocks`` splits raw byte data into ``canvas_length`` blocks.
2. ``mask_canvas`` masks a fraction of each block (uniform state).
3. ``renoise_to_random_bytes`` refills masked positions with random bytes.
4. ``byte_diffusion_loss`` scores the denoiser against the true bytes over the
   264-vocab byte space.
5. The fake ``step`` only records the loss history — NO actual optimization
   (no torch, no GPU). The returned final weights are byte-aligned via
   ``quantize_ste`` so the follow-up ``.x8D`` export is near-lossless.

Runnable end-to-end offline with synthetic weights + synthetic byte data, so the
whole scaffold is testable in pure stdlib.

Run::

    python3 tools/finetune_qat.py                          # synthetic data
    python3 tools/finetune_qat.py --x8d kokoro.x8D --bytes corpus.bin
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.byte_diffusion import stable_hash  # noqa: E402
from omni_diffusion.x8d_qat import (  # noqa: E402
    QATConfig,
    byte_diffusion_loss,
    mask_canvas,
    quantize_ste,
    renoise_to_random_bytes,
)
from omni_diffusion.x8d_quanta import QuantizedServingReader  # noqa: E402


def load_quanta_weights(path: str) -> Dict[str, List[float]]:
    """Load quantized ``.x8D`` weight coordinates via ``QuantizedServingReader``.

    The container stores ONLY raw U8 byte coordinates (``Quanta[i] = byte *
    0.001``); here they are re-read as float weight values in ``[0, 255]`` —
    the byte domain QAT fine-tuning co-adapts to.

    Args:
        path: path to a raw ``.x8D`` quanta container.

    Returns:
        Dict name -> list of float weight values (one per stored byte).
    """
    reader = QuantizedServingReader(path)
    return {name: [float(b) for b in reader.tensor_bytes(name)] for name in reader.tensor_names}


def synth_weights(num_weights: int = 512, seed: int = 0) -> Dict[str, List[float]]:
    """Deterministic float weights near the U8 byte domain (QAT input).

    Args:
        num_weights: number of weight values to synthesize.
        seed: RNG seed.

    Returns:
        ``{"data": [...]}`` of floats in ``[-0.4, 255.4]``.
    """
    rng = random.Random(seed)
    return {"data": [float(rng.randint(0, 255)) + rng.uniform(-0.4, 0.4) for _ in range(num_weights)]}


def synth_dataset(num_bytes: int = 1024, seed: int = 1) -> bytes:
    """Deterministic synthetic byte stream (tier-0 style raw bytes).

    Args:
        num_bytes: number of raw bytes to synthesize.
        seed: RNG seed.

    Returns:
        Raw bytes over ids 0-255.
    """
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(num_bytes))


def split_canvas_blocks(byte_data: Sequence[int], canvas_length: int) -> List[List[int]]:
    """Split raw byte data into equal-length canvas blocks, zero-padding the tail.

    Args:
        byte_data: raw byte stream (ids 0-255).
        canvas_length: fixed canvas length (DiffusionGemma parity: 256).

    Returns:
        List of blocks, each exactly ``canvas_length`` bytes.
    """
    if canvas_length < 1:
        raise ValueError(f"canvas_length must be >= 1, got {canvas_length}")
    data = bytes(int(b) & 0xFF for b in byte_data)
    blocks: List[List[int]] = []
    for i in range(0, len(data), canvas_length):
        chunk = list(data[i : i + canvas_length])
        if len(chunk) < canvas_length:
            chunk += [0] * (canvas_length - len(chunk))
        blocks.append(chunk)
    return blocks


def _pseudo_logits(noisy_canvas: Sequence[int], weights_flat: Sequence[float], step: int) -> List[List[float]]:
    """Deterministic stand-in denoiser logits over the 264-vocab byte space.

    Each position's logit row is a stable hash over ``(position, byte, step)``
    plus a weight-guided boost on the observed (re-noised) byte channel — the
    surrogate for the future trained torch denoiser's output over ids 0-255.
    Byte-aligned weights give a larger boost, i.e. better calibration.

    Args:
        noisy_canvas: the re-noised canvas (masked positions refilled).
        weights_flat: flattened weight values conditioning the logits.
        step: current diffusion/training step.

    Returns:
        ``[n_pos][264]`` logits.
    """
    vocab = 264
    nw = max(len(weights_flat), 1)
    logits: List[List[float]] = []
    for i, observed in enumerate(noisy_canvas):
        weight = weights_flat[i % nw] if weights_flat else 0.0
        row = [float(stable_hash(i, b, step) & 0xFF) for b in range(vocab)]
        boost = 40.0 + (max(0.0, min(255.0, float(weight))) / 255.0) * 120.0
        row[observed % vocab] += boost
        logits.append(row)
    return logits


def fine_tune_qat(
    weights: Dict[str, Sequence[float]],
    dataset_bytes: Sequence[int],
    cfg: QATConfig,
    seed: int = 0,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Run the byte-diffusion QAT fine-tuning scaffold.

    The loop: split bytes into canvas blocks -> mask a fraction -> re-noise ->
    compute ``byte_diffusion_loss`` -> record the epoch loss. This is the fake
    ``step`` (no optimization, no torch); the final weights are byte-aligned
    with :func:`quantize_ste` so a follow-up ``.x8D`` export is near-lossless.

    Args:
        weights: name -> weight value sequence to co-adapt to the byte domain.
        dataset_bytes: raw byte training stream (ids 0-255).
        cfg: QAT hyper-parameters.
        seed: RNG seed (deterministic end-to-end).

    Returns:
        ``(loss_curve, final_weights)`` — per-epoch mean losses, and the
        byte-aligned final weights (all values integer floats in ``[0, 255]``).
    """
    if isinstance(weights, dict):
        weights_flat = [float(v) for values in weights.values() for v in values]
    else:
        weights_flat = [float(v) for v in weights]
    blocks = split_canvas_blocks(dataset_bytes, cfg.canvas_length)
    mask_ratio = 0.7
    loss_curve: List[float] = []
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        count = 0
        for block_index, block in enumerate(blocks):
            block_seed = seed + epoch * max(len(blocks), 1) + block_index
            masked, truth = mask_canvas(block, mask_ratio=mask_ratio, seed=block_seed)
            noisy = renoise_to_random_bytes(masked, seed=block_seed)
            logits = _pseudo_logits(noisy, weights_flat, step=epoch)
            epoch_loss += byte_diffusion_loss(logits, truth)
            count += 1
        loss_curve.append(epoch_loss / count if count else 0.0)
    lo, hi = cfg.quant_clamp
    final_weights = {name: [quantize_ste(v, lo, hi) for v in values] for name, values in weights.items()}
    return loss_curve, final_weights


def main(argv: Sequence[str] = None) -> int:
    """CLI entry point: QAT scaffold over a quantized ``.x8D`` container.

    Args:
        argv: optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="x8D QAT fine-tuning scaffold")
    parser.add_argument("--x8d", default=None, help="path to a quantized .x8D container")
    parser.add_argument("--bytes", default=None, help="path to a raw byte dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.x8d is not None:
        weights = load_quanta_weights(args.x8d)
    else:
        weights = synth_weights(seed=args.seed)
    if args.bytes is not None:
        with open(args.bytes, "rb") as f:
            data = f.read()
    else:
        data = synth_dataset(seed=args.seed + 1)

    cfg = QATConfig(epochs=args.epochs, batch_size=args.batch_size, canvas_length=args.canvas_length)
    curve, final = fine_tune_qat(weights, data, cfg, seed=args.seed)
    print(f"x8D QAT scaffold: {len(weights)} weight name(s), {len(data)} dataset bytes, "
          f"{cfg.epochs} epoch(s), canvas {cfg.canvas_length}")
    for i, loss in enumerate(curve):
        print(f"  epoch {i + 1}: loss={loss:.6f}")
    print(f"  final weights byte-aligned: {all(v == float(int(v)) for vals in final.values() for v in vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
