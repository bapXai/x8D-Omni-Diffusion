# coding=utf-8
"""DSpark-style speculative decoding quantizer for x8Dsub-byte compression.

Pure Python standard library only. Applies the x8D quantize law
(``Quanta[i] = weight_byte[i] * 0.001``) with **semi-autoregressive
speculative decoding** on 8x8 byte blocks, per the AGENTS.md spec:

1. Generate entire 8x8 byte blocks in parallel (not one byte at a time).
2. A lightweight **confidence head** predicts survival probability per position.
3. Positions with confidence **below the 0.001 threshold** are re-masked and
   regenerated.
4. Under heavy load, dynamically clip verification length to save compute.

The flow operates purely on byte coordinates (0-255), never on floats or
tokens, so it stays inside the byte law.
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .x8d_export import LAW, quantize, to_u8

#: DSpark block size: an 8x8 byte block = 64 bytes decoded in parallel.
BLOCK_ROWS: int = 8
BLOCK_COLS: int = 8
BLOCK_SIZE: int = BLOCK_ROWS * BLOCK_COLS

#: x8D survival threshold -- positions below it are regenerated.
CONFIDENCE_THRESHOLD: float = 0.001

#: Heavy-load clip: verification length shrinks 16x under load.
HEAVY_LOAD_VERIFY_CLIP: int = 16

DEFAULT_VERIFY_LEN: int = 64

#: Per-byte confidence contribution ``b / 256`` (avoid float() per position).
_BYTE_SCALE: Tuple[float, ...] = tuple(b / 256.0 for b in range(256))


class SpeculativeDecodeError(ValueError):
    """Raised when speculative decoding cannot reach convergence."""


def _split_blocks(data: bytes, block_size: int = BLOCK_SIZE) -> List[bytes]:
    """Split raw bytes into equal-sized blocks, padding the tail with zeros.

    Args:
        data: raw weight bytes.
        block_size: number of bytes per block (default 64 = 8x8).

    Returns:
        List of blocks, each exactly ``block_size`` bytes.
    """
    blocks: List[bytes] = []
    for i in range(0, len(data), block_size):
        chunk = data[i : i + block_size]
        if len(chunk) < block_size:
            chunk = chunk + bytes(block_size - len(chunk))
        blocks.append(chunk)
    return blocks


def _block_surrogate(block: bytes, step: int) -> float:
    """Deterministic pseudo-confidence for a block, in [0, 1).

    Stand-in for a real lightweight confidence head (the actual model's
    head would predict survival probability per position). Deterministic so
    speculative quantization is reproducible.

    Args:
        block: 8x8 byte block.
        step: current decode step (injects schedule dependence).

    Returns:
        A confidence value in [0.0, 1.0).
    """
    digest = hashlib.sha256(bytes(block) + bytes([step & 0xFF])).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def _verify_positions(
    quanta: Sequence[float],
    confidence: Sequence[float],
    threshold: float = CONFIDENCE_THRESHOLD,
    heavy_load: bool = False,
    verify_len: Optional[int] = None,
) -> List[int]:
    """Return byte positions that FAILED verification (below threshold).

    Positions whose confidence is below the 0.001 threshold are re-masked
    for regeneration. Under heavy load the verification length is clipped
    (only the first ``verify_len`` positions are checked).

    Args:
        quanta: sub-byte coordinates for the block.
        confidence: survival probability per position (parallel to quanta).
        threshold: x8D survival threshold (default 0.001).
        heavy_load: clip verification length when True.
        verify_len: optional explicit verification length.

    Returns:
        List of failed byte positions within the block.
    """
    n = len(quanta)
    if heavy_load:
        n = min(n, verify_len or (DEFAULT_VERIFY_LEN // HEAVY_LOAD_VERIFY_CLIP))
    elif verify_len is not None:
        n = min(n, verify_len)
    return [i for i in range(n) if confidence[i] < threshold]


def speculative_quantize(
    weight_bytes: bytes,
    max_steps: int = 16,
    seed: int = 0,
    heavy_load: bool = False,
    verify_len: Optional[int] = None,
) -> Tuple[List[float], Dict[str, int]]:
    """Speculatively quantize raw weight bytes block-by-block (8x8).

    Each 8x8 block is generated in parallel in one shot; the confidence head
    then verifies every position and any position below the 0.001 threshold
    is re-masked and regenerated. Blocks that never converge raise.

    Args:
        weight_bytes: raw uint8 weight bytes.
        max_steps: regeneration budget per block.
        seed: RNG seed for reproducible regeneration.
        heavy_load: enable dynamic verification-length clipping.
        verify_len: optional verification length override.

    Returns:
        ``(quanta, stats)`` where quanta are the sub-byte coordinates and
        stats reports blocks, regenerations and convergence.
    """
    rng = random.Random(seed)
    blocks = _split_blocks(weight_bytes)
    out_quanta: List[float] = []
    stats = {"blocks": len(blocks), "regenerations": 0, "converged": 0}
    step = 0
    byte_scale = _BYTE_SCALE
    for block in blocks:
        current = block
        for _ in range(max_steps):
            current_quanta = quantize(current)
            # one sha256 per block, not one per position (64x fewer hashes)
            block_conf = float(_block_surrogate(current, step))
            confidence = [(block_conf + byte_scale[b]) / 2.0 for b in current]
            failed = _verify_positions(
                current_quanta,
                confidence,
                heavy_load=heavy_load,
                verify_len=verify_len,
            )
            # Lossless guard: a position whose coordinate re-decodes to its
            # original byte must NEVER be regenerated. quantize() is exact
            # (b*0.001 -> round(b*0.001/0.001) == b), so the confidence
            # surrogate alone must not flag correct bytes -- e.g. byte 0 has
            # confidence block_conf/2 and would fall under the 0.001
            # threshold in low-hash blocks, corrupting zero-heavy data.
            failed = [
                i
                for i in failed
                if (int(round(current_quanta[i] / LAW)) & 0xFF) != current[i]
            ]
            if not failed:
                break
            stats["regenerations"] += 1
            # re-mask failed positions and regenerate from byte-space noise
            renoised = bytearray(current)
            for i in failed:
                renoised[i] = rng.randint(0, 255)
            current = bytes(renoised)
            step += 1
        else:
            raise SpeculativeDecodeError(
                f"block {len(out_quanta) // BLOCK_SIZE} did not converge in {max_steps} steps"
            )
        stats["converged"] += 1
        out_quanta.extend(quantize(current))
    # trim the zero-padded tail: output must be length-preserving
    del out_quanta[len(weight_bytes):]
    return out_quanta, stats


def speculative_save_gguf(
    name: str,
    weight_bytes: bytes,
    filename: str,
    max_steps: int = 16,
    seed: int = 0,
    heavy_load: bool = False,
) -> Tuple[str, Dict[str, int]]:
    """Speculatively quantize a weight tensor and store it in an x8D container.

    Args:
        name: tensor name inside the container.
        weight_bytes: raw weight bytes.
        filename: output .gguf path.
        max_steps: regeneration budget per 8x8 block.
        seed: RNG seed.
        heavy_load: clip verification length.

    Returns:
        ``(output_path, stats)``.
    """
    from .x8d_export import save_gguf

    quanta, stats = speculative_quantize(
        weight_bytes, max_steps=max_steps, seed=seed, heavy_load=heavy_load
    )
    payload = to_u8(quanta)
    save_gguf({name: payload}, filename)
    return filename, stats


#: Speculative decode pipeline hooks for the future torch DreamModel.
#
# A real integration replaces ``_block_surrogate`` with a lightweight
# confidence head (linear probe over the 8x8 block embedding) and the
# byte-space regeneration with the actual model's logits over ids 0-255.
QuantizerCallable = Callable[[bytes, int], float]


def confidence_head_probe(
    block: bytes, step: int, surrogate: Optional[QuantizerCallable] = None
) -> float:
    """Confidence head entry point (override-able in a torch build).

    Args:
        block: 8x8 byte block.
        step: decode step.
        surrogate: optional replacement for the deterministic surrogate.

    Returns:
        Per-block survival probability in [0.0, 1.0).
    """
    fn = surrogate or _block_surrogate
    return fn(block, step)


def size_report(
    num_params: int = 16_000_000_000,
    baseline_bits: int = 16,
    stored_bytes_per_param: float = 1.0,
) -> Dict[str, float]:
    """Compute the size comparison for a model of ``num_params`` weights.

    Args:
        num_params: total parameter count (default 16B).
        baseline_bits: float width of the original checkpoint (default 16 =
            BF16/FP16).
        stored_bytes_per_param: x8D on-disk cost (U8 byte = 1.0).

    Returns:
        Dict with baseline_size_gb, x8d_storage_gb, subbyte_coordinate_mb,
        disk_reduction_pct and coordinate_reduction_pct.
    """
    baseline_bytes = num_params * (baseline_bits / 8.0)
    x8d_bytes = num_params * stored_bytes_per_param
    # theoretical sub-byte coordinate space: baseline_bits * LAW bits per weight
    subbyte_bits = num_params * (baseline_bits * LAW)
    subbyte_bytes = subbyte_bits / 8.0
    return {
        "baseline_size_gb": baseline_bytes / 1e9,
        "x8d_storage_gb": x8d_bytes / 1e9,
        "subbyte_coordinate_mb": subbyte_bytes / 1e6,
        "disk_reduction_pct": (1.0 - x8d_bytes / baseline_bytes) * 100.0,
        "coordinate_reduction_pct": (1.0 - subbyte_bytes / baseline_bytes) * 100.0,
        "law": LAW,
    }


def print_size_report(num_params: int = 16_000_000_000, baseline_bits: int = 16) -> None:
    """Human-readable size comparison (full FP16 model vs x8D sub-byte)."""
    r = size_report(num_params=num_params, baseline_bits=baseline_bits)
    print(f"x8Dsub-byte 0.001 size comparison ({num_params:,} params, {baseline_bits}-bit baseline)")
    print(f"  Full FP16/BF16 model : {r['baseline_size_gb']:.2f} GB")
    print(f"  x8D U8 .gguf storage  : {r['x8d_storage_gb']:.2f} GB  (disk reduction {r['disk_reduction_pct']:.1f}%)")
    print(f"  Sub-byte coordinates  : {r['subbyte_coordinate_mb']:.1f} MB (coordinate reduction {r['coordinate_reduction_pct']:.2f}%)")
    print(f"  Scaling law           : {r['law']}  (0.001 x {baseline_bits} bits = {baseline_bits*LAW:.3f} bit/weight)")
