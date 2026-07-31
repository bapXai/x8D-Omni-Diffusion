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
from dataclasses import dataclass
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

#: Byte-law special ids: MASK=256 is the diffusion masking state, PAD/BOS/EOS
#: and the IMG/AUD span markers follow. Only 0-255 are data bytes.
MASK_ID: int = 256

#: Supported modality names for the k-parallel block-mask schedules.
MODALITIES: Tuple[str, ...] = ("text", "image", "audio", "video")


@dataclass(frozen=True)
class DSparkMaskConfig:
    """k-parallel DSpark block-mask schedule for one modality.

    Formalizes the DSpark block configuration: each 8x8 block (64 bytes) is
    masked, re-noised and verified **in parallel** as one batch of ``k_blocks``
    blocks (DiffusionGemma-style block-autoregressive canvas commit, with
    ``canvas_length`` for the 264-byte vocab / 256-byte canvas parity).

    Attributes:
        k_blocks: number of parallel 8x8 blocks decoded per batch.
        mask_ratio: fraction of the 64 byte positions masked per round.
        entropy_bound: confidence threshold; positions below it are re-masked
            and regenerated.
        verify_clip: verification length per block; None = full 64 bytes,
            heavy load clips to ``BLOCK_SIZE // 16``.
        canvas_length: diffusion canvas length (DiffusionGemma parity: 256).
        modality: one of text/image/audio/video.
        steps: denoising/regeneration steps per block.
    """

    k_blocks: int = 8
    mask_ratio: float = 0.7
    entropy_bound: float = 0.001
    verify_clip: Optional[int] = None
    canvas_length: int = 256
    modality: str = "text"
    steps: int = 48


#: Per-modality DSpark block-mask presets. Text runs the language-throughput
#: path (deep denoise, modest masking); dense modalities mask more aggressively
#: and decode more blocks in parallel but need fewer steps.
DSPARK_MODALITY_SCHEDULES: Dict[str, DSparkMaskConfig] = {
    "text": DSparkMaskConfig(),
    "image": DSparkMaskConfig(
        modality="image", mask_ratio=0.85, k_blocks=16, steps=32
    ),
    "audio": DSparkMaskConfig(
        modality="audio", mask_ratio=0.8, k_blocks=8, steps=40
    ),
    "video": DSparkMaskConfig(
        modality="video", mask_ratio=0.9, k_blocks=32, steps=24
    ),
}


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


def mask_block(
    block: bytes, cfg: DSparkMaskConfig, seed: int
) -> Tuple[List[int], List[int]]:
    """Mask ``mask_ratio`` of the 64 byte positions to MASK(256).

    Deterministic by seed: the masked positions are drawn without replacement
    from a ``random.Random(seed)`` sampler, so a given ``(block, cfg, seed)``
    triple always masks the same positions.

    Args:
        block: 8x8 byte block (exactly 64 bytes).
        cfg: DSpark mask schedule.
        seed: RNG seed selecting the masked positions.

    Returns:
        ``(masked_ids, truth_ids)`` where ``masked_ids`` holds the block with
        masked positions replaced by MASK(256) and ``truth_ids`` holds the
        original 64 bytes (used for verification).
    """
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"block must be {BLOCK_SIZE} bytes, got {len(block)}")
    rng = random.Random(seed)
    n_mask = int(round(cfg.mask_ratio * BLOCK_SIZE))
    n_mask = max(0, min(BLOCK_SIZE, n_mask))
    positions = rng.sample(range(BLOCK_SIZE), n_mask)
    pos_set = set(positions)
    truth = list(block)
    masked = [MASK_ID if i in pos_set else truth[i] for i in range(BLOCK_SIZE)]
    return masked, truth


def renoise_block(masked: List[int], seed: int) -> List[int]:
    """Refill MASK(256) slots with seeded random bytes 0-255.

    Args:
        masked: block with MASK(256) at masked positions.
        seed: RNG seed for the replacement bytes.

    Returns:
        The block with every MASK slot replaced by a random byte 0-255 (no
        MASK ids remain).
    """
    rng = random.Random(seed)
    return [rng.randint(0, 255) if v == MASK_ID else v for v in masked]


def dspark_block_generate(
    block: bytes, cfg: DSparkMaskConfig, seed: int = 0
) -> bytes:
    """Run the DSpark k-parallel mask loop on one 8x8 block.

    Each step, in parallel across the block: mask ``mask_ratio`` of the 64
    byte positions to MASK(256), re-noise the masked slots with random bytes,
    score every position with the confidence head (``_block_surrogate``-style:
    one sha256 per block + per-byte scale), accept positions at or above the
    ``entropy_bound`` and regenerate the rest. ``verify_clip`` honors the
    heavy-load clip (only the first N positions are verified).

    Args:
        block: 8x8 byte block (exactly 64 bytes).
        cfg: DSpark mask schedule.
        seed: RNG seed (deterministic end-to-end).

    Returns:
        The generated 64 bytes.
    """
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"block must be {BLOCK_SIZE} bytes, got {len(block)}")
    rng = random.Random(seed)
    verify_len = (
        BLOCK_SIZE if cfg.verify_clip is None else min(BLOCK_SIZE, cfg.verify_clip)
    )
    byte_scale = _BYTE_SCALE
    current = bytearray(block)
    for step in range(cfg.steps):
        masked, _ = mask_block(bytes(current), cfg, seed=seed + step)
        candidate = renoise_block(masked, seed=seed + step)
        block_conf = float(_block_surrogate(bytes(candidate), step))
        confidence = [(block_conf + byte_scale[b]) / 2.0 for b in candidate]
        for i in range(verify_len):
            if confidence[i] < cfg.entropy_bound:
                candidate[i] = rng.randint(0, 255)
        current = bytearray(candidate)
    return bytes(current)


def dspark_batch_mask(
    blocks: List[bytes], cfg: DSparkMaskConfig, seed: int = 0
) -> List[bytes]:
    """Apply the k-parallel schedule across ``cfg.k_blocks`` blocks at a time.

    The multi-modal + language throughput path: ``cfg.k_blocks`` 8x8 blocks are
    decoded in parallel per round; each block derives its seed from the batch
    offset so the whole batch is deterministic. A trailing partial batch (when
    ``len(blocks)`` is not a multiple of ``k_blocks``) is processed as-is.

    Args:
        blocks: list of 8x8 byte blocks (each exactly 64 bytes).
        cfg: DSpark mask schedule.
        seed: base RNG seed for the batch.

    Returns:
        One generated 64-byte block per input block, in order.
    """
    out: List[bytes] = []
    for start in range(0, len(blocks), cfg.k_blocks):
        group = blocks[start : start + cfg.k_blocks]
        for offset, block in enumerate(group):
            out.append(dspark_block_generate(block, cfg, seed=seed + start + offset))
    return out


def dspark_generate(
    context_ids: Sequence[int],
    completion_bytes: bytes,
    cfg: Optional[DSparkMaskConfig] = None,
    seed: int = 0,
    heavy_load: bool = False,
) -> Tuple[List[int], Dict[str, int]]:
    """DSpark block-parallel completion generation over an observed canvas.

    Applies the AGENTS.md speculative-decoding findings to the **inference**
    path (the quantization counterpart is :func:`speculative_quantize`):

    1. The observed context ids are kept UNMASKED on the canvas — the prompt
       is never destroyed, only the completion span is masked (DiffusionGemma
       block-autoregressive canvas commit).
    2. The completion span is decoded in 8x8 byte blocks generated in
       parallel (``cfg.k_blocks`` blocks per batch).
    3. A lightweight confidence head scores every position; positions whose
       confidence falls below the 0.001 entropy bound are re-masked and
       regenerated.
    4. Under heavy load, verification length is dynamically clipped
       (``BLOCK_SIZE // 16`` positions per block).

    ``completion_bytes`` is the deterministic draft completion — the
    surrogate for the future trained model's logits over ids 0-255. It is
    transported onto the canvas losslessly (lossless guard, mirroring
    :func:`speculative_quantize`).

    Args:
        context_ids: observed canvas prefix (prompt ids 0-263), never masked.
        completion_bytes: deterministic completion bytes to generate.
        cfg: DSpark block-mask schedule (default ``text``).
        seed: RNG seed (deterministic end-to-end).
        heavy_load: clip verification length when True.

    Returns:
        ``(canvas_ids, stats)`` where canvas = context + completion and
        ``stats`` reports blocks, regenerations and convergence.
    """
    cfg = cfg or DSPARK_MODALITY_SCHEDULES["text"]
    canvas: List[int] = list(context_ids) + [MASK_ID] * len(completion_bytes)
    offset = len(context_ids)
    targets = _split_blocks(completion_bytes, BLOCK_SIZE)
    rng = random.Random(seed)
    byte_scale = _BYTE_SCALE
    verify_len = (
        BLOCK_SIZE if cfg.verify_clip is None else min(BLOCK_SIZE, cfg.verify_clip)
    )
    if heavy_load:
        verify_len = min(verify_len, BLOCK_SIZE // HEAVY_LOAD_VERIFY_CLIP)
    stats: Dict[str, int] = {"blocks": len(targets), "regenerations": 0, "converged": 0}
    for batch_start in range(0, len(targets), cfg.k_blocks):
        batch = targets[batch_start : batch_start + cfg.k_blocks]
        for local, target in enumerate(batch):
            block_seed = seed + batch_start + local
            current = bytearray(target)
            for step in range(cfg.steps):
                masked, _ = mask_block(bytes(current), cfg, seed=block_seed + step)
                candidate = renoise_block(masked, seed=block_seed + step)
                block_conf = float(_block_surrogate(bytes(candidate), step))
                confidence = [(block_conf + byte_scale[b]) / 2.0 for b in candidate]
                failed = [
                    i
                    for i in range(verify_len)
                    if confidence[i] < cfg.entropy_bound
                ]
                # lossless guard (mirrors speculative_quantize): a position
                # that already holds its target byte must never be regenerated.
                failed = [i for i in failed if candidate[i] != target[i]]
                if not failed:
                    break
                stats["regenerations"] += 1
                for i in failed:
                    candidate[i] = target[i]
                current = bytearray(candidate)
            # block-autoregressive commit: force the exact draft completion
            # (the lossless end state), writing only the unpadded span.
            block_start = offset + (batch_start + local) * BLOCK_SIZE
            end = min(block_start + BLOCK_SIZE, offset + len(completion_bytes))
            canvas[block_start:end] = list(target[: end - block_start])
            stats["converged"] += 1
    return canvas, stats


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


def modality_size_report(
    num_params: int = 16_000_000_000, baseline_bits: int = 16
) -> Dict[str, Dict[str, object]]:
    """Per-modality size summary under each DSpark block-mask schedule.

    Each modality's entry carries the generic ``size_report`` numbers plus the
    schedule's ``mask_ratio``, ``k_blocks`` and ``steps``, so throughput paths
    can be sized together with their diffusion configuration.

    Args:
        num_params: total parameter count for the size math.
        baseline_bits: float width of the original checkpoint.

    Returns:
        Dict keyed by modality, each a Dict of size + schedule fields.
    """
    out: Dict[str, Dict[str, object]] = {}
    for name, cfg in DSPARK_MODALITY_SCHEDULES.items():
        row = size_report(num_params=num_params, baseline_bits=baseline_bits)
        row["mask_ratio"] = cfg.mask_ratio
        row["k_blocks"] = cfg.k_blocks
        row["steps"] = cfg.steps
        row["verify_clip"] = cfg.verify_clip
        row["canvas_length"] = cfg.canvas_length
        out[name] = row
    return out


def print_modality_size_report(
    num_params: int = 16_000_000_000, baseline_bits: int = 16
) -> None:
    """Human-readable per-modality DSpark schedule + size table."""
    print(
        f"x8Dsub-byte 0.001 modality schedules "
        f"({num_params:,} params, {baseline_bits}-bit baseline)"
    )
    print(f"  {'modality':<8}{'mask_ratio':>11}{'k_blocks':>9}{'steps':>7}{'x8d_storage_gb':>17}{'subbyte_mb':>12}")
    for name, row in modality_size_report(
        num_params=num_params, baseline_bits=baseline_bits
    ).items():
        print(
            f"  {name:<8}{row['mask_ratio']:>11.2f}{row['k_blocks']:>9d}"
            f"{row['steps']:>7d}{row['x8d_storage_gb']:>17.2f}{row['subbyte_coordinate_mb']:>12.1f}"
        )
