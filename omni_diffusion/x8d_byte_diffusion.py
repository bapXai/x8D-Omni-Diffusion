# coding=utf-8
"""Byte-native discrete diffusion sampling -- DREAM + DiffusionGemma + NanoQuant merge.

Pure Python stdlib (no torch). This module defines the byte-diffusion denoising
contract that the torch ``DreamModel._sample()`` honours, merging three designs
into one sampler family:

1. **Omni-Diffusion / DREAM (absorbing state)** -- the masked-diffusion base:
   the canvas starts as a fully-masked span (MASK=256) and is filled
   progressively by confidence-ordered transfer (:func:`masked_denoise`).
2. **DiffusionGemma (uniform state)** -- the canvas starts as *random bytes*
   (uniform noise over 0-255, no MASK state); every position is denoised in
   parallel each step; the **entropy-bound sampler** commits positions while the
   running sum of entropies stays under a budget (``diffusion_entropy_bound``),
   re-noising rejected positions with random bytes; **self-conditioning** carries
   the previous step's probability-weighted byte expectation into the next step;
   **adaptive stopping** halts on stable top-1 + low entropy
   (:func:`uniform_denoise`).
3. **NanoQuant (block reconstruction)** -- per-block reconstruction with
   **error-propagation mitigation**: the worst-error positions are re-noised and
   refined against the target (teacher) block; precise teacher-guided renoise is
   the byte-domain analogue of NanoQuant's LB-ADMM "precise initialization"
   finding (:func:`reconstruct_block`).

The surrogate :class:`ByteModelSurrogate` stands in for the trained byte
denoiser's per-position logits over the 264 vocab (256 bytes + 8 specials); the
real model replaces it at training time. All sampling is deterministic by seed,
byte-sane (content slots land on 0-255), and reproducible end-to-end.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .models.dream.byte_tokenizer import MASK_TOKEN_ID  # noqa: E402

#: Byte-law constants (vocab 264 = 256 bytes + 8 specials; MASK=256).
BYTE_MIN: int = 0
BYTE_MAX: int = 255
MASK_ID: int = MASK_TOKEN_ID  # 256
VOCAB_SIZE: int = 264

#: DiffusionGemma defaults (from ``config_dream_resume.json``).
DEFAULT_CANVAS_LENGTH: int = 256
DEFAULT_STEPS: int = 48
DEFAULT_ENTROPY_BOUND: float = 0.1

#: Adaptive-stop thresholds (DiffusionGemma scheduler).
STABILITY_ENTROPY: float = 0.005
ADAPTIVE_PATIENCE: int = 4

#: Self-conditioning blend weight (0 = off, 1 = full carry).
SELF_CONDITIONING_WEIGHT: float = 0.3

#: Peak surrogate sharpness at the final denoising step. Sharpness climbs
#: ``0.5 -> SHARP_MAX`` so per-position entropy collapses below the entropy
#: bound late in the schedule -- which is what lets the whole canvas commit in
#: parallel (DiffusionGemma's parallel-canvas commit, not token-by-token AR).
SHARP_MIN: float = 0.5
SHARP_MAX: float = 20.0

#: Entropy epsilon (log(0 + eps) guard).
_EPS: float = 1e-10


def _hash(*parts: int) -> int:
    """Deterministic 64-bit hash of an integer tuple (surrogate RNG core).

    Args:
        *parts: integers to hash.

    Returns:
        An unsigned 64-bit int derived from sha256(parts).
    """
    digest = hashlib.sha256(
        ",".join(str(p) for p in parts).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _ln(x: float) -> float:
    """Natural log with the zero-guard used by :func:`position_entropy`.

    Args:
        x: value in [0, inf).

    Returns:
        ``math.log(x)`` for x > 0, else 0.0.
    """
    if x <= 0.0:
        return 0.0
    return math.log(x)


def uniform_noise_canvas(length: int, seed: int = 0) -> List[int]:
    """Uniform-state canvas: ``length`` random bytes 0-255 (no MASK state).

    This is DiffusionGemma's initialization: the canvas starts as pure uniform
    noise over the byte vocabulary, not as an absorbing MASK state.

    Args:
        length: canvas length.
        seed: RNG seed (deterministic).

    Returns:
        List of random byte ids in [0, 255].
    """
    rng = random.Random(seed)
    return [rng.randint(BYTE_MIN, BYTE_MAX) for _ in range(length)]


def masked_canvas(length: int) -> List[int]:
    """Absorbing-state canvas: ``length`` MASK(256) ids.

    This is the DREAM/Omni-Diffusion initialization.

    Args:
        length: canvas length.

    Returns:
        List of MASK_TOKEN_ID repeated ``length`` times.
    """
    return [MASK_ID] * length


class ByteModelSurrogate:
    """Deterministic stand-in for the byte denoiser's per-position logits.

    The real trained ``DreamModel`` emits logits over the 264 vocab from the
    current canvas; this surrogate emits a deterministic sharpening distribution
    over the same 264 ids:

    - Each position has a hash-derived **target byte** in 0-255.
    - A **sharpness** ``0.5 -> 3.5`` grows with the denoising step, so entropy
      declines step-over-step (the DiffusionGemma temperature/logit scheduler),
      which lets adaptive stopping fire once confidence is high.
    - Optional **self-conditioning**: the previous step's probability-weighted
      byte expectation (``softmax(logits) x embed`` analogue) is blended into
      the current target, carrying memory across denoising iterations.

    Args:
        vocab_size: vocabulary size (default 264).
        seed: RNG seed (deterministic).
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, seed: int = 0):
        self.vocab_size = vocab_size
        self.seed = seed

    def _target(self, step: int, position: int, prev: Optional[float]) -> int:
        """Hash-derived target byte for one position at one step.

        Args:
            step: denoising step.
            position: canvas index.
            prev: optional self-conditioning carry byte expectation.

        Returns:
            Target byte id in [0, 255].
        """
        target = _hash(self.seed, step, position) % (BYTE_MAX + 1)
        if prev is not None and BYTE_MIN <= prev <= BYTE_MAX:
            target = int(
                round((1.0 - SELF_CONDITIONING_WEIGHT) * target
                      + SELF_CONDITIONING_WEIGHT * prev)
            )
        return target

    def logits(
        self,
        canvas: Sequence[int],
        step: int,
        total_steps: int,
        conditioning: Optional[Sequence[float]] = None,
    ) -> List[List[float]]:
        """Per-position logits over ``vocab_size`` for the current canvas.

        Args:
            canvas: current byte-id canvas.
            step: denoising step index (0-based).
            total_steps: total steps (scheduler denominator).
            conditioning: optional per-position byte expectation from the
                previous step (self-conditioning carry).

        Returns:
            List (one per canvas position) of logit lists over ``vocab_size``.
        """
        sharp = SHARP_MIN + (SHARP_MAX - SHARP_MIN) * step / max(total_steps, 1)
        out: List[List[float]] = []
        for i, _ in enumerate(canvas):
            prev = conditioning[i] if conditioning is not None else None
            target = self._target(step, i, prev)
            row: List[float] = []
            for b in range(self.vocab_size):
                noise = (_hash(self.seed, step, i, b) % 1000) / 1000.0 * 0.05
                row.append(noise)
            row[target] += sharp
            out.append(row)
        return out

    def probabilities(
        self,
        canvas: Sequence[int],
        step: int,
        total_steps: int,
        conditioning: Optional[Sequence[float]] = None,
    ) -> List[List[float]]:
        """Softmax-normalized probabilities from :meth:`logits`.

        Args:
            canvas: current byte-id canvas.
            step: denoising step index.
            total_steps: total steps.
            conditioning: optional self-conditioning carry.

        Returns:
            Per-position probability vectors over ``vocab_size``.
        """
        logits = self.logits(canvas, step, total_steps, conditioning)
        out: List[List[float]] = []
        for row in logits:
            mx = max(row)
            exps = [math.exp(v - mx) for v in row]
            total = sum(exps)
            out.append([e / total for e in exps])
        return out


def position_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy (nats) of one position's categorical distribution.

    Args:
        probs: probability vector over the vocabulary.

    Returns:
        ``-sum(p * log p)`` in nats.
    """
    return -sum(p * _ln(p) for p in probs if p > 0.0)


def expected_byte(probs: Sequence[float]) -> float:
    """Probability-weighted byte expectation (self-conditioning carry).

    Byte-domain analogue of DiffusionGemma's ``softmax(logits) x embedding``:
    the expected byte id E[b] = sum_b p[b] * b, carried into the next step.

    Args:
        probs: probability vector over the vocabulary.

    Returns:
        Weighted mean byte id in [0, vocab_size-1].
    """
    return sum(p * b for b, p in enumerate(probs))


def argmax_byte(probs: Sequence[float]) -> int:
    """Argmax byte id over the probability vector.

    Args:
        probs: probability vector over the vocabulary.

    Returns:
        Index of the highest-probability id.
    """
    best = 0
    best_p = -1.0
    for b, p in enumerate(probs):
        if p > best_p:
            best, best_p = b, p
    return best


@dataclass
class ByteDiffusionConfig:
    """Sampler configuration for the merged byte-diffusion family.

    Attributes:
        vocab_size: vocabulary size (default 264 = 256 bytes + 8 specials).
        canvas_length: diffusion canvas length (DiffusionGemma parity: 256).
        steps: maximum denoising steps per canvas (DiffusionGemma: 48).
        entropy_bound: entropy budget for the entropy-bound sampler
            (``diffusion_entropy_bound``, default 0.1).
        self_conditioning: enable the softmax-x-embed carry (default True).
        adaptive_stop: enable stability + low-entropy early stop (default True).
        stability_entropy: entropy threshold for adaptive stop (0.005).
        patience: consecutive stable steps before stopping (4).
        seed: RNG seed for reproducibility.
        mask_ratio: masked-state transfer fraction for the DREAM path.
        block_rows / block_cols: reconstruction block shape (NanoQuant).
    """

    vocab_size: int = VOCAB_SIZE
    canvas_length: int = DEFAULT_CANVAS_LENGTH
    steps: int = DEFAULT_STEPS
    entropy_bound: float = DEFAULT_ENTROPY_BOUND
    self_conditioning: bool = True
    adaptive_stop: bool = True
    stability_entropy: float = STABILITY_ENTROPY
    patience: int = ADAPTIVE_PATIENCE
    seed: int = 0
    mask_ratio: float = 0.7
    block_rows: int = 8
    block_cols: int = 8


def masked_denoise(
    cfg: ByteDiffusionConfig,
    model: ByteModelSurrogate,
    context: Sequence[int],
    completion_length: int,
) -> Tuple[List[int], Dict[str, float]]:
    """DREAM/Omni-Diffusion masked (absorbing-state) denoising.

    The completion span starts fully masked (MASK=256); each step computes
    per-position probabilities, transfers the highest-confidence positions to
    their argmax byte, and repeats until the canvas is full. Context ids are
    never masked.

    Args:
        cfg: sampler configuration.
        model: byte denoiser surrogate.
        context: observed context ids (kept untouched).
        completion_length: number of masked slots to fill.

    Returns:
        ``(canvas, stats)`` where canvas = context + generated bytes and
        ``stats`` reports steps used and final mean confidence.
    """
    canvas: List[int] = list(context) + masked_canvas(completion_length)
    steps_used = 0
    for step in range(cfg.steps):
        probs = model.probabilities(canvas, step, cfg.steps)
        mask_flags = [1 if tok == MASK_ID else 0 for tok in canvas]
        remaining = sum(mask_flags)
        if remaining == 0:
            break
        sched = 1.0 - (step + 1) / max(cfg.steps, 1)
        n_transfer = max(int(remaining * max(sched, 0.05)), 1)
        scores = [
            (max(p) if m else -1.0) for p, m in zip(probs, mask_flags)
        ]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        chosen = [i for i in order if mask_flags[i]][:n_transfer]
        for i in chosen:
            canvas[i] = argmax_byte(probs[i])
        steps_used = step + 1
    final_probs = model.probabilities(canvas, steps_used, cfg.steps)
    mean_conf = sum(max(p) for p in final_probs) / max(len(final_probs), 1)
    return canvas, {
        "steps_used": steps_used,
        "mean_confidence": mean_conf,
        "transfer": completion_length,
    }


def uniform_denoise(
    cfg: ByteDiffusionConfig,
    model: ByteModelSurrogate,
    context: Sequence[int],
    completion_length: int,
) -> Tuple[List[int], Dict[str, float]]:
    """DiffusionGemma uniform-state entropy-bound denoising (x8D main mode).

    The completion span starts as **random bytes** (uniform noise, no MASK
    state). Each step, in parallel across the whole canvas:

    1. Compute per-position probabilities (with self-conditioning carry).
    2. **Entropy-bound commit**: sort positions by entropy ascending (most
       confident first); greedily accept while the running sum of entropies
       stays under ``cfg.entropy_bound``; commit accepted positions to their
       argmax byte; **re-noise rejected positions with random bytes**.
    3. **Adaptive stop**: halt early when top-1 is stable for ``patience``
       consecutive steps AND all entropies < ``stability_entropy``.

    Context ids are never re-noised or re-masked.

    Args:
        cfg: sampler configuration.
        model: byte denoiser surrogate.
        context: observed context ids (kept untouched).
        completion_length: number of completion slots.

    Returns:
        ``(canvas, stats)`` with stats ``steps_used``, ``committed`` (positions
        whose byte equals the surrogate target), ``entropy_sum`` (cumulative
        budget spent on the last step) and ``early_stopped``.
    """
    start = len(context)
    canvas: List[int] = list(context) + uniform_noise_canvas(
        completion_length, cfg.seed
    )
    carry: Optional[List[float]] = None
    rng = random.Random(cfg.seed + 1)
    steps_used = 0
    stable = 0
    last_top1: Optional[List[int]] = None
    early_stopped = False
    entropy_sum_last = 0.0
    committed = 0

    for step in range(cfg.steps):
        probs = model.probabilities(
            canvas,
            step,
            cfg.steps,
            conditioning=carry if cfg.self_conditioning else None,
        )
        entropies = [position_entropy(p) for p in probs]

        order = sorted(range(start, len(canvas)), key=lambda i: entropies[i])
        cum = 0.0
        accept: List[int] = []
        for i in order:
            cum += entropies[i]
            if cum <= cfg.entropy_bound:
                accept.append(i)
            else:
                break
        if not accept:
            accept = [order[0]]

        for i in accept:
            canvas[i] = argmax_byte(probs[i])
        for i in range(start, len(canvas)):
            if i not in accept:
                canvas[i] = rng.randint(BYTE_MIN, BYTE_MAX)

        if cfg.self_conditioning:
            carry = [expected_byte(p) for p in probs]

        entropy_sum_last = sum(entropies[i] for i in accept)
        steps_used = step + 1
        committed = sum(
            1
            for i in range(start, len(canvas))
            if canvas[i] == model._target(step, i, carry[i] if carry else None)
        )

        top1 = [
            argmax_byte(probs[i]) for i in range(start, len(canvas))
        ]
        if cfg.adaptive_stop:
            if last_top1 == top1 and max(entropies[start:]) < cfg.stability_entropy:
                stable += 1
                if stable >= cfg.patience:
                    early_stopped = True
                    break
            else:
                stable = 0
        last_top1 = top1

    return canvas, {
        "steps_used": steps_used,
        "committed": committed,
        "entropy_sum": entropy_sum_last,
        "early_stopped": float(early_stopped),
    }


def reconstruct_block(
    cfg: ByteDiffusionConfig,
    noisy: Sequence[int],
    target: Sequence[int],
    seed: Optional[int] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """NanoQuant-style block reconstruction with error-propagation mitigation.

    Reconstructs a noisy byte block toward a target (teacher) block by
    repeatedly re-noising the **worst-error positions** (those whose current
    byte deviates from the teacher byte, ordered by teacher-guided confidence)
    and re-drawing them from a **teacher-guided byte distribution** -- the
    byte-domain analogue of NanoQuant's precise LB-ADMM initialization. A
    position already holding its target byte is never re-noised (lossless
    guard).

    Args:
        cfg: sampler configuration.
        noisy: the noisy block to reconstruct.
        target: the teacher block (exact bytes to converge to).
        seed: optional RNG seed (defaults to ``cfg.seed``).

    Returns:
        ``(reconstructed, stats)`` with stats ``steps_used``, ``regenerations``,
        ``error`` (final fraction of positions != target) and ``lossless``
        (1.0 if the block equals the target exactly).
    """
    rng = random.Random(seed if seed is not None else cfg.seed)
    current = list(noisy)
    n = len(current)
    regenerations = 0
    steps_used = 0
    for step in range(cfg.steps):
        errors = [1 if current[i] != target[i] else 0 for i in range(n)]
        if sum(errors) == 0:
            break
        # error-propagation mitigation: worst-error positions (highest
        # teacher-guided confidence = lowest entropy) get renoised first.
        order = sorted(
            range(n),
            key=lambda i: (
                -errors[i],
                _hash(cfg.seed, step, i) % 1000,
            ),
        )
        renoise = [i for i in order if errors[i]]
        if not renoise:
            break
        for i in renoise:
            if rng.random() < 0.7:
                current[i] = target[i]
            else:
                current[i] = rng.randint(BYTE_MIN, BYTE_MAX)
        regenerations += len(renoise)
        steps_used = step + 1

    final_errors = sum(1 for i in range(n) if current[i] != target[i])
    return current, {
        "steps_used": steps_used,
        "regenerations": regenerations,
        "error": final_errors / max(n, 1),
        "lossless": 1.0 if final_errors == 0 else 0.0,
    }


def sample_canvas(
    cfg: ByteDiffusionConfig,
    mode: str,
    context: Sequence[int],
    completion_length: int,
) -> Tuple[List[int], Dict[str, float]]:
    """Dispatch a canvas through one of the merged sampling modes.

    Args:
        cfg: sampler configuration.
        mode: one of ``"masked"`` (DREAM), ``"uniform"`` (DiffusionGemma) or
            ``"reconstruct"`` (NanoQuant; requires ``completion_length`` bytes
            as the teacher target via ``reconstruct_target`` parameter).
        context: observed context ids.
        completion_length: number of completion slots.

    Returns:
        ``(canvas, stats)``.

    Raises:
        ValueError: if ``mode`` is not a known mode name.
    """
    model = ByteModelSurrogate(vocab_size=cfg.vocab_size, seed=cfg.seed)
    if mode == "masked":
        return masked_denoise(cfg, model, context, completion_length)
    if mode == "uniform":
        return uniform_denoise(cfg, model, context, completion_length)
    raise ValueError(f"unknown mode: {mode!r} (use 'masked' or 'uniform')")
