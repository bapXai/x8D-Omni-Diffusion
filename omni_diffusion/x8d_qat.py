# coding=utf-8
"""x8Dsub-byte QAT (Quantization-Aware Training) scaffold — pure stdlib + lazy torch.

Per AGENTS.md "Definitions": QAT = fake quantization in the forward pass —
``x_q = (x/scale + zp).round().clamp()`` then dequantize back, so training sees
the exact numerics the deployed low-bit model will run. The backward pass uses
the straight-through estimator (STE): the zero-a.e. gradient of ``round`` is
replaced by identity so weights co-adapt to quantization noise.

For THIS project QAT means: every forward pass quantizes each weight to its U8
sub-byte coordinate — ``round(clamp(w, 0, 255))`` -> byte 0-255 — with STE, so
training co-adapts weights to the byte domain and the final ``.x8D`` export
(``Quanta[i] = weight_byte[i] * 0.001``, disk = source_bytes x 0.001) is
near-lossless. We do NOT lose precision and do NOT reduce the parameter count —
we add different precision in less size (0.008 bit per weight byte).

Everything in this module is pure Python standard library; torch is imported
lazily and only when an actual ``torch.Tensor`` is handed to ``quantize_ste``.
The byte-diffusion helpers ``mask_canvas``/``renoise_to_random_bytes`` delegate
to ``omni_diffusion.byte_diffusion.ByteDiffusionSampler`` (the single source of
truth; no logic is duplicated here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .byte_diffusion import ByteDiffusionSampler

#: Byte-native vocabulary size: 256 bytes + 8 specials (MASK/PAD/BOS/EOS/IMG/AUD).
VOCAB_SIZE: int = 264

#: x8D U8 byte coordinate axis bounds for the fake-quant op.
U8_MIN: int = 0
U8_MAX: int = 255

#: QAT weight clamp defaults (byte axis 0-255).
DEFAULT_QUANT_CLAMP: Tuple[int, int] = (U8_MIN, U8_MAX)


def _torch():
    """Return the torch module when importable, else None (lazy torch)."""
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _quantize_ste_py(x, lo: int, hi: int):
    """Pure-Python fake-quant: ``float(round(clamp(x, lo, hi)))``.

    Recurses over list/tuple containers so weight lists can be quantized in one
    call. There is no autograd graph in pure Python; the identity backward is
    exposed separately via :func:`ste_grad`.
    """
    if isinstance(x, dict):
        return {name: _quantize_ste_py(v, lo, hi) for name, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_quantize_ste_py(v, lo, hi) for v in x)
    return float(round(min(max(float(x), lo), hi)))


def quantize_ste(x, lo: int = U8_MIN, hi: int = U8_MAX):
    """Fake-quantize to the U8 byte domain with a straight-through estimator.

    The forward returns ``round(clamp(x, lo, hi))`` — the U8 byte coordinate
    that ``.x8D`` storage will later persist — as a float (or tensor). The
    backward replaces the zero-a.e. gradient of ``round`` with identity (STE),
    so weight updates co-adapt to the byte domain.

    For a ``torch.Tensor`` input the classic STE trick is applied:
    ``x.round().clamp(lo, hi) + (x - x.detach())`` — the forward value equals
    the rounded/clamped byte, and the autograd backward is the identity map on
    the raw weight. For plain Python scalars/sequences the same forward
    quantization is returned; :func:`ste_grad` exposes the identity backward.

    Args:
        x: weight value(s) to quantize (float/int scalar, sequence, or a
            ``torch.Tensor`` when torch is installed).
        lo: lower byte coordinate bound (default 0).
        hi: upper byte coordinate bound (default 255).

    Returns:
        The U8 fake-quantized value(s) in the same container type.
    """
    torch = _torch()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.round().clamp(lo, hi) + (x - x.detach())
    return _quantize_ste_py(x, lo, hi)


def hard_quantize(x, lo: int = U8_MIN, hi: int = U8_MAX):
    """Hard (non-STE) fake-quant: no gradient path flows back to the weight.

    The torch variant detaches the rounded/clamped result, so this is the
    ``ste=False`` deploy-time quantization — useful for measuring what the
    model would produce if quantization were applied without co-adaptation.

    Args:
        x: weight value(s) to quantize.
        lo: lower byte coordinate bound (default 0).
        hi: upper byte coordinate bound (default 255).

    Returns:
        The U8 quantized value(s); torch tensors carry no gradient link.
    """
    torch = _torch()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.round().clamp(lo, hi).detach()
    return _quantize_ste_py(x, lo, hi)


def ste_grad(x) -> float:
    """Identity backward of the STE fake-quant (``d/dx ~ 1.0``).

    ``round`` has zero gradient almost everywhere; QAT replaces it with this
    identity so the error gradient flows unchanged to the raw float weight,
    letting the weight cross the rounding boundary during fine-tuning.

    Args:
        x: the forward-pass value (ignored; the STE gradient is constant).

    Returns:
        ``1.0`` — the identity multiplier applied to ``dLoss/dy``.
    """
    return 1.0


class QATWrapper:
    """Shim that runs every weight through :func:`quantize_ste`.

    Accepts a plain ``dict`` of name -> weights OR any module-like object
    exposing ``named_parameters()`` (torch ``nn.Module`` included — torch is
    only imported lazily). ``qat_weights()`` returns the fake-quantized
    weights; ``named_parameters``/``parameters`` expose them as the QAT-visible
    parameters; ``forward`` routes to an optional underlying callable that
    receives the quantized weight dict first.
    """

    def __init__(
        self,
        weights: Union[Dict[str, object], object],
        forward: Optional[object] = None,
        quant_clamp: Tuple[int, int] = DEFAULT_QUANT_CLAMP,
        ste: bool = True,
    ) -> None:
        """Initialize the wrapper.

        Args:
            weights: dict of name -> weight values, or an object exposing
                ``named_parameters()``.
            forward: optional callable invoked as
                ``forward(quantized_weights, *args, **kwargs)``; defaults to the
                wrapped object's ``forward`` when present.
            quant_clamp: ``(lo, hi)`` byte coordinate clamp bounds.
            ste: apply the straight-through fake-quant when True (default),
                hard detached quantization when False.
        """
        raw: Dict[str, object]
        if isinstance(weights, dict):
            raw = dict(weights)
        elif hasattr(weights, "named_parameters"):
            raw = dict(weights.named_parameters())
            if forward is None:
                forward = getattr(weights, "forward", None)
        else:
            raise TypeError("weights must be a dict or expose named_parameters()")
        self._raw: Dict[str, object] = raw
        self._forward: Optional[object] = forward
        self._clamp: Tuple[int, int] = (int(quant_clamp[0]), int(quant_clamp[1]))
        self._ste: bool = bool(ste)

    @property
    def raw_weights(self) -> Dict[str, object]:
        """The original (pre-quantization) weights."""
        return dict(self._raw)

    def qat_weights(self) -> Dict[str, object]:
        """Fake-quantized weights: every tensor/value through ``quantize_ste``.

        Returns:
            Dict mapping each name to its U8 fake-quantized value.
        """
        lo, hi = self._clamp
        fn = quantize_ste if self._ste else hard_quantize
        return {name: fn(value, lo, hi) for name, value in self._raw.items()}

    def named_parameters(self) -> Iterator[Tuple[str, object]]:
        """Iterate over ``(name, quantized_value)`` QAT-visible parameters."""
        for name, value in self.qat_weights().items():
            yield name, value

    def parameters(self) -> List[object]:
        """The QAT-visible (fake-quantized) parameter values."""
        return list(self.qat_weights().values())

    def forward(self, *args, **kwargs) -> object:
        """Run the wrapped forward on the fake-quantized weights.

        The quantized weight dict is passed as the first argument:
        ``forward(quantized_weights, *args, **kwargs)``. With no underlying
        callable, the quantized weight dict is returned directly.
        """
        qw = self.qat_weights()
        if self._forward is not None:
            return self._forward(qw, *args, **kwargs)
        return qw


def wrap_for_qat(
    module_like: Union[Dict[str, object], object],
    quant_clamp: Tuple[int, int] = DEFAULT_QUANT_CLAMP,
    ste: bool = True,
) -> QATWrapper:
    """Wrap a module (or weight dict) so every weight runs through fake-quant.

    Args:
        module_like: dict of name -> weights, or any object exposing
            ``named_parameters()`` (e.g. a torch ``nn.Module``; torch is never
            imported eagerly).
        quant_clamp: ``(lo, hi)`` byte coordinate clamp bounds (default
            ``(0, 255)``).
        ste: apply the STE fake-quant when True (default), hard detached
            quantization when False.

    Returns:
        A :class:`QATWrapper` whose weights are the U8 fake-quantized values.
    """
    return QATWrapper(module_like, quant_clamp=quant_clamp, ste=ste)


def _flatten_values(value) -> List[float]:
    """Flatten a weight value (scalar, nested sequence, or tensor) to floats."""
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return [float(v) for v in value.flatten().detach().cpu().tolist()]
    if isinstance(value, (list, tuple)):
        flat: List[float] = []
        for item in value:
            flat.extend(_flatten_values(item))
        return flat
    return [float(value)]


def _abs_diffs(a, b) -> List[float]:
    """Element-wise absolute differences between two weight collections."""
    fa = _flatten_values(a)
    fb = _flatten_values(b)
    if len(fa) != len(fb):
        raise ValueError(f"weight collections must have equal length, got {len(fa)} vs {len(fb)}")
    return [abs(x - y) for x, y in zip(fa, fb)]


def x8d_qat_roundtrip_loss(weights, quantized_weights) -> float:
    """Mean absolute difference between raw and quantized weights.

    Args:
        weights: dict name -> values, or a flat sequence of values.
        quantized_weights: same shape as ``weights``.

    Returns:
        ``0.0`` for byte-aligned weights (identical to their quantization),
        ``> 0`` otherwise.
    """
    if isinstance(weights, dict) and isinstance(quantized_weights, dict):
        if sorted(weights) != sorted(quantized_weights):
            raise ValueError("weight dicts must have identical keys")
        diffs: List[float] = []
        for name in weights:
            diffs.extend(_abs_diffs(weights[name], quantized_weights[name]))
    else:
        diffs = _abs_diffs(weights, quantized_weights)
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)


def byte_diffusion_loss(
    pred_logits: Sequence[Sequence[float]],
    target_bytes: Sequence[int],
    vocab_size: int = VOCAB_SIZE,
) -> float:
    """Pure-Python cross-entropy over the 264-vocab byte space.

    Computes ``mean(-log softmax(pred_logits)[target])`` with the numerically
    stable softmax (max subtraction), summing ``-log(softmax)`` over all
    positions and normalizing by the position count. This is the byte-diffusion
    denoising loss QAT fine-tuning optimizes: the denoiser must recover the
    masked byte ids over ids 0-255 (specials 256-263 never appear in content).

    Args:
        pred_logits: per-position logits; either ``[n_pos][vocab_size]`` or a
            single flat ``[vocab_size]`` row.
        target_bytes: iterable of target byte ids in ``[0, vocab_size)``.
        vocab_size: byte-native vocabulary size (default 264).

    Returns:
        ``0.0`` for perfectly calibrated (one-hot) logits, ``> 0`` otherwise,
        and lower for better-calibrated logits.
    """
    seq = list(pred_logits)
    if seq and not isinstance(seq[0], (list, tuple)):
        seq = [seq]
    rows = [list(r) for r in seq]
    targets = [int(t) for t in target_bytes]
    if len(rows) != len(targets):
        raise ValueError(
            f"pred_logits has {len(rows)} positions but target_bytes has {len(targets)}"
        )
    total = 0.0
    for logits, target in zip(rows, targets):
        if len(logits) != vocab_size:
            raise ValueError(f"each logits row must have {vocab_size} entries, got {len(logits)}")
        if not 0 <= target < vocab_size:
            raise ValueError(f"target byte must be in [0, {vocab_size}), got {target}")
        peak = max(logits)
        shifted = [v - peak for v in logits]
        log_sum_exp = math.log(sum(math.exp(v) for v in shifted))
        total += log_sum_exp - shifted[target]
    return total / len(rows) if rows else 0.0


def mask_canvas(ids: Sequence[int], mask_ratio: float = 0.7, seed: int = 0):
    """Uniform-state masking; delegates to ``ByteDiffusionSampler.mask_canvas``.

    Args:
        ids: byte-id canvas to mask.
        mask_ratio: fraction of positions masked, in ``[0, 1]``.
        seed: RNG seed (deterministic masking).

    Returns:
        ``(masked, truth)`` — masked ids and the untouched original.
    """
    return ByteDiffusionSampler(seed=seed).mask_canvas(ids, mask_ratio=mask_ratio)


def renoise_to_random_bytes(ids: Sequence[int], seed: int = 0) -> List[int]:
    """Uniform-state re-noise; delegates to ``ByteDiffusionSampler``.

    Args:
        ids: byte-id sequence possibly containing MASK(256).
        seed: RNG seed (deterministic replacement bytes).

    Returns:
        The sequence with every MASK position replaced by a random byte 0-255.
    """
    return ByteDiffusionSampler(seed=seed).renoise_to_random_bytes(ids)


@dataclass(frozen=True)
class QATConfig:
    """Hyper-parameters for x8D QAT fine-tuning.

    Defaults match the AGENTS.md byte-diffusion settings (DiffusionGemma
    parity): ``diffusion_steps=48``, ``entropy_bound=0.1`` and
    ``canvas_length=256``, over the 264-id byte vocabulary.
    """

    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 8
    ste: bool = True
    quant_clamp: Tuple[int, int] = DEFAULT_QUANT_CLAMP
    diffusion_steps: int = 48
    entropy_bound: float = 0.1
    canvas_length: int = 256


#: Lower-case alias for ``QATConfig`` (convenience).
qat_config = QATConfig
