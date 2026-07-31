# coding=utf-8
"""Whole-omni-model parameter + compressed-size comparison (issue #38).

Pure Python stdlib. Computes, for every upstream model in the x8D omni
expert stack, its size under the x8Dsub-byte 0.001 scaling law:

    U8 byte-native weights:     params * 8 * LAW  bit/param = params * 0.001 B
    BF16 pointer-map weights:   params * 16 * LAW bit/param = params * 0.002 B
    serveable size              = U8 (of the U8 portion) + BF16-pointer
                                  (of the BF16 portion) -- mixed-dtype models

Registry values are frozen from web research (2026-07-31, see
``research/Omni-Stack-Parameters-and-Size-2026.md``) plus the repo's verified
pointer-map artifacts (``research/Status-and-Optimization-Audit-2026.md``,
``research/Kimi-K3-x8D-Pointer-Quantization.md``). Rows marked "est." use
computed-from-research values; kimi_k3 uses the exact repo/README values
(pointer 163,374,871 B, upstream 1.56e12 B, total 2.8e12).
"""

from __future__ import annotations

from typing import Dict, List, Optional

#: x8Dsub-byte scaling law (0.001 sub-byte coordinate space).
LAW: float = 0.001
#: U8 byte-native weights: 8 bits * LAW.
BIT_PER_PARAM_U8: float = 8.0 * LAW  # 0.008 bit/param
#: BF16 pointer-map weights: 16 bits * LAW.
BIT_PER_PARAM_PTR_BF16: float = 16.0 * LAW  # 0.016 bit/param

#: Exact pointer-map sizes hosted in ``bapX/x8D-Omni-Diffusion/x8d_weights/``.
KIMI_K3_POINTER_BYTES: int = 163_374_871
WHISPER_POINTER_BYTES: int = 343_642
KOKORO_POINTER_BYTES: int = 171
LTX2_POINTER_BYTES: int = 2_319_390

#: Kimi-K3 exact parameter split (from HF safetensors index, issue #10).
KIMI_K3_TOTAL_PARAMS: int = 2_779_931_837_184
KIMI_K3_U8_PARAMS: int = 2_722_740_830_208
KIMI_K3_BF16_PARAMS: int = 57_179_884_544
KIMI_K3_UPSTREAM_BYTES: int = 1_560_860_324_864


def _estimate_pointer_bytes(total_params: int) -> int:
    """Estimate an X8DPTR01 pointer-map size, scaled from Kimi-K3's map.

    Kimi-K3's 497,220-tensor pointer map is 163,374,871 B for 2.78 T params.
    Models with no quantized artifact yet are estimated proportionally to
    their parameter count; these rows are marked "est." in the report.
    """
    return round(KIMI_K3_POINTER_BYTES * total_params / KIMI_K3_TOTAL_PARAMS)


class OmniSizeReport:
    """Size-comparison registry for the x8D omni expert stack."""

    _TASK_EXPERTS: Dict[str, List[str]] = {
        "asr": ["whisper"],
        "tts": ["kokoro", "kitten_tts"],
        "text": ["glm_5_2", "kimi_k3", "deepseek_v4_pro", "diffusiongemma"],
        "video": ["ltx2"],
        "image": ["ltx2"],
    }

    def __init__(self, registry: Optional[Dict[str, dict]] = None):
        self.models: Dict[str, dict] = registry if registry is not None else self.default_registry()

    @staticmethod
    def default_registry() -> Dict[str, dict]:
        """The 8-model omni stack, keyed by expert name.

        Each record carries ``repo``, ``model_type``, ``total_params``,
        ``active_params``, ``pointer_bytes``, ``upstream_bytes`` and, for
        mixed-dtype models, the optional ``u8_params`` / ``bf16_params``
        split that reproduces Kimi-K3's 2.723 GB + 114.4 MB exactly.
        """
        est_glm = _estimate_pointer_bytes(743_000_000_000)
        est_dsv4 = _estimate_pointer_bytes(1_600_000_000_000)
        est_gemma = _estimate_pointer_bytes(26_000_000_000)
        est_kitten = _estimate_pointer_bytes(80_000_000)
        return {
            "glm_5_2": {
                "repo": "zai-org/GLM-5.2",
                "model_type": "MoE",
                "total_params": 743_000_000_000,
                "active_params": 39_000_000_000,
                "pointer_bytes": est_glm,
                "upstream_bytes": 1_486_000_000_000,  # est. BF16
            },
            "kimi_k3": {
                "repo": "moonshotai/Kimi-K3",
                "model_type": "MoE",
                "total_params": KIMI_K3_TOTAL_PARAMS,
                "active_params": 104_200_000_000,
                "pointer_bytes": KIMI_K3_POINTER_BYTES,
                "upstream_bytes": KIMI_K3_UPSTREAM_BYTES,
                "u8_params": KIMI_K3_U8_PARAMS,
                "bf16_params": KIMI_K3_BF16_PARAMS,
            },
            "deepseek_v4_pro": {
                "repo": "deepseek-ai/DeepSeek-V4-Pro",
                "model_type": "MoE",
                "total_params": 1_600_000_000_000,
                "active_params": 49_000_000_000,
                "pointer_bytes": est_dsv4,
                "upstream_bytes": 865_000_000_000,  # FP4+FP8 on disk
            },
            "diffusiongemma": {
                "repo": "google/diffusiongemma-26B-A4B-it",
                "model_type": "MoE",
                "total_params": 26_000_000_000,
                "active_params": 4_000_000_000,
                "pointer_bytes": est_gemma,
                "upstream_bytes": 52_000_000_000,  # est. BF16
            },
            "kokoro": {
                "repo": "hexgrad/Kokoro-82M",
                "model_type": "dense",
                "total_params": 82_000_000,
                "active_params": 82_000_000,
                "pointer_bytes": KOKORO_POINTER_BYTES,
                "upstream_bytes": 164_000_000,  # est. FP16
            },
            "whisper": {
                "repo": "openai/whisper-large-v3",
                "model_type": "dense",
                "total_params": 1_550_000_000,
                "active_params": 1_550_000_000,
                "pointer_bytes": WHISPER_POINTER_BYTES,
                "upstream_bytes": 3_100_000_000,  # est. FP16
            },
            "ltx2": {
                "repo": "Lightricks/LTX-2",
                "model_type": "dense",
                "total_params": 19_000_000_000,
                "active_params": 19_000_000_000,
                "pointer_bytes": LTX2_POINTER_BYTES,
                "upstream_bytes": 9_500_000_000,  # est. FP4 checkpoint
            },
            "kitten_tts": {
                "repo": "KittenML/kitten-tts-mini-0.8",
                "model_type": "dense",
                "total_params": 80_000_000,
                "active_params": 80_000_000,
                "pointer_bytes": est_kitten,
                "upstream_bytes": 80_000_000,  # ONNX int8 on disk
            },
        }

    # -- the 0.001 sub-byte law -------------------------------------------
    @staticmethod
    def bit_per_param_u8() -> float:
        """Bits per parameter under U8 byte-native weights (8 * LAW)."""
        return BIT_PER_PARAM_U8

    @staticmethod
    def bit_per_param_ptr_bf16() -> float:
        """Bits per parameter under BF16 pointer-map weights (16 * LAW)."""
        return BIT_PER_PARAM_PTR_BF16

    def _params(self, name: str, key: str, default: int) -> int:
        rec = self.models[name]
        return int(rec.get(key, default))

    def _u8_params(self, name: str) -> int:
        """Parameters stored as U8 byte coordinates (default = all)."""
        return self._params(name, "u8_params", self.models[name]["total_params"])

    def _bf16_params(self, name: str) -> int:
        """Parameters stored as BF16 pointer-map coordinates (default = none)."""
        return self._params(name, "bf16_params", 0)

    def size_u8(self, name: str) -> int:
        """U8 compressed size in bytes: u8_params * 0.001 B/param."""
        return round(self._u8_params(name) * BIT_PER_PARAM_U8 / 8.0)

    def size_ptr_bf16(self, name: str) -> int:
        """BF16 pointer-map size in bytes: bf16_params * 0.002 B/param."""
        return round(self._bf16_params(name) * BIT_PER_PARAM_PTR_BF16 / 8.0)

    def serveable_bytes(self, name: str) -> int:
        """Serveable size = U8 + BF16-pointer for one expert (in bytes)."""
        return self.size_u8(name) + self.size_ptr_bf16(name)

    # -- combined totals ----------------------------------------------------
    def combined_total_params(self) -> int:
        return sum(int(r["total_params"]) for r in self.models.values())

    def combined_active_params(self) -> int:
        return sum(int(r["active_params"]) for r in self.models.values())

    def combined_pointer_bytes(self) -> int:
        return sum(int(r["pointer_bytes"]) for r in self.models.values())

    def combined_u8_bytes(self) -> int:
        return sum(self.size_u8(n) for n in self.models)

    def combined_serveable_bytes(self) -> int:
        return sum(self.serveable_bytes(n) for n in self.models)

    def combined_upstream_bytes(self) -> int:
        return sum(int(r["upstream_bytes"]) for r in self.models.values())

    def compression_ratio(self) -> float:
        """Upstream disk vs combined pointer maps (upstream / pointer)."""
        return self.combined_upstream_bytes() / max(1, self.combined_pointer_bytes())

    def serveable_compression_ratio(self) -> float:
        """Upstream disk vs combined serveable sizes (upstream / serveable)."""
        return self.combined_upstream_bytes() / max(1, self.combined_serveable_bytes())

    # -- MoE framing: active params per task ---------------------------------
    def active_params_for(self, task: str) -> Dict[str, int]:
        """Active parameters by expert for a task (each model = one expert).

        Dense models activate their full parameter count; MoE models activate
        only their per-token active slice. Unknown tasks return ``{}``.
        """
        out: Dict[str, int] = {}
        for expert in self._TASK_EXPERTS.get(task, []):
            out[expert] = int(self.models[expert]["active_params"])
        return out

    # -- reporting -----------------------------------------------------------
    def row(self, name: str) -> str:
        rec = self.models[name]
        est = "" if name in ("kimi_k3", "kokoro", "whisper", "ltx2") else " est."
        return (
            f"  {name:<16} {rec['model_type']:<5} "
            f"{self._fmt_params(rec['total_params']):>6} "
            f"{self._fmt_params(rec['active_params']):>6} "
            f"{self._fmt_bytes(rec['upstream_bytes']):>8}{est:>6} "
            f"{self._fmt_bytes(rec['pointer_bytes']):>9}{est:>6} "
            f"{self._fmt_bytes(self.size_u8(name)):>7} "
            f"{self._fmt_bytes(self.serveable_bytes(name)):>7}"
        )

    @staticmethod
    def _fmt_params(n: int) -> str:
        if n >= 1e12:
            return f"{n / 1e12:.2f}T"
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        if n >= 1e6:
            return f"{n / 1e6:.0f}M"
        return f"{n}"

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        if n >= 1e12:
            return f"{n / 1e12:.2f}TB"
        if n >= 1e9:
            return f"{n / 1e9:.2f}GB"
        if n >= 1e6:
            return f"{n / 1e6:.2f}MB"
        if n >= 1e3:
            return f"{n / 1e3:.1f}KB"
        return f"{n}B"

    def table(self) -> List[str]:
        lines = [
            "  model             type  total   active  upstream   pointer map   U8 0.008bit  serveable",
            "  " + "-" * 88,
        ]
        for name in self.models:
            lines.append(self.row(name))
        return lines

    def summary(self) -> List[str]:
        t = self.combined_total_params()
        a = self.combined_active_params()
        u = self.combined_upstream_bytes()
        p = self.combined_pointer_bytes()
        u8 = self.combined_u8_bytes()
        s = self.combined_serveable_bytes()
        return [
            "",
            "  COMBINED (all 8 experts):",
            f"    total params          : {self._fmt_params(t)} ({t:,})",
            f"    active params (sum, all 8 experts): {self._fmt_params(a)} ({a:,})",
            f"    upstream disk         : {self._fmt_bytes(u)} ({u:,} B)",
            f"    pointer maps          : {self._fmt_bytes(p)} ({p:,} B)",
            f"    U8 serveable          : {self._fmt_bytes(u8)} ({u8:,} B)",
            f"    serveable (U8+BF16ptr): {self._fmt_bytes(s)} ({s:,} B)",
            f"    compression vs upstream (pointer maps): {self.compression_ratio():,.0f}:1",
            f"    compression vs upstream (serveable)  : {self.serveable_compression_ratio():,.0f}:1",
        ]


def main() -> None:
    report = OmniSizeReport()
    print("x8D omni-stack parameters & compressed size (issue #38)")
    print("  (est. rows: no pointer artifact yet; value proportional to Kimi-K3's map)")
    print()
    print("\n".join(report.table()))
    print("\n".join(report.summary()))


if __name__ == "__main__":
    main()
