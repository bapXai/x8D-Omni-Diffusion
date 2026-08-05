# coding=utf-8
"""Real on-container expert serving (issue #48).

Serves generation from the **x8D quantized weights** (X8DGGUF1 container),
never from a full float model. The container is mmap'd; the exact quantized
spans are sliced out and /0.001-reversed live to torch tensors at query time
(the serving law). The full checkpoint is used only once at quantization
time and deleted.

Experts:
- ``KokoroTTS`` (kokoro-82m): real text-to-speech, 82M params, runs on
  CPU/MPS. Weights live in ``kokoro.x8dgguf``.
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Dict, Optional, Tuple

from .x8d_quanta import (
    LAW,
    QuantizedServingReader,
    mmap_gguf,
)

_HAVE_TORCH = True
try:
    import torch  # noqa: E402
except Exception:  # pragma: no cover - torch optional
    _HAVE_TORCH = False


def _kokoro_code_path() -> Optional[str]:
    """Locate the Kokoro model code directory (hexgrad/Kokoro clone)."""
    env = os.environ.get("KOKORO_CODE")
    if env and os.path.isdir(env):
        return env
    for cand in ("/tmp/Kokoro", os.path.expanduser("~/Kokoro")):
        if os.path.isdir(cand):
            return cand
    return None


class KokoroTTS:
    """Real Kokoro-82M TTS served from an x8D quantized container.

    Args:
        gguf_path: path to the quantized ``kokoro.x8dgguf`` container.
        config_path: path to Kokoro ``config.json`` (architecture + vocab).
        voice_path: optional path to a ``.pt`` voice embedding file.
        device: torch device (default auto: mps/cuda/cpu).
    """

    def __init__(
        self,
        gguf_path: str,
        config_path: str,
        voice_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if not _HAVE_TORCH:
            raise RuntimeError("KokoroTTS requires torch (not installed)")
        code = _kokoro_code_path()
        if code is None:
            raise RuntimeError("Kokoro model code not found (set KOKORO_CODE=...)")
        if code not in sys.path:
            sys.path.insert(0, code)
        self._device = device or (
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._config = json.load(open(config_path, encoding="utf-8"))
        self._voice = None
        if voice_path:
            self._voice = torch.load(voice_path, map_location="cpu", weights_only=True)
        self._model = self._build_model(gguf_path).to(self._device).eval()

    def _build_model(self, gguf_path: str):
        """Construct the real Kokoro architecture and load quantized weights.

        The weights come from the mmap'd x8D container (live /0.001 reverse);
        the architecture is built from config.json only. No full model file.
        """
        import types

        # kokoro.pipeline imports misaki.espeak -> phonemizer (espeak-ng).
        # We only need the English G2P, so stub the espeak backend out.
        for name in (
            "phonemizer",
            "phonemizer.backend",
            "phonemizer.backend.espeak",
            "phonemizer.backend.espeak.wrapper",
            "espeakng_loader",
        ):
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)

        class _EspeakStub:
            @staticmethod
            def set_library(*_a, **_k):
                return None

            @staticmethod
            def set_data_path(*_a, **_k):
                return None

        sys.modules["phonemizer.backend.espeak.wrapper"].EspeakWrapper = _EspeakStub
        sys.modules["espeakng_loader"].get_library_path = lambda: "/dev/null"
        sys.modules["espeakng_loader"].get_data_path = lambda: "/dev/null"

        try:
            import kokoro.models as kmodel  # type: ignore[import-not-found]
        except ImportError:
            import kokoro.model as kmodel  # type: ignore[import-not-found]

        mapping = mmap_gguf(gguf_path)
        reader = QuantizedServingReader(gguf_path)
        blob = reader.tensor_bytes()
        model = kmodel.KModel.__new__(kmodel.KModel)
        torch.nn.Module.__init__(model)
        cfg = self._config
        from transformers import AlbertConfig

        model.vocab = cfg["vocab"]
        model.bert = kmodel.CustomAlbert(
            AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
        )
        model.bert_encoder = torch.nn.Linear(
            model.bert.config.hidden_size, cfg["hidden_dim"]
        )
        model.context_length = model.bert.config.max_position_embeddings
        model.predictor = kmodel.ProsodyPredictor(
            style_dim=cfg["style_dim"],
            d_hid=cfg["hidden_dim"],
            nlayers=cfg["n_layer"],
            max_dur=cfg["max_dur"],
            dropout=cfg["dropout"],
        )
        model.text_encoder = kmodel.TextEncoder(
            channels=cfg["hidden_dim"],
            kernel_size=cfg["text_encoder_kernel_size"],
            depth=cfg["n_layer"],
            n_symbols=cfg["n_token"],
        )
        model.decoder = kmodel.Decoder(
            dim_in=cfg["hidden_dim"],
            style_dim=cfg["style_dim"],
            dim_out=cfg["n_mels"],
            **cfg["istftnet"],
        )
        submodules = {
            "bert": model.bert,
            "bert_encoder": model.bert_encoder,
            "predictor": model.predictor,
            "text_encoder": model.text_encoder,
            "decoder": model.decoder,
        }
        offset = 0
        for prefix, mod in submodules.items():
            sub_sd = {}
            for k, v in mod.state_dict().items():
                nbytes = v.numel()
                chunk = blob[offset : offset + nbytes]
                offset += nbytes
                sub_sd[k] = torch.frombuffer(chunk, dtype=torch.uint8).float().reshape(v.shape)
            mod.load_state_dict(sub_sd, strict=True)
        if offset != len(blob):
            raise ValueError(
                f"quantized container byte span mismatch: consumed {offset} of {len(blob)} bytes"
            )
        return model

    @torch.no_grad()
    def synth(self, text: str, speed: float = 1.0) -> Tuple["object", int]:
        """Synthesize real speech for ``text`` (real Kokoro forward pass).

        Args:
            text: prompt text.
            speed: speaking rate.

        Returns:
            ``(audio_tensor, sample_rate)`` — mono float tensor + 24000 Hz.
        """
        import misaki  # type: ignore

        g2p = misaki.en.G2P(trf=False, fallback=None, unk="")
        _, tokens = g2p(text)
        vocab = self._config.get("vocab") or {}
        phonemes = ""
        for w in tokens:
            for t in (w if isinstance(w, list) else [w]):
                if t.phonemes is None:
                    continue
                prespace = bool((t._ or {}).get("prespace", False))
                next_ps = (
                    " " if prespace and phonemes and not phonemes.endswith(" ") else ""
                )
                next_ps += "".join(
                    p for p in t.phonemes.replace("ɾ", "T") if p in vocab
                )
                next_ps += " " if t.whitespace else ""
                phonemes += next_ps
        input_ids = [vocab[p] for p in phonemes.split() if p in vocab]
        if not input_ids:
            raise ValueError(f"no phonemes in vocab for: {text!r}")
        if self._voice is None:
            raise ValueError("a voice embedding is required (voice_path)")
        ref_s = self._voice[len(input_ids) - 1].to(self._device)
        audio = torch.from_numpy(self._model.forward(input_ids, ref_s, speed=speed))
        return audio, 24000

    def synth_bytes(self, text: str, sample_rate: int = 24000) -> bytes:
        """Synthesize and return 16-bit PCM bytes (playable, no file)."""
        import numpy as np

        audio, _ = self.synth(text)
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        audio = audio.detach().cpu().numpy().astype(np.float32)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        return pcm.tobytes()


def build_kokoro(
    gguf_path: str,
    config_path: str,
    voice_path: Optional[str] = None,
    device: Optional[str] = None,
) -> KokoroTTS:
    """Convenience constructor for the real Kokoro TTS expert."""
    return KokoroTTS(gguf_path, config_path, voice_path, device)
