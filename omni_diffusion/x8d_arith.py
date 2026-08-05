# coding=utf-8
"""Pure-stdlib arithmetic (range) coder for the x8D sub-byte 0.001 law.

The sub-byte law maps every weight byte to a coordinate ``Quanta[i] =
byte[i] * 0.001`` (0.008 bit/weight). Storing those coordinates as raw
bytes pads every symbol back to 8 bits. Arithmetic coding stores the whole
coordinate stream as ONE fractional number in [0, 1) whose length converges
to the true entropy of the stream -- fractional bits per symbol, NO padding
to byte boundaries. This is the Python-native way to write the sub-byte
math as-is.

Formula (integer range coder, from the arithmetic-coding literature):

    range = high - low + 1
    high  = low + (range * hi_cum) / total - 1
    low   = low + (range * lo_cum) / total
    renormalize()   # emit/consume settled digits -> fractional output

Pure Python 3.10+ standard library only. No numpy, no torch.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union

#: 32-bit integer state (Top Value must be power of two for renormalization).
STATE_BITS: int = 32
TOP: int = 1 << STATE_BITS
HALF: int = 1 << (STATE_BITS - 1)
QUARTER: int = 1 << (STATE_BITS - 2)

#: Number of byte symbols in the x8D space.
ALPHABET: int = 256

#: Hard ceiling on the cumulative frequency total. Renormalization guarantees
#: the coder span is never below ``QUARTER + 2``, so capping ``total`` at
#: ``QUARTER`` keeps ``span >= total`` at every step, which is the exact
#: condition that keeps every symbol's sub-range from collapsing to zero.
MAX_TOTAL: int = QUARTER


class SubByteFrequencyTable:
    """Fixed symbol model over the 256 byte states.

    ``count[b]`` weights symbol ``b``; ``total`` = sum of counts. The coder
    splits the current range proportionally to these counts, which is what
    gives fractional bits per symbol.
    """

    def __init__(self, counts: Optional[Dict[int, int]] = None) -> None:
        if counts is None:
            counts = {b: 1 for b in range(ALPHABET)}  # uniform model
        # Every symbol keeps a floor weight of 1 so no sub-range ever
        # collapses to zero length (mirrors the classic Laplace/penny
        # prior and keeps the coder total well under the 2^30 headroom).
        weights = [max(1, counts.get(b, 0)) for b in range(ALPHABET)]
        self._cum = [0] * (ALPHABET + 1)
        for b in range(ALPHABET):
            self._cum[b + 1] = self._cum[b] + weights[b]
        self.total = self._cum[-1]
        if self.total > MAX_TOTAL:
            # Scale down but preserve a floor of 1 per symbol.
            scale = float(MAX_TOTAL - ALPHABET) / float(self.total)
            weights = [max(1, int(w * scale)) for w in weights]
            self._cum = [0] * (ALPHABET + 1)
            for b in range(ALPHABET):
                self._cum[b + 1] = self._cum[b] + weights[b]
            self.total = self._cum[-1]

    def low_high(self, symbol: int) -> Tuple[int, int]:
        """Cumulative sub-range [lo, hi) for ``symbol`` (0-255)."""
        return self._cum[symbol], self._cum[symbol + 1]


def _frequency_counts(byte_stream: Iterable[int]) -> Dict[int, int]:
    """Count symbol frequencies from a raw byte stream (adaptive model)."""
    return dict(Counter(int(b) & 0xFF for b in byte_stream))


class Encoder:
    """Fractional-bit encoder: whole byte stream -> one small integer.

    The emitted bit stream has length ~ sum over symbols of -log2(p(symbol)),
    i.e. the sub-byte entropy of the input. No padding is added.
    """

    def __init__(self, freq: SubByteFrequencyTable) -> None:
        self._freq = freq
        self._low: int = 0
        self._high: int = TOP - 1
        self._out: List[int] = []  # bits
        self._underflow: int = 0

    def _renormalize(self) -> None:
        while True:
            if self._high < HALF:
                self._emit(0)
                self._low <<= 1
                self._high = ((self._high << 1) | 1) & (TOP - 1)
            elif self._low >= HALF:
                self._emit(1)
                self._low = (self._low - HALF) << 1
                self._high = ((self._high - HALF) << 1 | 1) & (TOP - 1)
            elif self._low >= QUARTER and self._high < 3 * QUARTER:
                self._underflow += 1
                self._low = (self._low - QUARTER) << 1
                self._high = ((self._high - QUARTER) << 1 | 1) & (TOP - 1)
            else:
                break

    def _emit(self, bit: int) -> None:
        self._out.append(bit)
        while self._underflow:
            self._out.append(bit ^ 1)
            self._underflow -= 1

    def encode_symbol(self, symbol: int) -> None:
        lo, hi = self._freq.low_high(symbol)
        total = self._freq.total
        span = self._high - self._low + 1
        self._high = self._low + ((span * hi) // total) - 1
        self._low = self._low + ((span * lo) // total)
        self._renormalize()

    def finish(self) -> bytes:
        """Terminate: emit enough bits to disambiguate the final range."""
        self._underflow += 1
        if self._low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        if self._out:
            n = (len(self._out) + 7) // 8
            out = bytearray(n)
            for i, bit in enumerate(self._out[: n * 8]):
                if bit:
                    out[i >> 3] |= 1 << (7 - (i & 7))
            return bytes(out)
        return b""

    def nbits(self) -> int:
        """Number of emitted fractional bits (before byte packing)."""
        return len(self._out)


class Decoder:
    """Fractional-bit decoder mirroring :class:`Encoder` exactly."""

    def __init__(self, freq: SubByteFrequencyTable, data: bytes) -> None:
        self._freq = freq
        self._low: int = 0
        self._high: int = TOP - 1
        self._code: int = 0
        self._bits = [(b >> (7 - i)) & 1 for b in data for i in range(8)]
        self._pos: int = 0
        for _ in range(STATE_BITS):
            self._code = ((self._code << 1) | self._next_bit()) & (TOP - 1)

    def _next_bit(self) -> int:
        if self._pos < len(self._bits):
            b = self._bits[self._pos]
            self._pos += 1
            return b
        return 0  # pad with zeros past the end (termination handled by encoder)

    def _renormalize(self) -> None:
        while True:
            if self._high < HALF:
                self._low <<= 1
                self._high = ((self._high << 1) | 1) & (TOP - 1)
                self._code = ((self._code << 1) | self._next_bit()) & (TOP - 1)
            elif self._low >= HALF:
                self._low = (self._low - HALF) << 1
                self._high = ((self._high - HALF) << 1 | 1) & (TOP - 1)
                self._code = ((self._code - HALF) << 1 | self._next_bit()) & (TOP - 1)
            elif self._low >= QUARTER and self._high < 3 * QUARTER:
                self._low = (self._low - QUARTER) << 1
                self._high = ((self._high - QUARTER) << 1 | 1) & (TOP - 1)
                self._code = (self._code - QUARTER) << 1
                self._code = ((self._code | self._next_bit()) & (TOP - 1))
            else:
                break

    def decode_symbol(self) -> int:
        total = self._freq.total
        span = self._high - self._low + 1
        target = (((self._code - self._low + 1) * total) - 1) // span
        lo, hi = 0, ALPHABET
        cum = self._freq._cum
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cum[mid] > target:
                hi = mid
            else:
                lo = mid
        symbol = lo
        slo, shi = self._freq.low_high(symbol)
        self._high = self._low + ((span * shi) // total) - 1
        self._low = self._low + ((span * slo) // total)
        self._renormalize()
        return symbol


def arith_encode(byte_stream: Iterable[int], adaptive: bool = False) -> bytes:
    """Arithmetically code a raw byte stream into sub-byte fractional bits.

    Args:
        byte_stream: iterable of byte values (0-255).
        adaptive: if True, build the model from the stream's own frequencies
            (must be supplied back to the decoder); if False, uniform model.

    Returns:
        Packed fractional-bit representation (no per-symbol padding).
    """
    stream = list(byte_stream)
    if adaptive:
        freq = SubByteFrequencyTable(_frequency_counts(stream))
    else:
        freq = SubByteFrequencyTable()  # uniform 256-symbol model
    enc = Encoder(freq)
    for b in stream:
        enc.encode_symbol(int(b) & 0xFF)
    return enc.finish()


def arith_decode(data: bytes, count: int, freq: Optional[SubByteFrequencyTable] = None) -> List[int]:
    """Decode ``count`` symbols from an arithmetic-coded byte string.

    Args:
        data: packed fractional-bit stream.
        count: number of symbols to recover.
        freq: the exact model used at encode time (required when adaptive).

    Returns:
        List of reconstructed byte values (length ``count``).
    """
    if freq is None:
        freq = SubByteFrequencyTable()
    dec = Decoder(freq, data)
    return [dec.decode_symbol() for _ in range(count)]


def subbyte_entropy(byte_stream: Iterable[int]) -> float:
    """Shannon entropy of a byte stream in bits per symbol.

    Args:
        byte_stream: iterable of byte values.

    Returns:
        Bits per symbol (the target the arithmetic coder converges to).
    """
    counts = Counter(int(b) & 0xFF for b in byte_stream)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    from math import log2

    return -sum((c / n) * log2(c / n) for c in counts.values())


def size_for_byte_stream(byte_stream: Iterable[int]) -> Tuple[int, int, float]:
    """(original bytes, coded bytes, bits/symbol) for a byte stream.

    Args:
        byte_stream: iterable of byte values.

    Returns:
        ``(n_original, n_coded, bits_per_symbol)``.
    """
    stream = list(byte_stream)
    coded = arith_encode(stream, adaptive=True)
    return len(stream), len(coded), (len(coded) * 8.0) / max(1, len(stream))
