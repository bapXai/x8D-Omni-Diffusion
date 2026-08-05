# coding=utf-8
"""Tests for the pure-stdlib arithmetic coder (x8D sub-byte 0.001 law).

Verifies the fractional-bit range coder: uniform and adaptive round-trips,
reproducibility, edge cases (empty, length-1, all-identical, full alphabet),
sub-byte compression of skewed streams, and the frequency-table cap that
guarantees no symbol sub-range can ever collapse to zero.
"""

import random
import unittest
from collections import Counter

from omni_diffusion.x8d_arith import (
    ALPHABET,
    MAX_TOTAL,
    Decoder,
    Encoder,
    SubByteFrequencyTable,
    arith_decode,
    arith_encode,
    size_for_byte_stream,
    subbyte_entropy,
)


def _adaptive_freq(stream):
    return SubByteFrequencyTable(Counter(int(b) & 0xFF for b in stream))


class UniformRoundTripTest(unittest.TestCase):
    def test_random_1000_seed7(self):
        random.seed(7)
        d = [random.randrange(256) for _ in range(1000)]
        self.assertEqual(arith_decode(arith_encode(d), len(d)), d)

    def test_empty_input(self):
        self.assertEqual(arith_decode(arith_encode([]), 0), [])

    def test_length_one(self):
        for b in (0, 1, 7, 128, 255):
            self.assertEqual(arith_decode(arith_encode([b]), 1), [b])

    def test_all_identical_bytes(self):
        d = [42] * 500
        self.assertEqual(arith_decode(arith_encode(d), len(d)), d)

    def test_full_alphabet(self):
        d = list(range(256)) * 4
        self.assertEqual(arith_decode(arith_encode(d), len(d)), d)

    def test_reproducible_output(self):
        d = [random.randrange(256) for _ in range(200)]
        self.assertEqual(arith_encode(d), arith_encode(d))

    def test_skewed_high_entropy_uniform(self):
        d = [0] * 400 + [1, 2, 3] * 20
        self.assertEqual(arith_decode(arith_encode(d), len(d)), d)


class AdaptiveRoundTripTest(unittest.TestCase):
    def test_random_round_trip(self):
        random.seed(11)
        d = [random.randrange(256) for _ in range(500)]
        freq = _adaptive_freq(d)
        self.assertEqual(arith_decode(arith_encode(d, adaptive=True), len(d), freq), d)

    def test_skewed_round_trip(self):
        d = [0] * 900 + list(range(1, 256))
        freq = _adaptive_freq(d)
        self.assertEqual(arith_decode(arith_encode(d, adaptive=True), len(d), freq), d)

    def test_all_identical_round_trip(self):
        d = [7] * 300
        freq = _adaptive_freq(d)
        self.assertEqual(arith_decode(arith_encode(d, adaptive=True), len(d), freq), d)


class SubByteCompressionTest(unittest.TestCase):
    def test_skewed_codes_to_fewer_bytes(self):
        d = [0] * 1000
        freq = _adaptive_freq(d)
        coded = arith_encode(d, adaptive=True)
        self.assertEqual(arith_decode(coded, len(d), freq), d)
        self.assertLess(len(coded), len(d))

    def test_mostly_zero_codes_to_fewer_bytes(self):
        d = [0] * 800 + [1, 2, 3] * 20
        freq = _adaptive_freq(d)
        coded = arith_encode(d, adaptive=True)
        self.assertEqual(arith_decode(coded, len(d), freq), d)
        self.assertLess(len(coded), len(d))

    def test_bits_per_symbol_well_under_8(self):
        d = [0] * 1000
        n_orig, n_coded, bps = size_for_byte_stream(d)
        self.assertEqual(n_orig, len(d))
        self.assertEqual(n_coded, len(arith_encode(d, adaptive=True)))
        self.assertLess(bps, 1.0)
        self.assertLess(subbyte_entropy(d), 0.01)

    def test_uniform_data_stays_near_8_bits(self):
        random.seed(7)
        d = [random.randrange(256) for _ in range(1000)]
        n_orig, n_coded, bps = size_for_byte_stream(d)
        self.assertGreater(bps, 7.0)
        self.assertLess(bps, 9.0)


class FrequencyTableCollapseTest(unittest.TestCase):
    def test_adaptive_total_capped_at_max(self):
        counts = {0: MAX_TOTAL + (1 << 30)}
        for b in range(1, ALPHABET):
            counts[b] = 1
        freq = SubByteFrequencyTable(counts)
        self.assertLessEqual(freq.total, MAX_TOTAL)

    def test_every_symbol_keeps_positive_width(self):
        counts = {0: MAX_TOTAL + (1 << 30)}
        for b in range(1, ALPHABET):
            counts[b] = 1
        freq = SubByteFrequencyTable(counts)
        for b in range(ALPHABET):
            lo, hi = freq.low_high(b)
            self.assertGreater(hi, lo)

    def test_pathological_total_round_trip(self):
        counts = {0: MAX_TOTAL + (1 << 30)}
        for b in range(1, ALPHABET):
            counts[b] = 1
        freq = SubByteFrequencyTable(counts)
        self.assertLessEqual(freq.total, MAX_TOTAL)
        random.seed(3)
        d = []
        for _ in range(60):
            d.append(0 if random.random() < 0.7 else random.randrange(200, 256))
        enc = Encoder(freq)
        for b in d:
            enc.encode_symbol(b)
        coded = enc.finish()
        dec = Decoder(freq, coded)
        got = [dec.decode_symbol() for _ in range(len(d))]
        self.assertEqual(got, d)


if __name__ == "__main__":
    unittest.main()
