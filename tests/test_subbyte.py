# coding=utf-8
"""Tests for the packed 0.016 bit/weight sub-byte model (32 MB = full 32 GB)."""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_subbyte import (  # noqa: E402
    BITS_PER_WEIGHT,
    SUB_BYTE_MAGIC,
    WEIGHTS_PER_COORD,
    SubByteModel,
    SubByteHeaderError,
    coords_per_pack,
    load_subbyte_gguf,
    pack_subbyte,
    packed_size_bytes,
    print_size_report_subbyte,
    quanta_of,
    save_subbyte_gguf,
    size_report_subbyte,
    unpack_subbyte,
    weight_of,
)


class SubBytePackingTest(unittest.TestCase):
    def test_size_16b_is_32mb(self):
        self.assertEqual(packed_size_bytes(16_000_000_000), 32_000_000)
        self.assertEqual(BITS_PER_WEIGHT, 0.016)
        self.assertEqual(WEIGHTS_PER_COORD, 500)

    def test_size_report_equivalence(self):
        r = size_report_subbyte(16_000_000_000, baseline_bits=16)
        self.assertAlmostEqual(r["full_precision_gb"], 32.0)
        self.assertAlmostEqual(r["subbyte_mb"], 32.0)
        self.assertAlmostEqual(r["reduction_pct"], 99.9, places=6)
        self.assertAlmostEqual(r["bits_per_weight"], 0.016)

    def test_quanta_pointer_map_roundtrip(self):
        for w in range(256):
            self.assertEqual(weight_of(quanta_of(w)), w)

    def test_pack_unpack_small(self):
        data = bytes(range(256)) * 2  # 512 bytes = 1.024 coords -> 2 coords
        packed = pack_subbyte(data)
        self.assertEqual(len(packed), coords_per_pack(len(data)))
        restored = unpack_subbyte(packed, len(data))
        self.assertEqual(len(restored), len(data))

    def test_save_load_roundtrip(self):
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_subbyte.gguf")
        data = os.urandom(1024)
        try:
            path, packed_len = save_subbyte_gguf("model.weight", data, tmp)
            payloads, meta = load_subbyte_gguf(path)
            self.assertEqual(meta["num_params"], len(data))
            self.assertEqual(len(payloads["model.weight"]), packed_len)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_mmap_serves_weights(self):
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_subbyte2.gguf")
        data = bytes((i * 3) & 0xFF for i in range(2000))
        try:
            path, _ = save_subbyte_gguf("model.weight", data, tmp)
            model = SubByteModel(path)
            self.assertEqual(len(model), len(data))
            self.assertTrue(model.packed_size_mb() < 1.0)
            # the running state is the sub-byte coordinate pointer map:
            # served weights reconstruct exactly from the packed coordinates
            expected = unpack_subbyte(pack_subbyte(data), len(data))
            self.assertEqual(model.weights(), expected)
            model.close()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_bad_magic_raises(self):
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_bad.gguf")
        with open(tmp, "wb") as f:
            f.write(b"NOTSUB" + os.urandom(32))
        try:
            with self.assertRaises(SubByteHeaderError):
                load_subbyte_gguf(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_print_report(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_size_report_subbyte()
        out = buf.getvalue()
        self.assertIn("16,000,000,000", out)
        self.assertIn("32.00 GB", out)
        self.assertIn("32.0 MB", out)
        self.assertIn("0.016 bit/weight", out)


if __name__ == "__main__":
    unittest.main()
