# coding=utf-8
"""Tests for the packed 0.008 bit/weight sub-byte model (1000:1, per-byte law)."""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.x8d_subbyte import (  # noqa: E402
    BITS_PER_WEIGHT,
    WEIGHTS_PER_COORD,
    SubByteModel,
    coords_per_pack,
    load_subbyte_gguf,
    mmap_load_subbyte_gguf,
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
    def test_size_16b_is_16mb(self):
        self.assertEqual(packed_size_bytes(16_000_000_000), 16_000_000)
        self.assertEqual(BITS_PER_WEIGHT, 0.008)
        self.assertEqual(WEIGHTS_PER_COORD, 1000)

    def test_size_report_equivalence(self):
        r = size_report_subbyte(16_000_000_000, baseline_bits=16)
        self.assertAlmostEqual(r["full_precision_gb"], 32.0)
        self.assertAlmostEqual(r["subbyte_mb"], 16.0)
        self.assertAlmostEqual(r["reduction_pct"], 99.95, places=6)
        self.assertAlmostEqual(r["bits_per_weight"], 0.008)

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

    def test_mmap_packed_size_excludes_name_length(self):
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_subbyte_meta.gguf")
        name = "a_very_long_tensor_name_for_testing"
        data = os.urandom(1024)
        try:
            path, _ = save_subbyte_gguf(name, data, tmp)
            mapping, meta = mmap_load_subbyte_gguf(path)
            try:
                # magic-free layout: <u64 num_params><u32 name_len><name><packed>
                self.assertEqual(meta["packed_size"], os.path.getsize(path) - 12 - len(name))
            finally:
                mapping.close()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_mmap_bounds_checks(self):
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_subbyte_bounds.gguf")
        data = bytes([200]) * 1000
        try:
            path, _ = save_subbyte_gguf("w", data, tmp)
            model = SubByteModel(path)
            for fn, args in [("weight_at", (1000,)), ("weight_at", (-1,)),
                             ("weights", (0, 5000)), ("weights", (5, 3))]:
                with self.assertRaises(IndexError, msg=f"{fn}{args}"):
                    getattr(model, fn)(*args)
            self.assertEqual(model.weights(0, 0), [])
            model.close()
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

    def test_weights_slices_match_weight_at(self):
        """Bulk C-speed slice must equal per-index weight_at on every edge."""
        import random

        tmp = os.path.join(os.path.dirname(__file__), "_tmp_subbyte3.gguf")
        random.seed(3)
        data = bytes(random.getrandbits(8) for _ in range(1500))  # 3 coords
        try:
            path, _ = save_subbyte_gguf("model.weight", data, tmp)
            model = SubByteModel(path)
            ref = [model.weight_at(i) for i in range(len(data))]
            self.assertEqual(model.weights(), ref)
            for s, e in [
                (0, 1),
                (1, 2),
                (499, 500),
                (500, 501),
                (123, 987),
                (1200, 1500),
                (1499, 1500),
                (0, 500),
                (0, 1500),
            ]:
                self.assertEqual(model.weights(s, e), ref[s:e], f"slice {s}:{e}")
            model.close()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_container_is_magic_free(self):
        # The container is raw packed coordinates with no X8DSUB01 magic.
        tmp = os.path.join(os.path.dirname(__file__), "_tmp_magicfree.gguf")
        data = os.urandom(1024)
        try:
            path, _ = save_subbyte_gguf("model.weight", data, tmp)
            with open(path, "rb") as f:
                blob = f.read()
            self.assertNotIn(b"X8DSUB01", blob)
            payloads, meta = load_subbyte_gguf(path)
            self.assertEqual(set(payloads), {"model.weight"})
            self.assertEqual(meta["num_params"], len(data))
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
        self.assertIn("16.0 MB", out)
        self.assertIn("0.008 bit/weight", out)


if __name__ == "__main__":
    unittest.main()
