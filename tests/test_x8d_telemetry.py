# coding=utf-8
"""Tests for the per-8x8-block I/O + memory telemetry layer (issue #41).

Pure stdlib unittest. Verifies the Colibrì ``telemetry.h``-style counters:
I/O byte accounting, RSS, per-block timing, expert-hit tier split, snapshot
serialization and the dashboard stats line.
"""

import unittest

from omni_diffusion.x8d_telemetry import BLOCK_SIZE, Telemetry, _selftest


class TelemetryIOTest(unittest.TestCase):
    def test_record_io(self):
        t = Telemetry("io")
        t.record_io(4096)
        t.record_io(1024)
        self.assertEqual(t.io_bytes, 5120)
        self.assertEqual(t.fault_bytes, 0)

    def test_record_fault(self):
        t = Telemetry("fault")
        t.record_fault(2048)
        self.assertEqual(t.fault_bytes, 2048)
        self.assertEqual(t.io_bytes, 0)

    def test_ignore_nonpositive(self):
        t = Telemetry("z")
        t.record_io(0)
        t.record_io(-5)
        self.assertEqual(t.io_bytes, 0)


class TelemetryHitsTest(unittest.TestCase):
    def test_hit_split(self):
        t = Telemetry("hit")
        t.record_hit("pin")
        t.record_hit("pin")
        t.record_hit("lru")
        snap = t.snapshot()
        self.assertEqual(snap["hits_pin"], 2)
        self.assertEqual(snap["hits_lru"], 1)

    def test_unknown_tier_raises(self):
        t = Telemetry("bad")
        with self.assertRaises(ValueError):
            t.record_hit("nope")


class TelemetryBlocksTest(unittest.TestCase):
    def test_block_timing(self):
        t = Telemetry("blk")
        t.begin_block()
        elapsed = t.end_block()
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(t.block_count, 1)
        self.assertEqual(t.blocks_count, 1)
        snap = t.snapshot()
        self.assertEqual(snap["blocks"], 1)
        self.assertEqual(snap["block_us_mean"], elapsed)
        self.assertEqual(snap["block_us_max"], elapsed)

    def test_multiple_blocks(self):
        t = Telemetry("multi")
        for _ in range(3):
            t.begin_block()
            t.end_block()
        snap = t.snapshot()
        self.assertEqual(snap["blocks"], 3)
        self.assertGreaterEqual(snap["block_us_mean"], 0)
        self.assertGreaterEqual(snap["block_us_max"], snap["block_us_mean"])


class TelemetryMemoryTest(unittest.TestCase):
    def test_rss_positive(self):
        t = Telemetry("mem")
        self.assertGreater(t.rss_bytes, 0)
        self.assertGreater(t.snapshot()["rss_mb"], 0)

    def test_snapshot_shape(self):
        t = Telemetry("shape")
        t.record_io(64)
        t.begin_block()
        t.end_block()
        snap = t.snapshot()
        for key in (
            "label",
            "io_bytes",
            "fault_bytes",
            "blocks",
            "block_us_mean",
            "block_us_max",
            "hits_pin",
            "hits_lru",
            "rss_mb",
            "elapsed_s",
        ):
            self.assertIn(key, snap)
        self.assertEqual(snap["label"], "shape")


class TelemetryDashboardTest(unittest.TestCase):
    def test_dashboard_line(self):
        t = Telemetry("dash")
        t.record_io(BLOCK_SIZE)
        t.record_fault(32)
        t.begin_block()
        t.end_block()
        line = t.dashboard()
        self.assertIn("[dash]", line)
        self.assertIn("blk=1", line)
        self.assertIn("io=", line)
        self.assertIn("rss=", line)
        self.assertIn("hit_pin=", line)

    def test_close_idempotent(self):
        t = Telemetry("c")
        t.close()
        t.close()


class TelemetrySelftestTest(unittest.TestCase):
    def test_selftest_passes(self):
        self.assertEqual(_selftest(), 0)


if __name__ == "__main__":
    unittest.main()
