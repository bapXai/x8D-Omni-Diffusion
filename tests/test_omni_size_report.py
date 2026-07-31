# coding=utf-8
"""Tests for the omni-stack size report (issue #38). Pure stdlib, offline."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from omni_size_report import (  # noqa: E402
    BIT_PER_PARAM_PTR_BF16,
    BIT_PER_PARAM_U8,
    KIMI_K3_POINTER_BYTES,
    KIMI_K3_TOTAL_PARAMS,
    OmniSizeReport,
)

MODELS = {
    "glm_5_2",
    "kimi_k3",
    "deepseek_v4_pro",
    "diffusiongemma",
    "kokoro",
    "whisper",
    "ltx2",
    "kitten_tts",
}


class OmniSizeReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = OmniSizeReport()

    def test_registry_contains_all_8_models(self):
        self.assertEqual(set(self.report.models), MODELS)

    def test_bit_per_param_values(self):
        self.assertEqual(self.report.bit_per_param_u8(), 0.008)
        self.assertEqual(self.report.bit_per_param_ptr_bf16(), 0.016)
        self.assertEqual(BIT_PER_PARAM_U8, 0.008)
        self.assertEqual(BIT_PER_PARAM_PTR_BF16, 0.016)

    def test_kimi_k3_pointer_bytes_match_readme(self):
        self.assertEqual(self.report.models["kimi_k3"]["pointer_bytes"], 163_374_871)
        self.assertEqual(KIMI_K3_POINTER_BYTES, 163_374_871)
        self.assertEqual(self.report.models["kimi_k3"]["upstream_bytes"], 1_560_860_324_864)
        self.assertGreater(self.report.models["kimi_k3"]["total_params"], 2.7e12)

    def test_kimi_k3_serveable_matches_research(self):
        # research/Status-and-Optimization-Audit-2026.md: 2.723 GB U8 +
        # 114.4 MB BF16-pointer = 2.837 GB serveable.
        u8 = self.report.size_u8("kimi_k3")
        bf16 = self.report.size_ptr_bf16("kimi_k3")
        self.assertAlmostEqual(u8 / 1e9, 2.723, delta=0.005)
        self.assertAlmostEqual(bf16 / 1e6, 114.4, delta=0.5)
        self.assertAlmostEqual((u8 + bf16) / 1e9, 2.837, delta=0.005)

    def test_compression_ratio_gt_one(self):
        self.assertGreater(self.report.compression_ratio(), 1.0)
        self.assertGreater(self.report.serveable_compression_ratio(), 1.0)

    def test_active_params_tts_maps_kokoro_82m(self):
        act = self.report.active_params_for("tts")
        self.assertIn("kokoro", act)
        self.assertEqual(act["kokoro"], 82_000_000)
        self.assertEqual(self.report.models["kokoro"]["active_params"], 82_000_000)

    def test_active_params_asr_text_video(self):
        self.assertEqual(self.report.active_params_for("asr"), {"whisper": 1_550_000_000})
        text = self.report.active_params_for("text")
        self.assertEqual(text["glm_5_2"], 39_000_000_000)
        self.assertEqual(text["kimi_k3"], 104_200_000_000)
        self.assertEqual(text["deepseek_v4_pro"], 49_000_000_000)
        self.assertEqual(text["diffusiongemma"], 4_000_000_000)
        self.assertEqual(self.report.active_params_for("video"), {"ltx2": 19_000_000_000})
        self.assertEqual(self.report.active_params_for("image"), {"ltx2": 19_000_000_000})
        self.assertEqual(self.report.active_params_for("unknown"), {})

    def test_combined_total_params_gt_3e12(self):
        self.assertGreater(self.report.combined_total_params(), 3e12)
        self.assertGreater(self.report.combined_total_params(), KIMI_K3_TOTAL_PARAMS)

    def test_combined_sizes_and_ratios(self):
        upstream = self.report.combined_upstream_bytes()
        pointer = self.report.combined_pointer_bytes()
        serveable = self.report.combined_serveable_bytes()
        u8 = self.report.combined_u8_bytes()
        self.assertGreater(upstream, 3.5e12)  # ~3.98 TB of upstream models
        self.assertLess(pointer, 1e9)         # pointer maps < 1 GB
        self.assertLess(serveable, 6e9)       # serveable < 6 GB
        self.assertLess(u8, serveable)
        self.assertGreater(self.report.compression_ratio(), 1000.0)

    def test_dense_models_activate_all_params(self):
        for name in ("kokoro", "whisper", "ltx2", "kitten_tts"):
            rec = self.report.models[name]
            self.assertEqual(rec["total_params"], rec["active_params"])


if __name__ == "__main__":
    unittest.main()
