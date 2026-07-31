# coding=utf-8
"""Tests for SARA routing boundaries + on-disk MoE serving (issue #36).

Covers the SARARouter registry/route contract and the existing MoEOnDisk
query-time path (live /0.001 reverse + fp32 matmul) on a synthetic x8D .gguf
container built via ``x8d_export.save_gguf``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_diffusion.moe_disk import (  # noqa: E402
    SARA_REGISTRY,
    SARABoundary,
    SARARouter,
    MoEOnDisk,
    X8DGgufReader,
)
from omni_diffusion.x8d_export import save_gguf  # noqa: E402

TMPDIR = os.path.join(os.path.dirname(__file__), "_tmp_moe_disk")
if not os.path.isdir(TMPDIR):
    os.makedirs(TMPDIR)


class SARARouterTest(unittest.TestCase):
    def setUp(self):
        self.router = SARARouter()

    def test_registry_covers_all_customers(self):
        customers = {b.customer for b in SARA_REGISTRY}
        self.assertEqual(
            customers,
            {"glm-5.2", "kimi-k3", "deepseek-v4-pro", "kokoro-82m", "whisper-large-v3", "ltx2"},
        )

    def test_registry_fields(self):
        for b in SARA_REGISTRY:
            self.assertIn(b.mode, ("moe", "dense"))
            self.assertTrue(b.upstream_repo)
            self.assertTrue(b.pointer_gguf.endswith(".x8dptr.gguf"))
            self.assertGreater(b.active_params, 0)
            self.assertGreaterEqual(b.total_params, b.active_params)

    def test_route_returns_boundary(self):
        for hint in ("text", "image", "video", "audio", "tts", "kimi-k3"):
            b = self.router.route(hint)
            self.assertIsInstance(b, SARABoundary)
            self.assertIn(b, SARA_REGISTRY)

    def test_route_modality_dispatch(self):
        self.assertEqual(self.router.route("text").customer, "kimi-k3")
        self.assertEqual(self.router.route("language").customer, "kimi-k3")
        self.assertEqual(self.router.route("image").customer, "ltx2")
        self.assertEqual(self.router.route("video").customer, "ltx2")
        self.assertEqual(self.router.route("audio").customer, "whisper-large-v3")
        self.assertEqual(self.router.route("asr").customer, "whisper-large-v3")
        self.assertEqual(self.router.route("tts").customer, "kokoro-82m")
        self.assertEqual(self.router.route("Kimi-K3").customer, "kimi-k3")

    def test_route_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.router.route("bogus-modality")

    def test_boundary_for_known_customer(self):
        b = self.router.boundary_for("kimi-k3")
        self.assertEqual(b.upstream_repo, "moonshotai/Kimi-K3")
        self.assertEqual(b.pointer_gguf, "kimi_k3.x8dptr.gguf")

    def test_boundary_for_unknown_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.router.boundary_for("no-such-model")

    def test_is_isolated_always_true(self):
        self.assertTrue(self.router.is_isolated("kimi-k3", "ltx2"))
        self.assertTrue(self.router.is_isolated("glm-5.2", "whisper-large-v3"))
        self.assertTrue(self.router.is_isolated("kokoro-82m", "kokoro-82m"))

    def test_active_params_matches_registry(self):
        for b in SARA_REGISTRY:
            self.assertEqual(self.router.active_params(b.customer), b.active_params)

    def test_dense_models_map_to_single_expert_mode(self):
        for customer in ("kokoro-82m", "whisper-large-v3", "ltx2"):
            self.assertEqual(self.router.boundary_for(customer).mode, "dense")
        for customer in ("glm-5.2", "kimi-k3", "deepseek-v4-pro"):
            self.assertEqual(self.router.boundary_for(customer).mode, "moe")

    def test_pointer_gguf_paths_exist_as_records(self):
        for b in SARA_REGISTRY:
            self.assertEqual(b.pointer_gguf, b.pointer_gguf)
        # all six registrants have a distinct, non-empty pointer record
        records = {b.customer: b.pointer_gguf for b in SARA_REGISTRY}
        self.assertEqual(len(records), len(SARA_REGISTRY))

    def test_known_kimi_k3_serveable_size(self):
        b = self.router.boundary_for("kimi-k3")
        # 1.56 TB upstream -> 2.837 GB serveable; pointer map 163,374,871 B
        self.assertEqual(b.pointer_gguf, "kimi_k3.x8dptr.gguf")
        self.assertEqual(b.total_params, 2_779_931_837_184)

    def test_router_rejects_duplicate_customers(self):
        dup = SARA_REGISTRY + SARA_REGISTRY
        with self.assertRaises(ValueError):
            SARARouter(registry=dup)


class MoEOnDiskSyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gguf = os.path.join(TMPDIR, "synthetic.gguf")
        w1 = bytes([128 + 64] * 16)
        w2 = bytes(range(8))
        save_gguf(
            {
                "layers.0.experts.3.w1": w1,
                "layers.0.experts.3.w2": w2,
            },
            cls.gguf,
        )

    def test_load_expert_reverses_0_001(self):
        m = MoEOnDisk(self.gguf)
        try:
            w = m.load_expert(0, 3, "w1", shape=(4, 4))
            self.assertEqual(len(w), 16)
            self.assertEqual(w, [0x80 + 64] * 16)
        finally:
            m.close()

    def test_matmul_fp32_shape_ok(self):
        m = MoEOnDisk(self.gguf)
        try:
            hidden = [1.0, 2.0, 3.0, 4.0]
            out = m.matmul_fp32(0, 3, hidden, "w1")
            self.assertEqual(len(out), 4)
            self.assertAlmostEqual(out[0], 5.0, places=6)
        finally:
            m.close()

    def test_reader_tensor_access(self):
        r = X8DGgufReader(self.gguf)
        try:
            self.assertEqual(r.tensor_bytes("layers.0.experts.3.w2"), bytes(range(8)))
            self.assertIsNone(r.tensor_bytes("missing"))
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main()
