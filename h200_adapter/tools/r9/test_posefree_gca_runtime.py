"""Run with the preparation Python; CUDA comparisons also run in the job."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from posefree_gca_runtime import ByteLRU, atomic_json, valid_scale, compare_stats, rank_images, summarize

try:
    import torch
except ImportError:
    torch = None


class StorageTests(unittest.TestCase):
    def test_lru_evicts_by_recent_use_and_obeys_byte_limit(self):
        cache = ByteLRU(10)
        cache.put("a", "A", 4)
        cache.put("b", "B", 4)
        self.assertEqual(cache.get("a"), "A")
        cache.put("c", "C", 4)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "A")
        self.assertEqual(cache.used, 8)
        cache.put("d", "D", 11)
        self.assertIsNone(cache.get("d"))
        cache.put("a", "A2", 2)
        self.assertEqual(cache.used, 6)

    def test_zero_budget_disables_cache(self):
        cache = ByteLRU(0)
        cache.put("a", "A", 1)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.used, 0)

    def test_atomic_report_preserves_old_file_on_invalid_payload(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "scale.json"
            atomic_json(path, {"global_scale": 1.0})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                atomic_json(path, {"global_scale": float("nan")})
            self.assertEqual(path.read_bytes(), before)
            atomic_json(path, {"global_scale": 2.0})
            self.assertEqual(json.loads(path.read_text())["global_scale"], 2.0)
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_resume_rejects_broken_or_nonpositive_scales(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "scale.json"
            self.assertFalse(valid_scale(path))
            report = {"global_scale": 2, "num_ratio_pixels": 120,
                      "method": "gca_metric_scale_adapted_to_vggt_omega"}
            atomic_json(path, report)
            self.assertTrue(valid_scale(path))
            report["global_scale"] = -1
            atomic_json(path, report)
            self.assertFalse(valid_scale(path))
            path.write_text('{"global_scale":')
            self.assertFalse(valid_scale(path))

    def test_comparison_detects_geometry_and_mask_differences(self):
        base = {"global_scale": 1.0, "global_log_mad": 0.1, "num_ratio_pixels": 128,
                "selected": [{"index": 1, "frame": 3, "camera": "0",
                              "valid_pixels": 128, "median_scale": 1.0}]}
        compare_stats(base, copy.deepcopy(base))
        other = copy.deepcopy(base)
        other["global_scale"] = 1.01
        with self.assertRaises(RuntimeError):
            compare_stats(base, other)
        other = copy.deepcopy(base)
        other["selected"][0]["valid_pixels"] -= 1
        with self.assertRaises(RuntimeError):
            compare_stats(base, other)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NumericTests(unittest.TestCase):
    def check_device(self, device):
        # Distinguishes percentile interpolation from a lower median; checks
        # tie order, lower-median scale and handling of non-finite depths.
        conf_np = np.tile(np.arange(256, dtype=np.float32).reshape(1, 16, 16), (5, 1, 1))
        depth_np = np.ones_like(conf_np)
        depth_np[4, 10, 10] = np.nan
        conf, depth = torch.from_numpy(conf_np).to(device), torch.from_numpy(depth_np).to(device)
        selected, thresholds, _ = rank_images(conf, 3, 0.5, torch)
        self.assertEqual(selected, [4, 3, 2])
        self.assertTrue(all(float(x.item()) == 127.5 for x in thresholds))
        entries = [{"frame_id": i, "camera_id": "0"} for i in range(5)]
        metric, ratios = {}, []
        for idx in selected:
            m = (np.arange(256, dtype=np.float32).reshape(16, 16) + 1) / 100 + idx
            metric[idx] = (torch.from_numpy(m), torch.ones((16, 16), dtype=torch.bool))
            valid = (conf_np[idx] > 127.5) & np.isfinite(depth_np[idx])
            ratios.append(m[valid] / depth_np[idx][valid])
        report = summarize(depth, conf, selected, thresholds, metric, entries, torch)
        ratios = np.concatenate(ratios)
        lower_median = float(np.sort(ratios)[(len(ratios) - 1) // 2])
        self.assertAlmostEqual(report["global_scale"], lower_median, places=5)
        self.assertEqual(report["num_ratio_pixels"], len(ratios))
        self.assertEqual(report["selected"][0]["valid_pixels"], 127)

    def test_cpu_numerics(self):
        self.check_device("cpu")

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_numerics(self):
        self.check_device("cuda")


if __name__ == "__main__":
    unittest.main()
