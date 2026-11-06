import unittest
import numpy as np
from fastapi.testclient import TestClient
from app import app, resolve_demo_sample, get_dataset_manifest

client = TestClient(app)

class TestGalleryAndBenchmarks(unittest.TestCase):
    def setUp(self):
        self.manifest = get_dataset_manifest()

    def test_01_manifest_structure(self):
        """Verifies all 3 datasets exist with complete metadata."""
        self.assertIn("datasets", self.manifest)
        self.assertIn("nyu", self.manifest["datasets"])
        self.assertIn("kitti", self.manifest["datasets"])
        self.assertIn("middlebury", self.manifest["datasets"])

        for dkey in ["nyu", "kitti", "middlebury"]:
            dset = self.manifest["datasets"][dkey]
            self.assertIn("name", dset)
            self.assertIn("sensor", dset)
            self.assertIn("samples", dset)
            self.assertEqual(len(dset["samples"]), 3, f"Dataset {dkey} must have exactly 3 scenes.")

    def test_02_dual_thumbnails_present(self):
        """Verifies all 9 scenes have non-empty RGB and Depth Base64 thumbnails."""
        for dkey, dset in self.manifest["datasets"].items():
            for sample in dset["samples"]:
                sid = sample["id"]
                self.assertIn("rgb_thumb_b64", sample, f"Missing rgb_thumb_b64 in {sid}")
                self.assertIn("depth_thumb_b64", sample, f"Missing depth_thumb_b64 in {sid}")
                self.assertTrue(sample["rgb_thumb_b64"].startswith("data:image/"), f"Invalid rgb_thumb_b64 in {sid}")
                self.assertTrue(sample["depth_thumb_b64"].startswith("data:image/"), f"Invalid depth_thumb_b64 in {sid}")
                self.assertGreater(len(sample["rgb_thumb_b64"]), 100)
                self.assertGreater(len(sample["depth_thumb_b64"]), 100)

    def test_03_per_scene_benchmarks_authentic(self):
        """Verifies each scene contains 9 benchmark rows covering A0 through A5 with finite metrics."""
        for dkey, dset in self.manifest["datasets"].items():
            for sample in dset["samples"]:
                sid = sample["id"]
                self.assertIn("benchmark_rows", sample, f"Missing benchmark_rows in {sid}")
                rows = sample["benchmark_rows"]
                self.assertEqual(len(rows), 9, f"Scene {sid} must have exactly 9 evaluated algorithms.")
                
                methods = [r["method"] for r in rows]
                self.assertTrue(any("A0: Nearest" in m for m in methods))
                self.assertTrue(any("A0: Bilinear" in m for m in methods))
                self.assertTrue(any("A1: Guided" in m or "A1: Joint" in m for m in methods))
                self.assertTrue(any("A2: Deep Learning" in m or "EdgeFuseNet Alone" in m for m in methods))
                self.assertTrue(any("A5: EdgeFuse-DCR Full Hybrid" in m for m in methods))

                for r in rows:
                    rmse = float(r["rmse"])
                    psnr = float(r["psnr"])
                    ssim = float(r["ssim"])
                    self.assertGreater(rmse, 0.0)
                    self.assertTrue(np.isfinite(rmse))
                    self.assertTrue(np.isfinite(psnr))
                    self.assertTrue(0.0 <= ssim <= 1.0)

    def test_04_api_gallery_endpoint(self):
        """Tests GET /api/dataset/gallery returns full manifest with thumbnails."""
        res = client.get("/api/dataset/gallery")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("datasets", data)
        self.assertEqual(len(data["datasets"]["kitti"]["samples"]), 3)

    def test_05_api_scene_benchmark_endpoint(self):
        """Tests GET /api/benchmark/scene/{sample_id} for all 9 scenes."""
        sample_ids = [
            "nyu_living_room", "nyu_bedroom", "nyu_study_office",
            "kitti_road_01", "kitti_residential_02", "kitti_drive_03",
            "middlebury_artroom1", "middlebury_chess1", "middlebury_ladder1"
        ]
        for sid in sample_ids:
            res = client.get(f"/api/benchmark/scene/{sid}")
            self.assertEqual(res.status_code, 200, f"Failed for {sid}")
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["sample_id"], sid)
            self.assertEqual(len(data["benchmark_rows"]), 9)

    def test_06_api_demo_loading_all_scenes(self):
        """Tests GET /api/demo/{sample_id} delivers RGB, LR Depth, Validity Mask, and benchmarks."""
        sample_ids = [
            "nyu_living_room", "kitti_drive_03", "middlebury_ladder1"
        ]
        for sid in sample_ids:
            res = client.get(f"/api/demo/{sid}")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["success"])
            self.assertTrue(data["rgb_b64"].startswith("data:image/"))
            self.assertTrue(data["lr_depth_b64"].startswith("data:image/"))
            self.assertTrue(data["valid_mask_b64"].startswith("data:image/"))
            self.assertIn("valid_mask_stats", data)
            self.assertGreater(data["valid_mask_stats"]["total_pixels"], 0)
            self.assertEqual(len(data["benchmark_rows"]), 9)

if __name__ == "__main__":
    unittest.main()

