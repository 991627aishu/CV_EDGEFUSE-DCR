"""
test_end_to_end_verification.py
Comprehensive End-to-End Test Suite for EdgeFuse-DCR.
Verifies all 12 test scenarios from Phase 9 of the master prompt.
"""

import os
import io
import json
import pytest
import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient

from app import app, AI_DEPTH_MGR

client = TestClient(app)

def create_synthetic_rgb_image(width=160, height=120):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (200, 100, 50), -1)
    cv2.circle(img, (width // 2, height // 2), 25, (50, 220, 120), -1)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()

def create_synthetic_depth_npy(width=40, height=30, min_d=1.0, max_d=5.0):
    y, x = np.mgrid[0:height, 0:width]
    depth = min_d + (max_d - min_d) * (y / float(height))
    buf = io.BytesIO()
    np.save(buf, depth.astype(np.float32))
    return buf.getvalue()

def create_synthetic_depth_png(width=40, height=30, min_d=1.0, max_d=5.0):
    y, x = np.mgrid[0:height, 0:width]
    depth_m = min_d + (max_d - min_d) * (y / float(height))
    depth_mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    success, encoded = cv2.imencode(".png", depth_mm)
    return encoded.tobytes()

class TestEndToEndVerification:

    def test_01_dataset_demo_reconstruction(self):
        """Test 1: Dataset Demo -> Actual dataset reconstruction"""
        res = client.post("/api/reconstruct", data={
            "demo_sample_id": "nyu_living_room",
            "method": "hybrid",
            "upsample_factor": 4,
            "apply_texture_suppression": True,
            "suppression_strength": 0.85,
            "colormap": "turbo",
            "input_source": "dataset"
        })
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["has_gt"] is True
        assert "metrics" in data
        assert "rmse" in data["metrics"]
        assert float(data["metrics"]["rmse"]) > 0.0
        assert "reconstructed_b64" in data
        assert len(data["reconstructed_b64"]) > 50

    def test_02_custom_rgb_4x_depth(self):
        """Test 2: Custom RGB + 4× depth -> Successful reconstruction"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        depth_bytes = create_synthetic_depth_npy(40, 30)
        
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png"),
            "depth_file": ("test_depth.npy", depth_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct", data={
            "method": "hybrid",
            "upsample_factor": 4,
            "apply_texture_suppression": True,
            "suppression_strength": 0.85,
            "colormap": "turbo",
            "input_source": "sensor"
        }, files=files)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["is_custom"] is True
        assert "reconstructed_b64" in data

    def test_03_custom_rgb_2x_depth(self):
        """Test 3: Custom RGB + 2× depth -> Successful reconstruction"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        depth_bytes = create_synthetic_depth_npy(80, 60)
        
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png"),
            "depth_file": ("test_depth.npy", depth_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct", data={
            "method": "hybrid",
            "upsample_factor": 2,
            "input_source": "sensor"
        }, files=files)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True

    def test_04_custom_rgb_8x_depth(self):
        """Test 4: Custom RGB + 8× depth -> Successful reconstruction"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        depth_bytes = create_synthetic_depth_npy(20, 15)
        
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png"),
            "depth_file": ("test_depth.npy", depth_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct", data={
            "method": "hybrid",
            "upsample_factor": 8,
            "input_source": "sensor"
        }, files=files)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True

    def test_05_custom_rgb_with_physical_gt(self):
        """Test 5: Custom RGB + physical GT -> Reconstruction + valid metrics"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        depth_bytes = create_synthetic_depth_npy(40, 30)
        gt_bytes = create_synthetic_depth_npy(160, 120)
        
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png"),
            "depth_file": ("test_depth.npy", depth_bytes, "application/octet-stream"),
            "gt_file": ("test_gt.npy", gt_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct", data={
            "method": "hybrid",
            "upsample_factor": 4,
            "input_source": "sensor"
        }, files=files)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["has_gt"] is True
        assert data["gt_type"] == "sensor"
        assert "metrics" in data
        assert "rmse" in data["metrics"]
        assert float(data["metrics"]["rmse"]) >= 0.0

    def test_06_custom_rgb_no_gt(self):
        """Test 6: Custom RGB + no GT -> Reconstruction + sensor fidelity metrics without error"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        depth_bytes = create_synthetic_depth_npy(40, 30)
        
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png"),
            "depth_file": ("test_depth.npy", depth_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct", data={
            "method": "hybrid",
            "upsample_factor": 4,
            "input_source": "sensor"
        }, files=files)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["has_gt"] is False
        assert "metrics" in data

    def test_07_generate_pseudo_gt(self):
        """Test 7: Generate Pseudo-GT -> Valid labelled estimated reference"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        depth_bytes = create_synthetic_depth_npy(40, 30)
        
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png"),
            "depth_file": ("test_depth.npy", depth_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct/synthesize_gt", data={
            "session_id": "test_pseudo_session_001",
            "depth_scale": "auto"
        }, files=files)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["is_pseudo_gt"] is True
        assert data["gt_type"] == "pseudo_gt"
        assert "gt_preview_b64" in data
        assert "gt_meta" in data
        assert "calib_mode" in data["gt_meta"]

    def test_08_ai_predicted_depth(self):
        """Test 8: AI-Predicted Depth -> Model inference or valid fallback"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png")
        }
        res = client.post("/api/ai_depth/predict", data={
            "input_size": 392
        }, files=files)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert "prediction_id" in data
        assert "preview_b64" in data
        assert "metadata" in data

    def test_09_invalid_depth_file(self):
        """Test 9: Invalid depth file -> Clear validation error (400)"""
        rgb_bytes = create_synthetic_rgb_image(160, 120)
        corrupted_bytes = b"CORRUPTED_NOT_A_VALID_DEPTH_ARRAY_12345"
        
        files = {
            "rgb_file": ("test_rgb.png", rgb_bytes, "image/png"),
            "depth_file": ("test_depth.npy", corrupted_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct", data={
            "method": "hybrid",
            "upsample_factor": 4,
            "input_source": "sensor"
        }, files=files)
        assert res.status_code == 400

    def test_10_unsupported_image(self):
        """Test 10: Unsupported image -> Clear validation error (400)"""
        corrupted_rgb = b"NOT_AN_IMAGE_CONTENT"
        depth_bytes = create_synthetic_depth_npy(40, 30)
        
        files = {
            "rgb_file": ("test_rgb.txt", corrupted_rgb, "text/plain"),
            "depth_file": ("test_depth.npy", depth_bytes, "application/octet-stream")
        }
        res = client.post("/api/reconstruct", data={
            "method": "hybrid",
            "upsample_factor": 4,
            "input_source": "sensor"
        }, files=files)
        assert res.status_code == 400

    def test_11_repeated_reconstruction(self):
        """Test 11: Repeated reconstruction -> Correct independent results"""
        res1 = client.post("/api/reconstruct", data={
            "demo_sample_id": "nyu_living_room",
            "method": "hybrid",
            "upsample_factor": 4,
            "input_source": "dataset"
        })
        assert res1.status_code == 200
        run_id_1 = res1.json().get("run_id")

        res2 = client.post("/api/reconstruct", data={
            "demo_sample_id": "nyu_living_room",
            "method": "nearest",
            "upsample_factor": 4,
            "input_source": "dataset"
        })
        assert res2.status_code == 200
        run_id_2 = res2.json().get("run_id")

        assert run_id_1 != run_id_2
        assert res1.json()["method"] == "hybrid"
        assert res2.json()["method"] == "nearest"

    def test_12_switch_between_datasets(self):
        """Test 12: Switch between datasets -> No stale results"""
        res_nyu = client.get("/api/demo/nyu_living_room")
        assert res_nyu.status_code == 200
        data_nyu = res_nyu.json()
        assert data_nyu["dataset_key"] == "nyu"

        res_kitti = client.get("/api/demo/kitti_road_01")
        assert res_kitti.status_code == 200
        data_kitti = res_kitti.json()
        assert data_kitti["dataset_key"] == "kitti"
        assert data_nyu["title"] != data_kitti["title"]
