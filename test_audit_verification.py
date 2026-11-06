"""
EDGEFUSE-DCR: Comprehensive Scientific Audit Verification Test Suite
Validates:
1. Optional Ground Truth 3rd upload slot & Error Heatmap generation.
2. Scientific depth unit labeling (relative 'rel' vs metric 'm').
3. Mathematical algorithm differentiation (A0 vs A2 vs A5 pairwise MAE).
4. Full tab visualization payloads (Primary, Ablation, Boundary, Confidence/Error).
5. Regression test for all built-in research demos (NYU, KITTI, Middlebury).
"""

import sys
import os
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app

client = TestClient(app)

def create_rgb_bytes(w=640, h=480, color=(140, 180, 210)):
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def create_16bit_depth_bytes(w=640, h=480, min_val=1500, max_val=4500):
    depth = np.linspace(min_val, max_val, h * w).reshape((h, w)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(depth).save(buf, format="PNG")
    return buf.getvalue()

def create_8bit_depth_bytes(w=640, h=480):
    depth = np.linspace(20, 240, h * w).reshape((h, w)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(depth, mode="L").save(buf, format="PNG")
    return buf.getvalue()

def run_scientific_audit():
    print("=" * 75)
    print("EDGEFUSE-DCR: SCIENTIFIC AUDIT & DIFFERENTIATION VERIFICATION")
    print("=" * 75)
    
    passed = 0
    total = 7

    # 1. Custom Upload WITHOUT Ground Truth
    print("\n[Audit 1] Custom Upload WITHOUT Ground Truth (Empty GT handling)...")
    rgb_b = create_rgb_bytes(640, 480)
    d_b = create_16bit_depth_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "upsample_factor": "4", "depth_scale": "mm_to_m"},
                      files={"rgb_file": ("rgb.png", rgb_b, "image/png"), "depth_file": ("depth.png", d_b, "image/png")})
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    data = res.json()
    assert data["has_gt"] is False, "has_gt must be False when GT is omitted"
    assert data["metrics"] is None, "Metrics must be None when no GT is provided"
    assert data["error_b64"] is None, "Error map must be None when no GT is provided"
    assert data["reconstructed_b64"] is not None
    assert data["confidence_b64"] is not None
    assert data["boundary_b64"] is not None
    assert data["classical_b64"] is not None
    assert data["cnn_b64"] is not None
    print(f" -> PASS: Confirmed has_gt=False, metrics=None, error_b64=None, all 4 tabs populated.")
    passed += 1

    # 2. Custom Upload WITH Ground Truth
    print("\n[Audit 2] Custom Upload WITH Ground Truth (3rd slot evaluation)...")
    gt_b = create_16bit_depth_bytes(640, 480, min_val=1400, max_val=4600)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "upsample_factor": "4", "depth_scale": "mm_to_m"},
                      files={"rgb_file": ("rgb.png", rgb_b, "image/png"), 
                             "depth_file": ("depth.png", d_b, "image/png"),
                             "gt_file": ("gt.png", gt_b, "image/png")})
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    data = res.json()
    assert data["has_gt"] is True, "has_gt must be True when GT is uploaded"
    assert data["metrics"] is not None, "Metrics must be computed when GT is uploaded"
    assert "rmse" in data["metrics"] and "mae" in data["metrics"] and "psnr" in data["metrics"]
    assert data["error_b64"] is not None, "Error heatmap must be rendered when GT is uploaded"
    print(f" -> PASS: Confirmed has_gt=True, RMSE: {data['metrics']['rmse']}m, MAE: {data['metrics']['mae']}m, Error heatmap generated.")
    passed += 1

    # 3. Algorithm Differentiation (A0 vs A2 vs A5)
    print("\n[Audit 3] Algorithm Differentiation Analysis (Pairwise MAE differences)...")
    algo_comp = data.get("algorithm_comparison")
    assert algo_comp is not None, "algorithm_comparison dict must be returned"
    mae_0_5 = algo_comp["diff_a0_a5_mae"]
    mae_2_5 = algo_comp["diff_a2_a5_mae"]
    mae_0_2 = algo_comp["diff_a0_a2_mae"]
    assert mae_0_5 > 0.0, f"A0 and A5 must differ, got {mae_0_5}"
    assert mae_2_5 > 0.0, f"A2 and A5 must differ, got {mae_2_5}"
    print(f" -> PASS: Verified algorithm divergence: |A0 - A5| = {mae_0_5}m, |A2 - A5| = {mae_2_5}m, |A0 - A2| = {mae_0_2}m")
    passed += 1

    # 4. Scientific Scale Detection: Relative Depth vs Metric Depth
    print("\n[Audit 4] Scientific Scale: Relative 8-bit depth verification...")
    d8_b = create_8bit_depth_bytes(640, 480)
    res = client.post("/api/preview_depth", files={"depth_file": ("rel_depth.png", d8_b, "image/png")})
    assert res.status_code == 200
    pdata = res.json()
    assert pdata["depth_type"] == "relative", f"Expected relative, got {pdata['depth_type']}"
    assert pdata["depth_unit"] == "rel", f"Expected rel unit, got {pdata['depth_unit']}"
    assert pdata["max_depth"] <= 1.0, f"Relative max must be <= 1.0, got {pdata['max_depth']}"
    assert "valid_mask_b64" in pdata and len(pdata["valid_mask_b64"]) > 100
    print(f" -> PASS: Verified scientific honesty: 8-bit depth correctly labeled as relative ({pdata['depth_unit']}), range [0.0 - 1.0].")
    passed += 1

    # 5. Built-in Demo: NYU Depth V2
    print("\n[Audit 5] Built-in Demo: NYU Depth V2 Indoor Scene...")
    res = client.post("/api/reconstruct", data={"demo_sample_id": "nyu", "method": "hybrid", "upsample_factor": "4"})
    assert res.status_code == 200, f"NYU failed: {res.text}"
    nyu_data = res.json()
    assert nyu_data["has_gt"] is True
    assert nyu_data["metrics"]["rmse"] > 0
    print(f" -> PASS: NYU reconstructed in {nyu_data['latency_ms']}ms. RMSE: {nyu_data['metrics']['rmse']}m, PSNR: {nyu_data['metrics']['psnr']}dB")
    passed += 1

    # 6. Built-in Demo: KITTI LiDAR
    print("\n[Audit 6] Built-in Demo: KITTI LiDAR Outdoor Scene...")
    res = client.post("/api/reconstruct", data={"demo_sample_id": "kitti", "method": "hybrid", "upsample_factor": "4"})
    assert res.status_code == 200, f"KITTI failed: {res.text}"
    kitti_data = res.json()
    assert kitti_data["has_gt"] is True
    print(f" -> PASS: KITTI reconstructed in {kitti_data['latency_ms']}ms. RMSE: {kitti_data['metrics']['rmse']}m, PSNR: {kitti_data['metrics']['psnr']}dB")
    passed += 1

    # 7. Built-in Demo: Middlebury 2021
    print("\n[Audit 7] Built-in Demo: Middlebury 2021 Stereo Metric Depth...")
    res = client.post("/api/reconstruct", data={"demo_sample_id": "middlebury", "method": "hybrid", "upsample_factor": "4"})
    assert res.status_code == 200, f"Middlebury failed: {res.text}"
    mid_data = res.json()
    assert mid_data["has_gt"] is True
    print(f" -> PASS: Middlebury reconstructed in {mid_data['latency_ms']}ms. RMSE: {mid_data['metrics']['rmse']}m, PSNR: {mid_data['metrics']['psnr']}dB")
    passed += 1

    print("\n" + "=" * 75)
    print(f"SCIENTIFIC AUDIT VERIFICATION: {passed}/{total} AUDIT CHECKS PASSED (100%)")
    print("=" * 75)

if __name__ == "__main__":
    run_scientific_audit()

