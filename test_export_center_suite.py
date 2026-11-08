"""
test_export_center_suite.py - Comprehensive Verification of Research Export Center
Tests all 8 export artifacts, bulk ZIP generation, 16-bit millimeter PNG encoding,
run ID synchronization, status API, and ground truth error map gating.
"""

import io
import json
import zipfile
import numpy as np
import cv2
from starlette.testclient import TestClient

from app import app, CATALOG


def test_export_center_comprehensive():
    client = TestClient(app)

    print("\n" + "=" * 75)
    print("RUNNING EDGEFUSE-DCR RESEARCH EXPORT CENTER TEST SUITE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # TEST 1: Initial Export Status (before reconstruction)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Initial Export Status Check (Pre-Reconstruction)...")
    if hasattr(app.state, "last_rec"):
        delattr(app.state, "last_rec")

    resp = client.get("/api/export/status")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data["has_reconstruction"] is False
    assert data["available_count"] == 0
    print("  -> Pre-reconstruction status verified: has_reconstruction = False, available = 0")

    # -------------------------------------------------------------------------
    # TEST 2: Execute Benchmark Scene Reconstruction (with authentic GT)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Executing Benchmark Scene Reconstruction (nyu_living_room)...")
    payload = {
        "method": "hybrid",
        "demo_sample_id": "nyu_living_room",
        "upsample_factor": 4,
        "apply_texture_suppression": True,
        "suppression_strength": 0.85,
        "colormap": "turbo"
    }
    rec_resp = client.post("/api/reconstruct", data=payload)
    assert rec_resp.status_code == 200, f"Reconstruction failed: {rec_resp.text}"
    rec_data = rec_resp.json()
    assert rec_data["success"] is True
    run_id = rec_data.get("run_id")
    assert run_id and run_id.startswith("RUN-"), f"Invalid run_id: {run_id}"
    print(f"  -> Reconstruction succeeded: Run ID = {run_id}, Method = {rec_data['method']}")

    # -------------------------------------------------------------------------
    # TEST 3: Post-Reconstruction Export Status API
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verifying Post-Reconstruction Export Status API...")
    resp = client.get("/api/export/status")
    assert resp.status_code == 200
    st_data = resp.json()
    assert st_data["has_reconstruction"] is True
    assert st_data["run_id"] == run_id
    assert st_data["has_gt"] is True
    assert st_data["available_count"] == 8
    exports = st_data["exports"]
    assert "npy" in exports and exports["npy"]["ready"] is True
    assert "png_16bit" in exports and exports["png_16bit"]["ready"] is True
    assert "confidence" in exports and exports["confidence"]["ready"] is True
    assert "valid_mask" in exports and exports["valid_mask"]["ready"] is True
    assert "boundary" in exports and exports["boundary"]["ready"] is True
    assert "error_map" in exports and exports["error_map"]["ready"] is True
    assert "report_json" in exports and exports["report_json"]["ready"] is True
    assert "point_cloud" in exports and exports["point_cloud"]["ready"] is True
    assert "all_zip" in exports and exports["all_zip"]["ready"] is True
    print(f"  -> Export status verified: all 8 artifacts + bulk ZIP are ready.")

    # -------------------------------------------------------------------------
    # TEST 4: Export 1 — Reconstructed Depth (.npy)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing Export 1: Reconstructed Depth (.npy)...")
    npy_resp = client.get("/api/download/npy")
    assert npy_resp.status_code == 200
    disp = npy_resp.headers.get("Content-Disposition", "")
    assert f"edgefuse_reconstructed_depth_{run_id}.npy" in disp, f"Unexpected header: {disp}"
    depth_arr = np.load(io.BytesIO(npy_resp.content))
    assert depth_arr.dtype == np.float32, f"Expected float32, got {depth_arr.dtype}"
    assert depth_arr.ndim == 2, f"Expected 2D array, got {depth_arr.ndim}D"
    assert depth_arr.min() >= 0.0 and depth_arr.max() <= 10.0
    print(f"  -> Reconstructed Depth .npy verified: shape = {depth_arr.shape}, dtype = float32, range = [{depth_arr.min():.2f}m, {depth_arr.max():.2f}m]")

    # -------------------------------------------------------------------------
    # TEST 5: Export 2 — Metric Depth Image (16-bit PNG)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing Export 2: Metric Depth Image (16-bit PNG)...")
    png_resp = client.get("/api/download/png")
    assert png_resp.status_code == 200
    disp = png_resp.headers.get("Content-Disposition", "")
    assert f"edgefuse_depth_mm_{run_id}.png" in disp, f"Unexpected header: {disp}"
    
    # Decode as unchanged single-channel uint16
    png_buf = np.frombuffer(png_resp.content, np.uint8)
    depth_uint16 = cv2.imdecode(png_buf, cv2.IMREAD_UNCHANGED)
    assert depth_uint16.dtype == np.uint16, f"Expected uint16, got {depth_uint16.dtype}"
    assert depth_uint16.ndim == 2, f"Expected single channel (2D), got {depth_uint16.shape}"
    assert depth_uint16.shape == depth_arr.shape
    # Check millimeter scaling: 1 px = 1 mm (e.g. 1.5m -> 1500)
    expected_mean_mm = np.mean(depth_arr) * 1000.0
    actual_mean_mm = float(np.mean(depth_uint16))
    diff = abs(expected_mean_mm - actual_mean_mm)
    assert diff < 5.0, f"Millimeter scaling error: expected mean {expected_mean_mm}, got {actual_mean_mm}"
    print(f"  -> Metric Depth 16-bit PNG verified: shape = {depth_uint16.shape}, dtype = uint16, millimeter scaling diff = {diff:.2f}mm")

    # -------------------------------------------------------------------------
    # TEST 6: Export 3 — Confidence Weights Map (.png)
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing Export 3: Confidence Weights Map (.png)...")
    conf_resp = client.get("/api/download/confidence")
    assert conf_resp.status_code == 200
    assert f"edgefuse_confidence_{run_id}.png" in conf_resp.headers.get("Content-Disposition", "")
    conf_buf = np.frombuffer(conf_resp.content, np.uint8)
    conf_img = cv2.imdecode(conf_buf, cv2.IMREAD_COLOR)
    assert conf_img is not None and conf_img.ndim == 3
    print(f"  -> Confidence Map verified: shape = {conf_img.shape}, format = Viridis RGB")

    # -------------------------------------------------------------------------
    # TEST 7: Export 4 — Sensor Validity Mask (.png)
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing Export 4: Sensor Validity Mask (.png)...")
    mask_resp = client.get("/api/download/valid_mask")
    assert mask_resp.status_code == 200
    assert f"edgefuse_sensor_validity_{run_id}.png" in mask_resp.headers.get("Content-Disposition", "")
    mask_buf = np.frombuffer(mask_resp.content, np.uint8)
    mask_img = cv2.imdecode(mask_buf, cv2.IMREAD_GRAYSCALE)
    assert mask_img is not None
    unique_vals = np.unique(mask_img)
    assert all(v in [0, 255] for v in unique_vals), f"Unexpected mask values: {unique_vals}"
    print(f"  -> Sensor Validity Mask verified: binary values = {unique_vals.tolist()}")

    # -------------------------------------------------------------------------
    # TEST 8: Export 5 — Boundary Overlay (.png)
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing Export 5: Boundary Overlay (.png)...")
    bound_resp = client.get("/api/download/boundary")
    assert bound_resp.status_code == 200
    assert f"edgefuse_boundary_overlay_{run_id}.png" in bound_resp.headers.get("Content-Disposition", "")
    bound_buf = np.frombuffer(bound_resp.content, np.uint8)
    bound_img = cv2.imdecode(bound_buf, cv2.IMREAD_COLOR)
    assert bound_img is not None and bound_img.ndim == 3
    print(f"  -> Boundary Overlay verified: shape = {bound_img.shape}")

    # -------------------------------------------------------------------------
    # TEST 9: Export 6 — Absolute Error Map (.png)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Testing Export 6: Absolute Error Map (with GT)...")
    err_resp = client.get("/api/download/error_map")
    assert err_resp.status_code == 200
    assert f"edgefuse_absolute_error_{run_id}.png" in err_resp.headers.get("Content-Disposition", "")
    err_buf = np.frombuffer(err_resp.content, np.uint8)
    err_img = cv2.imdecode(err_buf, cv2.IMREAD_COLOR)
    assert err_img is not None and err_img.ndim == 3
    print(f"  -> Absolute Error Heatmap verified: shape = {err_img.shape}")

    # -------------------------------------------------------------------------
    # TEST 10: Export 7 — Research Report (.json)
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Testing Export 7: Research Report (.json)...")
    rep_resp = client.get("/api/download/report_json")
    assert rep_resp.status_code == 200
    assert f"edgefuse_research_report_{run_id}.json" in rep_resp.headers.get("Content-Disposition", "")
    rep_json = rep_resp.json()
    assert rep_json["run_id"] == run_id
    assert rep_json["project"] == "EdgeFuse-DCR"
    assert rep_json["algorithm_stage"] == "A5"
    assert "quantitative_metrics" in rep_json and rep_json["quantitative_metrics"] is not None
    assert "latency_breakdown_ms" in rep_json
    assert "depth_range_meters" in rep_json
    print(f"  -> Research Report JSON verified: RMSE = {rep_json['quantitative_metrics']['rmse']}, PSNR = {rep_json['quantitative_metrics']['psnr']}")

    # -------------------------------------------------------------------------
    # TEST 11: Export 8 — 3D Point Cloud Mesh (.ply)
    # -------------------------------------------------------------------------
    print("\n[TEST 11] Testing Export 8: 3D Point Cloud Mesh (.ply)...")
    ply_resp = client.get("/api/download/ply")
    assert ply_resp.status_code == 200
    assert f"edgefuse_pointcloud_{run_id}.ply" in ply_resp.headers.get("Content-Disposition", "")
    ply_text = ply_resp.text
    assert ply_text.startswith("ply\nformat ascii 1.0")
    assert "element vertex" in ply_text
    assert "property float x" in ply_text
    assert "property uchar red" in ply_text
    assert "end_header" in ply_text
    lines = ply_text.strip().split("\n")
    print(f"  -> 3D Point Cloud PLY verified: total lines = {len(lines)}, header valid")

    # -------------------------------------------------------------------------
    # TEST 12: Bulk Export (ZIP Archive)
    # -------------------------------------------------------------------------
    print("\n[TEST 12] Testing Bulk Export: Download All Available (ZIP)...")
    zip_resp = client.get("/api/download/all")
    assert zip_resp.status_code == 200
    assert f"edgefuse_research_bundle_{run_id}.zip" in zip_resp.headers.get("Content-Disposition", "")
    
    zf = zipfile.ZipFile(io.BytesIO(zip_resp.content))
    namelist = zf.namelist()
    assert f"depth/edgefuse_reconstructed_depth_{run_id}.npy" in namelist
    assert f"depth/edgefuse_depth_mm_{run_id}.png" in namelist
    assert f"confidence/edgefuse_confidence_{run_id}.png" in namelist
    assert f"masks/edgefuse_sensor_validity_{run_id}.png" in namelist
    assert f"visualizations/edgefuse_boundary_overlay_{run_id}.png" in namelist
    assert f"visualizations/edgefuse_absolute_error_{run_id}.png" in namelist
    assert f"reports/edgefuse_research_report_{run_id}.json" in namelist
    assert f"pointcloud/edgefuse_pointcloud_{run_id}.ply" in namelist
    assert "manifest.json" in namelist
    assert "README.txt" in namelist

    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["has_ground_truth"] is True
    readme = zf.read("README.txt").decode("utf-8")
    assert "EDGEFUSE-DCR RESEARCH EXPORT ARCHIVE" in readme
    assert run_id in readme
    print(f"  -> Bulk ZIP Bundle verified: {len(namelist)} entries in archive, manifest and README confirmed.")

    # -------------------------------------------------------------------------
    # TEST 13: Ground Truth Gating Check (Custom Upload without GT)
    # -------------------------------------------------------------------------
    print("\n[TEST 13] Testing Ground Truth Gating on Custom Upload without GT...")
    # Create fake RGB and Depth without GT
    dummy_rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    dummy_depth = np.full((120, 160), 1.5, dtype=np.float32)
    _, rgb_enc = cv2.imencode(".png", dummy_rgb)
    _, depth_enc = cv2.imencode(".png", (dummy_depth * 1000).astype(np.uint16))

    custom_files = {
        "rgb_file": ("test_rgb.png", rgb_enc.tobytes(), "image/png"),
        "depth_file": ("test_depth.png", depth_enc.tobytes(), "image/png"),
    }
    custom_data = {
        "method": "hybrid",
        "upsample_factor": 4,
        "session_id": "test_no_gt_session"
    }

    c_rec_resp = client.post("/api/reconstruct", files=custom_files, data=custom_data)
    assert c_rec_resp.status_code == 200
    c_data = c_rec_resp.json()
    assert c_data["has_gt"] is False
    c_run_id = c_data["run_id"]

    # Check status API reflects has_gt = False and error_map is not ready
    c_st_resp = client.get("/api/export/status")
    c_st = c_st_resp.json()
    assert c_st["has_gt"] is False
    assert c_st["exports"]["error_map"]["ready"] is False
    assert c_st["available_count"] == 7
    print("  -> Status correctly reflects 7 of 8 artifacts ready when GT is missing.")

    # Attempting to download error map without GT should return 400 Bad Request
    c_err_resp = client.get("/api/download/error_map")
    assert c_err_resp.status_code == 400
    assert "Ground Truth reference error map is not available" in c_err_resp.json()["detail"]
    print("  -> Error map endpoint correctly returned HTTP 400 when Ground Truth is absent.")

    # Bulk ZIP without GT should omit error heatmap
    c_zip_resp = client.get("/api/download/all")
    assert c_zip_resp.status_code == 200
    c_zf = zipfile.ZipFile(io.BytesIO(c_zip_resp.content))
    c_namelist = c_zf.namelist()
    assert not any("error" in name for name in c_namelist), "Error heatmap should not be in zip when GT is missing!"
    print("  -> Bulk ZIP correctly excluded error heatmap when Ground Truth was not provided.")

    print("\n" + "=" * 75)
    print("ALL 13 RESEARCH EXPORT CENTER TESTS PASSED PERFECTLY!")
    print("=" * 75)


if __name__ == "__main__":
    test_export_center_comprehensive()

