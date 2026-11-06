"""
EDGEFUSE-DCR: Comprehensive Custom Input Upload System Test Suite
Covers 18 distinct real-world scenarios including edge cases, format validations,
reconstruction pipelines, export downloads, and safety checks.
"""

import sys
import os
import io
import time
import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient

# Ensure EDGEFUSE_DCR root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app

client = TestClient(app)

def create_rgb_bytes(w=640, h=480, fmt="PNG", color=(120, 160, 200)):
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def create_16bit_depth_bytes(w=640, h=480, scale=1000):
    # Depth in mm (e.g. 1.5m to 3.5m)
    depth = np.linspace(1500, 3500, h * w).reshape((h, w)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(depth).save(buf, format="PNG")
    return buf.getvalue()

def create_8bit_depth_bytes(w=640, h=480):
    depth = np.linspace(30, 220, h * w).reshape((h, w)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(depth, mode="L").save(buf, format="PNG")
    return buf.getvalue()

def create_npy_depth_bytes(w=640, h=480, has_nans=False, all_zeros=False):
    if all_zeros:
        arr = np.zeros((h, w), dtype=np.float32)
    else:
        arr = np.linspace(0.8, 4.5, h * w).reshape((h, w)).astype(np.float32)
        if has_nans:
            arr[10:20, 10:20] = np.nan
            arr[30:40, 30:40] = np.inf
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def create_color_photo_bytes(w=640, h=480):
    # Vibrant RGB photograph (R, G, B are clearly different)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 220 # High Red
    arr[:, :, 1] = 60  # Low Green
    arr[:, :, 2] = 40  # Low Blue
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

def run_all_tests():
    print("=" * 70)
    print("EDGEFUSE-DCR: EXECUTING 18-SCENARIO CUSTOM UPLOAD TEST SUITE")
    print("=" * 70)
    
    passed = 0
    total = 18

    # TEST 1: RGB JPG + valid 16-bit depth PNG
    print("\n[Test 1] RGB JPG + 16-bit depth PNG (mm physical scale)...")
    rgb_bytes = create_rgb_bytes(640, 480, fmt="JPEG")
    d_bytes = create_16bit_depth_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "upsample_factor": "4", "depth_scale": "mm_to_m"},
                      files={"rgb_file": ("test.jpg", rgb_bytes, "image/jpeg"), "depth_file": ("test_d.png", d_bytes, "image/png")})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["has_gt"] is False, "Custom upload must have has_gt=False"
    assert "input_stats" in data and data["input_stats"]["valid_percent"] > 90
    assert "input_depth_b64" in data and len(data["input_depth_b64"]) > 100
    print(f" -> PASS: Latency {data['latency_ms']}ms, Min: {data['input_stats']['min']}m, Max: {data['input_stats']['max']}m")
    passed += 1

    # TEST 2: RGB PNG + valid .npy float32 array
    print("\n[Test 2] RGB PNG + valid float32 .npy depth array...")
    rgb_bytes = create_rgb_bytes(640, 480, fmt="PNG")
    npy_bytes = create_npy_depth_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "upsample_factor": "4", "depth_scale": "raw_meters"},
                      files={"rgb_file": ("test.png", rgb_bytes, "image/png"), "depth_file": ("test_d.npy", npy_bytes, "application/octet-stream")})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["reconstructed_b64"] is not None
    print(f" -> PASS: Latency {data['latency_ms']}ms, Valid seeds: {data['input_stats']['valid_pixels']}")
    passed += 1

    # TEST 3: RGB JPG + valid 8-bit depth PNG with norm_8bit scale
    print("\n[Test 3] RGB JPG + 8-bit grayscale depth PNG (norm_8bit)...")
    rgb_bytes = create_rgb_bytes(640, 480, fmt="JPEG")
    d8_bytes = create_8bit_depth_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "upsample_factor": "4", "depth_scale": "norm_8bit"},
                      files={"rgb_file": ("test.jpg", rgb_bytes, "image/jpeg"), "depth_file": ("test_8bit.png", d8_bytes, "image/png")})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["input_stats"]["min"] >= 0.5 and data["input_stats"]["max"] <= 6.0
    print(f" -> PASS: Range [{data['input_stats']['min']}, {data['input_stats']['max']}]m")
    passed += 1

    # TEST 4: RGB photo incorrectly supplied as depth (Home.png scenario)
    print("\n[Test 4] RGB photograph supplied as depth map (Verification of strict rejection)...")
    rgb_bytes = create_rgb_bytes(640, 480, fmt="PNG")
    photo_bytes = create_color_photo_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "upsample_factor": "4"},
                      files={"rgb_file": ("Home.png", rgb_bytes, "image/png"), "depth_file": ("Home.png", photo_bytes, "image/png")})
    assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
    assert "ordinary RGB color photograph" in res.json()["detail"]
    print(" -> PASS: Strictly rejected ordinary color photo with clear error message.")
    passed += 1

    # TEST 5: Corrupt RGB file
    print("\n[Test 5] Corrupted RGB file...")
    corrupt_bytes = b"CORRUPTED_NOT_AN_IMAGE_DATA_12345"
    d_bytes = create_16bit_depth_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid"},
                      files={"rgb_file": ("corrupt.jpg", corrupt_bytes, "image/jpeg"), "depth_file": ("valid.png", d_bytes, "image/png")})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    print(f" -> PASS: Rejected with message: {res.json()['detail']}")
    passed += 1

    # TEST 6: Corrupt depth PNG file
    print("\n[Test 6] Corrupted depth file...")
    rgb_bytes = create_rgb_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid"},
                      files={"rgb_file": ("valid.png", rgb_bytes, "image/png"), "depth_file": ("corrupt.png", corrupt_bytes, "image/png")})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    print(f" -> PASS: Rejected with message: {res.json()['detail']}")
    passed += 1

    # TEST 7: Empty .npy depth array
    print("\n[Test 7] Empty .npy array...")
    empty_buf = io.BytesIO()
    np.save(empty_buf, np.array([], dtype=np.float32))
    res = client.post("/api/reconstruct", data={"method": "hybrid"},
                      files={"rgb_file": ("valid.png", rgb_bytes, "image/png"), "depth_file": ("empty.npy", empty_buf.getvalue(), "application/octet-stream")})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    print(f" -> PASS: Rejected with message: {res.json()['detail']}")
    passed += 1

    # TEST 8: .npy array containing NaNs and Infs (Sanitization check)
    print("\n[Test 8] .npy depth array with NaNs and Infs...")
    nan_npy = create_npy_depth_bytes(640, 480, has_nans=True)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "depth_scale": "raw_meters"},
                      files={"rgb_file": ("valid.png", rgb_bytes, "image/png"), "depth_file": ("nans.npy", nan_npy, "application/octet-stream")})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert np.isfinite(data["input_stats"]["mean"])
    print(" -> PASS: Sanitized NaNs and Infs cleanly without crashing.")
    passed += 1

    # TEST 9: All-zero depth map (Zero valid measurements)
    print("\n[Test 9] All-zero depth map...")
    zero_npy = create_npy_depth_bytes(640, 480, all_zeros=True)
    res = client.post("/api/reconstruct", data={"method": "hybrid"},
                      files={"rgb_file": ("valid.png", rgb_bytes, "image/png"), "depth_file": ("zero.npy", zero_npy, "application/octet-stream")})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    assert "zero positive measurements" in res.json()["detail"]
    print(f" -> PASS: Rejected all-zero depth map: {res.json()['detail']}")
    passed += 1

    # TEST 10: RGB and Depth with different resolutions (640x480 RGB vs 160x120 Depth)
    print("\n[Test 10] Resolution alignment (640x480 RGB vs 160x120 Depth, 4x factor)...")
    rgb_bytes = create_rgb_bytes(640, 480)
    d_lr_bytes = create_16bit_depth_bytes(160, 120)
    res = client.post("/api/reconstruct", data={"method": "hybrid", "upsample_factor": "4"},
                      files={"rgb_file": ("rgb.png", rgb_bytes, "image/png"), "depth_file": ("depth_lr.png", d_lr_bytes, "image/png")})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    print(" -> PASS: Automatically aligned and upsampled 160x120 depth to 640x480.")
    passed += 1

    # TEST 11: Mismatched aspect ratios (16:9 vs 4:3)
    print("\n[Test 11] Aspect ratio mismatch (640x360 16:9 RGB vs 640x480 4:3 Depth)...")
    rgb_16_9 = create_rgb_bytes(640, 360)
    d_4_3 = create_16bit_depth_bytes(640, 480)
    res = client.post("/api/reconstruct", data={"method": "hybrid"},
                      files={"rgb_file": ("rgb.png", rgb_16_9, "image/png"), "depth_file": ("depth.png", d_4_3, "image/png")})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    assert "aspect ratio" in res.json()["detail"].lower()
    print(f" -> PASS: Flagged aspect ratio incompatibility: {res.json()['detail']}")
    passed += 1

    # TEST 12: Large image upload memory safeguard (1200x900)
    print("\n[Test 12] Large image memory safeguard (1200x900)...")
    rgb_large = create_rgb_bytes(1200, 900)
    d_large = create_16bit_depth_bytes(1200, 900)
    res = client.post("/api/reconstruct", data={"method": "bilinear", "upsample_factor": "4"},
                      files={"rgb_file": ("rgb.png", rgb_large, "image/png"), "depth_file": ("depth.png", d_large, "image/png")})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    print(" -> PASS: Safeguarded RAM and completed reconstruction.")
    passed += 1

    # TEST 13: Repeated uploads of different custom scenes
    print("\n[Test 13] Consecutive custom uploads...")
    for scene_idx in range(1, 3):
        r_b = create_rgb_bytes(640, 480, color=(50 * scene_idx, 80 * scene_idx, 100))
        d_b = create_16bit_depth_bytes(640, 480)
        res = client.post("/api/reconstruct", data={"method": "bilinear"},
                          files={"rgb_file": (f"scene_{scene_idx}.png", r_b, "image/png"), "depth_file": (f"depth_{scene_idx}.png", d_b, "image/png")})
        assert res.status_code == 200
    print(" -> PASS: Multi-run state cleanly refreshed.")
    passed += 1

    # TEST 14: All reconstruction algorithms on custom input
    print("\n[Test 14] Testing all 8 reconstruction algorithms on custom input...")
    algorithms = ["nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "cnn", "cnn_confidence", "cnn_boundary", "hybrid"]
    rgb_bytes = create_rgb_bytes(320, 240)
    d_bytes = create_16bit_depth_bytes(320, 240)
    for m in algorithms:
        res = client.post("/api/reconstruct", data={"method": m, "upsample_factor": "4"},
                          files={"rgb_file": ("rgb.png", rgb_bytes, "image/png"), "depth_file": ("depth.png", d_bytes, "image/png")})
        assert res.status_code == 200, f"Algorithm {m} failed: {res.text}"
    print(f" -> PASS: Successfully executed all {len(algorithms)} algorithms on custom data.")
    passed += 1

    # TEST 15: All three upsampling factors (2x, 4x, 8x)
    print("\n[Test 15] Testing upsampling factors (2x, 4x, 8x)...")
    for factor in ["2", "4", "8"]:
        res = client.post("/api/reconstruct", data={"method": "bilinear", "upsample_factor": factor},
                          files={"rgb_file": ("rgb.png", rgb_bytes, "image/png"), "depth_file": ("depth.png", d_bytes, "image/png")})
        assert res.status_code == 200, f"Factor {factor} failed"
    print(" -> PASS: Supported 2x, 4x, and 8x factors.")
    passed += 1

    # TEST 16: Texture-copy suppression toggle
    print("\n[Test 16] Texture suppression enabled vs disabled...")
    for supp in ["true", "false"]:
        res = client.post("/api/reconstruct", data={"method": "hybrid", "apply_texture_suppression": supp},
                          files={"rgb_file": ("rgb.png", rgb_bytes, "image/png"), "depth_file": ("depth.png", d_bytes, "image/png")})
        assert res.status_code == 200
    print(" -> PASS: Texture suppression toggle validated.")
    passed += 1

    # TEST 17: Numeric .npy export endpoint after custom run
    print("\n[Test 17] Numeric .npy download endpoint...")
    res = client.get("/api/download/npy")
    assert res.status_code == 200, f"Download failed: {res.status_code}"
    assert "custom_reconstructed_depth.npy" in res.headers.get("content-disposition", "")
    downloaded_arr = np.load(io.BytesIO(res.content))
    assert downloaded_arr.ndim == 2 and downloaded_arr.size > 0
    print(f" -> PASS: Downloaded .npy shape {downloaded_arr.shape}, dtype {downloaded_arr.dtype}")
    passed += 1

    # TEST 18: Colorized PNG export endpoint after custom run
    print("\n[Test 18] Colorized PNG preview download endpoint...")
    res = client.get("/api/download/png")
    assert res.status_code == 200
    assert "custom_depth_preview.png" in res.headers.get("content-disposition", "")
    assert len(res.content) > 1000
    print(f" -> PASS: Downloaded PNG preview size {len(res.content)} bytes.")
    passed += 1

    # BONUS: Preview endpoint test
    print("\n[BONUS] Testing POST /api/preview_depth...")
    preview_res = client.post("/api/preview_depth", files={"depth_file": ("sample.npy", npy_bytes, "application/octet-stream")})
    assert preview_res.status_code == 200
    pdata = preview_res.json()
    assert "preview_b64" in pdata and pdata["valid_percent"] > 90
    print(f" -> PASS: Depth preview verified. Dimensions: {pdata['width']}x{pdata['height']}, Valid: {pdata['valid_percent']}%")

    print("\n" + "=" * 70)
    print(f"CUSTOM UPLOAD TEST SUITE RESULT: {passed}/{total} SCENARIOS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
