"""
test_custom_config_and_rag_gt.py
Verification suite for Custom Input Configuration, Algorithm Configuration,
and Reference Ground Truth synthesis.
"""

import sys
import os
import io
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app, parse_rgb_bytes, parse_depth_bytes, synthesize_gt_endpoint, run_reconstruction
from src.ai_depth_rag import synthesize_rag_reference_depth

class DummyFile:
    def __init__(self, filename, data):
        self.filename = filename
        self.file = io.BytesIO(data)
    def read(self):
        return self.file.getvalue()

def sync_run(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

def create_rgb_bytes(w=640, h=480, fmt="PNG"):
    arr = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()

def create_depth_bytes(w=640, h=480, fmt="PNG", is_16bit=True):
    if is_16bit:
        depth = np.linspace(1000, 4500, h * w).reshape((h, w)).astype(np.uint16)
        buf = io.BytesIO()
        Image.fromarray(depth).save(buf, format=fmt)
    else:
        depth = np.linspace(30, 220, h * w).reshape((h, w)).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(depth, mode="L").save(buf, format=fmt)
    return buf.getvalue()

def create_rgb_grayscale_depth_bytes(w=640, h=480):
    gray = np.linspace(40, 210, h * w).reshape((h, w)).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0].astype(int) + np.random.randint(-2, 3, (h, w)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def run_tests():
    print("=" * 75, flush=True)
    print("STARTING TEST SUITE: CUSTOM CONFIGURATION, ALGORITHMS & GROUND TRUTH", flush=True)
    print("=" * 75, flush=True)

    passed = 0
    total = 10

    # 1. Test Large Upload Resolution Clamping
    print("\n[TEST 1] Testing resolution clamping on 1920x1080 (1080p) uploads...", flush=True)
    rgb_large = create_rgb_bytes(1920, 1080)
    arr_rgb = parse_rgb_bytes(rgb_large, max_dim=720)
    assert max(arr_rgb.shape) <= 720, f"RGB not clamped: {arr_rgb.shape}"

    depth_large = create_depth_bytes(1920, 1080)
    arr_d, _, _, _ = parse_depth_bytes(depth_large, max_dim=720)
    assert max(arr_d.shape) <= 720, f"Depth not clamped: {arr_d.shape}"
    print(f" -> PASSED: Clamped 1920x1080 to {arr_rgb.shape} (RGB) and {arr_d.shape} (Depth).", flush=True)
    passed += 1

    # 2. Test 3-Channel Grayscale Depth PNG Handling
    print("\n[TEST 2] Testing 3-channel grayscale PNG parsing (e.g. from photo editors)...", flush=True)
    d_rgb_gray = create_rgb_grayscale_depth_bytes(640, 480)
    arr_d, d_type, d_unit, _ = parse_depth_bytes(d_rgb_gray, filename="Grayscale Depth Map of Building Facade.png")
    assert arr_d.ndim == 2, f"Expected 2D array, got {arr_d.ndim}D"
    assert arr_d.shape == (480, 640), f"Expected (480, 640), got {arr_d.shape}"
    print(f" -> PASSED: Cleanly parsed 3-channel grayscale depth map ({arr_d.shape}, {d_type}).", flush=True)
    passed += 1

    # 3. Test RAG Reference Ground Truth Synthesis Directly
    print("\n[TEST 3] Testing RAG Reference Ground Truth synthesis directly...", flush=True)
    rgb_test = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    sparse_d = (np.random.rand(240, 320) * 4.0 + 0.5).astype(np.float32)
    calib_gt, meta = synthesize_rag_reference_depth(rgb_test, sparse_or_input_depth=sparse_d)
    assert calib_gt.shape == (240, 320)
    assert calib_gt.min() > 0.05
    assert "calib_mode" in meta
    print(f" -> PASSED: Synthesized RAG GT: shape {calib_gt.shape}, range [{meta['min_depth']}m, {meta['max_depth']}m], mode: {meta['calib_mode']}", flush=True)
    passed += 1

    # 4. Test POST /api/reconstruct/synthesize_gt Endpoint
    print("\n[TEST 4] Testing POST /api/reconstruct/synthesize_gt endpoint...", flush=True)
    rgb_bytes = create_rgb_bytes(320, 240)
    f_rgb_gt = DummyFile(filename="facade.png", data=rgb_bytes)
    gt_data = synthesize_gt_endpoint(rgb_file=f_rgb_gt, depth_scale="auto")
    assert gt_data["success"] is True
    assert "gt_preview_b64" in gt_data and len(gt_data["gt_preview_b64"]) > 100
    assert "gt_meta" in gt_data
    session_id = gt_data["session_id"]
    print(f" -> PASSED: Endpoint returned session {session_id}, mode '{gt_data['gt_meta']['calib_mode']}'.", flush=True)
    passed += 1

    # 5. Test Full Reconstruction with Pseudo-GT in Session
    print("\n[TEST 5] Testing custom reconstruction with Pseudo-GT in session...", flush=True)
    d_bytes = create_depth_bytes(320, 240)
    f_rgb_rec = DummyFile(filename="Home.png", data=rgb_bytes)
    f_d_rec = DummyFile(filename="Grayscale Depth Map of Building Facade.png", data=d_bytes)
    
    data = sync_run(run_reconstruction(
        rgb_file=f_rgb_rec,
        depth_file=f_d_rec,
        session_id=session_id,
        method="hybrid",
        upsample_factor=4,
        apply_texture_suppression=True,
        suppression_strength=0.85,
        colormap="turbo"
    ))
    assert data["has_gt"] is True, "has_gt must be True when session has GT"
    assert data["metrics"] is not None, "Quantitative metrics must be computed"
    assert "rmse" in data["metrics"] and data["metrics"]["rmse"] > 0
    assert "mae" in data["metrics"] and data["metrics"]["mae"] > 0
    assert "psnr" in data["metrics"] and data["metrics"]["psnr"] > 0
    assert "ssim" in data["metrics"] and data["metrics"]["ssim"] > 0
    assert "boundary_f1" in data["metrics"]
    assert "error_b64" in data and len(data["error_b64"]) > 100, "Error Heatmap must be generated"
    print(f" -> PASSED: Reconstructed successfully! RMSE: {data['metrics']['rmse']:.4f}m, PSNR: {data['metrics']['psnr']:.2f}dB, SSIM: {data['metrics']['ssim']:.4f}, Latency: {data['latency_ms']}ms.", flush=True)
    passed += 1

    # 6. Test Custom Reconstruction WITHOUT Ground Truth (Missing GT Handling)
    print("\n[TEST 6] Testing custom reconstruction without Ground Truth (honest N/A evaluation)...", flush=True)
    f_rgb_fresh = DummyFile(filename="Home.png", data=rgb_bytes)
    f_d_fresh = DummyFile(filename="Depth.png", data=d_bytes)
    data_nogt = sync_run(run_reconstruction(
        rgb_file=f_rgb_fresh,
        depth_file=f_d_fresh,
        session_id="fresh_session_no_gt_123",
        method="hybrid",
        upsample_factor=4
    ))
    assert data_nogt["success"] is True
    assert data_nogt["has_gt"] is False, "has_gt must be False when no external GT was provided"
    assert data_nogt["metrics"] is not None, "Live real-time metrics must be computed"
    assert "rmse" in data_nogt["metrics"]
    assert "boundary_f1" in data_nogt["metrics"]
    assert data_nogt["error_b64"] is not None, "Real-time residual heatmap must be generated"
    assert data_nogt["reconstructed_b64"] is not None
    print(f" -> PASSED: Live real-time metrics computed! RMSE={data_nogt['metrics']['rmse']:.4f}, F1={data_nogt['metrics']['boundary_f1']:.4f}, latency={data_nogt['latency_ms']}ms.", flush=True)
    passed += 1

    # 7. Test All Upsampling Factors (2x, 4x, 8x)
    print("\n[TEST 7] Testing all Upsampling Factors (2x, 4x, 8x)...", flush=True)
    for factor in [2, 4, 8]:
        fdata = sync_run(run_reconstruction(
            session_id=session_id,
            method="bilinear",
            upsample_factor=factor
        ))
        assert fdata["success"] is True
        print(f"  * Upsampling Factor {factor}x: OK (Latency {fdata['latency_ms']}ms)", flush=True)
    passed += 1

    # 8. Test All 9 Reconstruction Algorithms
    print("\n[TEST 8] Testing all 9 Reconstruction Algorithms...", flush=True)
    algos = [
        "hybrid", "cnn_conf_refined", "cnn_conf", "cnn",
        "guided", "joint_bilateral", "bicubic", "bilinear", "nearest"
    ]
    for algo in algos:
        adata = sync_run(run_reconstruction(
            session_id=session_id,
            method=algo,
            upsample_factor=4
        ))
        assert adata["success"] is True
        print(f"  * Algorithm '{algo}' ({adata['algorithm_stage']}): OK ({adata['latency_ms']}ms)", flush=True)
    passed += 1

    # 9. Test All 6 Scientific Colormaps
    print("\n[TEST 9] Testing all 6 Scientific Colormaps...", flush=True)
    colormaps = ["turbo", "viridis", "magma", "inferno", "grayscale", "inverse_grayscale"]
    for cmap in colormaps:
        cdata = sync_run(run_reconstruction(
            session_id=session_id,
            method="bilinear",
            colormap=cmap
        ))
        assert cdata["reconstructed_b64"] is not None
        print(f"  * Colormap '{cmap}': OK", flush=True)
    passed += 1

    # 10. Test Scale Modes
    print("\n[TEST 10] Testing Physical Scale Calibration options...", flush=True)
    scale_modes = ["auto", "mm_to_m", "kitti", "raw_meters", "relative"]
    for sm in scale_modes:
        sdata = sync_run(run_reconstruction(
            session_id=session_id,
            method="bilinear",
            depth_scale=sm
        ))
        assert sdata["success"] is True
        print(f"  * Scale mode '{sm}': OK (Depth unit: {sdata['depth_range']['unit']})", flush=True)
    passed += 1

    print("\n" + "=" * 75, flush=True)
    print(f"RESULT: {passed}/{total} TESTS PASSED (100% SUCCESS)", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_tests()
