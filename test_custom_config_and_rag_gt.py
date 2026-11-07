"""
test_custom_config_and_rag_gt.py
Verification suite for Custom Input Configuration, Algorithm Configuration,
and RAG-driven Reference Ground Truth synthesis.
Writes progress and results to test_custom_config_results.txt.
"""

import sys
import os
import io
import time
import asyncio
import numpy as np
from PIL import Image
from fastapi import UploadFile

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app, parse_rgb_bytes, parse_depth_bytes, synthesize_gt_endpoint, run_reconstruction
from src.ai_depth_rag import synthesize_rag_reference_depth

LOG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_custom_config_results.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

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


async def run_tests_async():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("STARTING TEST SUITE\n")

    log("=" * 75)
    log("STARTING TEST SUITE: CUSTOM CONFIGURATION, ALGORITHMS & RAG GROUND TRUTH")
    log("=" * 75)

    passed = 0
    total = 9

    # 1. Test Large Upload Resolution Clamping
    log("\n[TEST 1] Testing resolution clamping on 1920x1080 (1080p) uploads...")
    rgb_large = create_rgb_bytes(1920, 1080)
    arr_rgb = parse_rgb_bytes(rgb_large, max_dim=720)
    assert max(arr_rgb.shape) <= 720, f"RGB not clamped: {arr_rgb.shape}"

    depth_large = create_depth_bytes(1920, 1080)
    arr_d, _, _, _ = parse_depth_bytes(depth_large, max_dim=720)
    assert max(arr_d.shape) <= 720, f"Depth not clamped: {arr_d.shape}"
    log(f" -> PASSED: Clamped 1920x1080 to {arr_rgb.shape} (RGB) and {arr_d.shape} (Depth).")
    passed += 1

    # 2. Test 3-Channel Grayscale Depth PNG Handling
    log("\n[TEST 2] Testing 3-channel grayscale PNG parsing (e.g. from photo editors)...")
    d_rgb_gray = create_rgb_grayscale_depth_bytes(640, 480)
    arr_d, d_type, d_unit, _ = parse_depth_bytes(d_rgb_gray, filename="Grayscale Depth Map of Building Facade.png")
    assert arr_d.ndim == 2, f"Expected 2D array, got {arr_d.ndim}D"
    assert arr_d.shape == (480, 640), f"Expected (480, 640), got {arr_d.shape}"
    log(f" -> PASSED: Cleanly parsed 3-channel grayscale depth map ({arr_d.shape}, {d_type}).")
    passed += 1

    # 3. Test RAG Reference Ground Truth Synthesis Directly
    log("\n[TEST 3] Testing RAG Reference Ground Truth synthesis directly...")
    rgb_test = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    sparse_d = (np.random.rand(480, 640) * 4.0 + 0.5).astype(np.float32)
    calib_gt, meta = synthesize_rag_reference_depth(rgb_test, sparse_or_input_depth=sparse_d)
    assert calib_gt.shape == (480, 640)
    assert calib_gt.min() > 0.05
    assert "calib_mode" in meta
    log(f" -> PASSED: Synthesized RAG GT: shape {calib_gt.shape}, range [{meta['min_depth']}m, {meta['max_depth']}m], mode: {meta['calib_mode']}")
    passed += 1

    # 4. Test POST /api/reconstruct/synthesize_gt Endpoint
    log("\n[TEST 4] Testing POST /api/reconstruct/synthesize_gt endpoint...")
    rgb_bytes = create_rgb_bytes(640, 480)
    f_rgb_gt = UploadFile(filename="facade.png", file=io.BytesIO(rgb_bytes))
    gt_data = synthesize_gt_endpoint(rgb_file=f_rgb_gt, depth_scale="auto")
    assert gt_data["success"] is True
    assert "gt_preview_b64" in gt_data and len(gt_data["gt_preview_b64"]) > 100
    assert "gt_meta" in gt_data
    session_id = gt_data["session_id"]
    log(f" -> PASSED: Endpoint returned session {session_id}, mode '{gt_data['gt_meta']['calib_mode']}'.")
    passed += 1

    # 5. Test Full Reconstruction with Auto RAG Ground Truth (No manual GT file)
    log("\n[TEST 5] Testing custom reconstruction with automatic RAG Reference Ground Truth...")
    d_bytes = create_depth_bytes(640, 480)
    f_rgb_rec = UploadFile(filename="Home.png", file=io.BytesIO(rgb_bytes))
    f_d_rec = UploadFile(filename="Grayscale Depth Map of Building Facade.png", file=io.BytesIO(d_bytes))
    
    data = await run_reconstruction(
        rgb_file=f_rgb_rec,
        depth_file=f_d_rec,
        session_id=session_id,
        method="hybrid",
        upsample_factor=4,
        apply_texture_suppression=True,
        suppression_strength=0.85,
        colormap="turbo",
        auto_synthesize_gt=True
    )
    assert data["has_gt"] is True, "has_gt must be True when RAG reference is active"
    assert data["metrics"] is not None, "Quantitative metrics must be computed"
    assert "rmse" in data["metrics"] and data["metrics"]["rmse"] > 0
    assert "mae" in data["metrics"] and data["metrics"]["mae"] > 0
    assert "psnr" in data["metrics"] and data["metrics"]["psnr"] > 0
    assert "ssim" in data["metrics"] and data["metrics"]["ssim"] > 0
    assert "boundary_f1" in data["metrics"]
    assert "error_b64" in data and len(data["error_b64"]) > 100, "Error Heatmap must be generated"
    log(f" -> PASSED: Reconstructed successfully! RMSE: {data['metrics']['rmse']:.4f}m, PSNR: {data['metrics']['psnr']:.2f}dB, SSIM: {data['metrics']['ssim']:.4f}, Latency: {data['latency_ms']}ms.")
    passed += 1

    # 6. Test All Upsampling Factors (2x, 4x, 8x)
    log("\n[TEST 6] Testing all Upsampling Factors (2x, 4x, 8x)...")
    for factor in [2, 4, 8]:
        fdata = await run_reconstruction(
            session_id=session_id,
            method="bilinear",
            upsample_factor=factor
        )
        assert fdata["success"] is True
        log(f"  * Upsampling Factor {factor}x: OK (Latency {fdata['latency_ms']}ms)")
    passed += 1

    # 7. Test All 9 Reconstruction Algorithms
    log("\n[TEST 7] Testing all 9 Reconstruction Algorithms...")
    algos = [
        "hybrid", "cnn_conf_refined", "cnn_conf", "cnn",
        "guided", "joint_bilateral", "bicubic", "bilinear", "nearest"
    ]
    for algo in algos:
        adata = await run_reconstruction(
            session_id=session_id,
            method=algo,
            upsample_factor=4
        )
        assert adata["success"] is True
        log(f"  * Algorithm '{algo}' ({adata['algorithm_stage']}): OK ({adata['latency_ms']}ms)")
    passed += 1

    # 8. Test All 6 Scientific Colormaps
    log("\n[TEST 8] Testing all 6 Scientific Colormaps...")
    colormaps = ["turbo", "viridis", "magma", "inferno", "grayscale", "inverse_grayscale"]
    for cmap in colormaps:
        cdata = await run_reconstruction(
            session_id=session_id,
            method="bilinear",
            colormap=cmap
        )
        assert cdata["reconstructed_b64"] is not None
        log(f"  * Colormap '{cmap}': OK")
    passed += 1

    # 9. Test Scale Modes
    log("\n[TEST 9] Testing Physical Scale Calibration options...")
    scale_modes = ["auto", "mm_to_m", "kitti", "raw_meters", "relative"]
    for sm in scale_modes:
        sdata = await run_reconstruction(
            session_id=session_id,
            method="bilinear",
            depth_scale=sm
        )
        assert sdata["success"] is True
        log(f"  * Scale mode '{sm}': OK (Depth unit: {sdata['depth_range']['unit']})")
    passed += 1

    log("\n" + "=" * 75)
    log(f"RESULT: {passed}/{total} TESTS PASSED (100% SUCCESS)")
    log("=" * 75)

if __name__ == "__main__":
    asyncio.run(run_tests_async())
