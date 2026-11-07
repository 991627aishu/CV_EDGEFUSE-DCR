"""
EDGEFUSE-DCR — Comprehensive Verification Test Suite for AI Monocular Depth Estimation
Tests Depth Anything V2 integration, inference speed, caching, downloads,
workspace transfer, reconstruction pipeline, and scientific honesty guards.
"""

import os
import sys
import io
import time
import asyncio
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import UploadFile, HTTPException
from app import (
    app,
    health,
    predict_ai_depth,
    download_ai_depth,
    use_ai_depth_as_input,
    run_reconstruction,
    get_dataset_scenes,
    get_ai_depth_rag_insights,
    query_ai_depth_rag,
    AI_DEPTH_MGR,
    AI_PREDICTIONS
)
from src.ai_depth import calibrate_relative_to_metric

def run_tests():
    print("=" * 75)
    print("EDGEFUSE-DCR: Depth Anything V2 Monocular Depth Verification Suite")
    print("=" * 75)
    
    # Pre-load model
    print("\n[INIT] Ensuring Depth Anything V2 model checkpoint is loaded in memory...")
    t0 = time.time()
    loaded = AI_DEPTH_MGR.ensure_model_loaded()
    print(f"Model initialization complete in {(time.time() - t0):.2f}s (loaded={loaded})")
    assert loaded is True, "Failed to load Depth Anything V2 model!"

    # Test 1: Health Check
    print("\n[TEST 1] Probing health() endpoint...")
    h = health()
    print(f"Health Response: Status={h.get('status')}, AI Model={h.get('ai_depth_model')}, Available={h.get('ai_depth_available')}")
    assert h.get("status") == "healthy"
    assert h.get("ai_depth_available") is True
    assert "Depth Anything V2" in h.get("ai_depth_model")
    print("PASSED: Health check confirmed Depth Anything V2 is active and ready.")

    # Test 2: Indoor RGB Inference
    print("\n[TEST 2] Testing Indoor RGB Monocular Depth Estimation (NYU sample)...")
    indoor_path = os.path.join(ROOT_DIR, "demo_samples", "nyu_depth_v2", "raw_rgb", "nyu_scene_0001_rgb.png")
    if os.path.exists(indoor_path):
        with open(indoor_path, "rb") as f:
            indoor_bytes = f.read()
    else:
        img = Image.new("RGB", (640, 480), color=(180, 150, 130))
        b = io.BytesIO()
        img.save(b, format="PNG")
        indoor_bytes = b.getvalue()

    upload_indoor = UploadFile(filename="indoor_test.png", file=io.BytesIO(indoor_bytes))
    t0 = time.time()
    res_indoor = predict_ai_depth(rgb_file=upload_indoor, session_id="test_sess_indoor", input_size=518)
    t_indoor = (time.time() - t0) * 1000
    
    assert res_indoor.get("success") is True
    assert "relative" in res_indoor.get("depth_type", "").lower()
    assert "prediction_id" in res_indoor
    assert "preview_b64" in res_indoor
    indoor_id = res_indoor["prediction_id"]
    print(f"Indoor Prediction PASSED (latency: {res_indoor.get('inference_time_ms'):.1f} ms, total: {t_indoor:.1f} ms)")
    print(f"Prediction ID: {indoor_id}, Resolution: {res_indoor.get('width')}x{res_indoor.get('height')}")

    # Test 3: Outdoor RGB Inference
    print("\n[TEST 3] Testing Outdoor RGB Monocular Depth Estimation (KITTI sample)...")
    outdoor_path = os.path.join(ROOT_DIR, "demo_samples", "kitti_depth", "raw_rgb", "kitti_sync_000000_rgb.png")
    if os.path.exists(outdoor_path):
        with open(outdoor_path, "rb") as f:
            outdoor_bytes = f.read()
    else:
        img = Image.new("RGB", (1216, 352), color=(70, 100, 130))
        b = io.BytesIO()
        img.save(b, format="PNG")
        outdoor_bytes = b.getvalue()

    upload_outdoor = UploadFile(filename="outdoor_test.png", file=io.BytesIO(outdoor_bytes))
    res_outdoor = predict_ai_depth(rgb_file=upload_outdoor, session_id="test_sess_outdoor", input_size=518)
    assert res_outdoor.get("success") is True
    print(f"Outdoor Prediction PASSED (latency: {res_outdoor.get('inference_time_ms'):.1f} ms, res: {res_outdoor.get('width')}x{res_outdoor.get('height')})")

    # Test 4: Varied Resolutions and Aspect Ratios (Square & Portrait)
    print("\n[TEST 4] Testing varied aspect ratios (Square & Portrait)...")
    # Square
    sq_img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    sq_b = io.BytesIO()
    sq_img.save(sq_b, format="PNG")
    upload_sq = UploadFile(filename="square.png", file=io.BytesIO(sq_b.getvalue()))
    res_sq = predict_ai_depth(rgb_file=upload_sq, input_size=252)
    assert res_sq.get("success") is True
    assert res_sq["width"] == 256 and res_sq["height"] == 256

    # Portrait
    pt_img = Image.new("RGB", (180, 320), color=(120, 200, 120))
    pt_b = io.BytesIO()
    pt_img.save(pt_b, format="PNG")
    upload_pt = UploadFile(filename="portrait.png", file=io.BytesIO(pt_b.getvalue()))
    res_pt = predict_ai_depth(rgb_file=upload_pt, input_size=392)
    assert res_pt.get("success") is True
    assert res_pt["width"] == 180 and res_pt["height"] == 320
    print("PASSED: Square (256x256) and Portrait (180x320) aspect ratios handled correctly.")

    # Test 5: Cache Verification (SHA256 Match)
    print("\n[TEST 5] Testing SHA256 Cache Hit Performance...")
    upload_cached = UploadFile(filename="indoor_test_cached.png", file=io.BytesIO(indoor_bytes))
    t0 = time.time()
    res_cached = predict_ai_depth(rgb_file=upload_cached, input_size=518)
    t_cache = (time.time() - t0) * 1000
    assert res_cached.get("success") is True
    print(f"Cache Hit PASSED (reported inference_time_ms: {res_cached.get('inference_time_ms'):.2f} ms, execution time: {t_cache:.2f} ms)")
    assert t_cache < 1000.0, "Cache did not return near-instantaneous roundtrip!"
    assert res_cached.get("inference_time_ms", 999) <= 1.0, "Cache did not report zero inference time!"

    # Test 6: Download Formats (.npy and .png)
    print("\n[TEST 6] Testing Prediction Downloads (NPY and PNG)...")
    resp_npy = download_ai_depth(prediction_id=indoor_id, format="npy")
    assert resp_npy.media_type == "application/octet-stream"
    npy_bytes = resp_npy.body if hasattr(resp_npy, "body") else b"".join(list(resp_npy.body_iterator))
    loaded_npy = np.load(io.BytesIO(npy_bytes))
    assert loaded_npy.ndim == 2
    assert loaded_npy.dtype == np.float32
    print(f"NPY Download PASSED: Shape {loaded_npy.shape}, Dtype {loaded_npy.dtype}, Range [{loaded_npy.min():.4f}, {loaded_npy.max():.4f}]")

    resp_png = download_ai_depth(prediction_id=indoor_id, format="png")
    assert resp_png.media_type == "image/png"
    png_bytes = resp_png.body if hasattr(resp_png, "body") else b"".join(list(resp_png.body_iterator))
    loaded_png = Image.open(io.BytesIO(png_bytes))
    assert loaded_png.size == (loaded_npy.shape[1], loaded_npy.shape[0])
    print(f"PNG Download PASSED: Format {loaded_png.format}, Size {loaded_png.size}")

    # Test 7: Workspace Transfer & Confirmation Guards
    print("\n[TEST 7] Testing Workspace Transfer & Explicit Confirmation Guards...")
    # Without confirmation
    try:
        use_ai_depth_as_input(prediction_id=indoor_id, session_id="sess_unconf", confirmed=False)
        assert False, "Unconfirmed transfer should have raised HTTPException 400!"
    except HTTPException as e:
        assert e.status_code == 400
        print(f"Unconfirmed transfer rejected cleanly with HTTP {e.status_code}: {e.detail}")

    # With confirmation
    test_session = f"session_test_{int(time.time())}"
    res_transfer = use_ai_depth_as_input(prediction_id=indoor_id, session_id=test_session, confirmed=True)
    assert res_transfer.get("success") is True
    assert res_transfer.get("input_source") == "ai_predicted"
    assert res_transfer.get("depth_type") == "relative"
    print(f"Confirmed transfer SUCCESSFUL: Session {test_session}, Source={res_transfer.get('input_source')}")

    def sync_run(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value
        except Exception:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    # Test 8: EdgeFuse-DCR Reconstruction Pipeline across Algorithms
    print("\n[TEST 8] Testing EdgeFuse-DCR Reconstruction with AI-Predicted Depth...")
    algorithms = ["nearest", "bilinear", "bicubic", "joint_bilateral", "guided", "cnn", "hybrid"]
    for alg in algorithms:
        rec_res = sync_run(run_reconstruction(
            session_id=test_session,
            method=alg,
            upsample_factor=4,
            apply_texture_suppression=True,
            input_source="ai_predicted"
        ))
        assert rec_res.get("success") is True
        # Verify scientific honesty: No reference GT was provided
        assert rec_res.get("has_gt") is False, "has_gt must be False when reference GT is absent"
        assert rec_res.get("metrics") is None, "metrics must be None when reference GT is absent"
        assert rec_res.get("depth_type") == "relative"
        print(f"  - Method '{alg}': Reconstructed successfully ({rec_res.get('latency_ms'):.1f} ms, GT evaluation honestly omitted)", flush=True)

    # Test 9: Median Scale Calibration
    print("\n[TEST 9] Testing Median Scale Alignment Calibration...")
    rel_pred = np.linspace(0.1, 1.0, 100, dtype=np.float32).reshape(10, 10)
    sparse_sensor = np.zeros((10, 10), dtype=np.float32)
    sparse_sensor[2, 2] = 2.5
    sparse_sensor[4, 4] = 5.0
    sparse_sensor[6, 6] = 7.5
    calibrated, scale, shift = calibrate_relative_to_metric(rel_pred, sparse_sensor)
    assert scale > 0
    print(f"Median scale alignment: scale={scale:.4f}, shift={shift:.4f}, calibrated range=[{calibrated.min():.2f}, {calibrated.max():.2f}] m")

    # Test 10: Dataset Explorer Catalog Integrity Check
    print("\n[TEST 10] Verifying Existing Dataset Catalog (NYU, KITTI, Middlebury) Integrity...")
    gallery_data = get_dataset_scenes(dataset_id="all", page=1, limit=9)
    assert gallery_data["total"] > 0
    assert len(gallery_data["scenes"]) == 9
    print(f"PASSED: Catalog has {gallery_data['total']} paired scenes, pagination returned exactly 9 items.")

    # Test 11: Deep Learning Scene Analysis & CV RAG Literature Engine
    print("\n[TEST 11] Testing Deep Learning Feature Analysis & CV RAG Literature Engine...")
    rag_insights_res = get_ai_depth_rag_insights(prediction_id=indoor_id)
    assert rag_insights_res["success"] is True
    feat = rag_insights_res["features"]
    assert "scene_type" in feat
    assert "edge_alignment_fidelity" in feat
    assert "texture_copy_risk_index" in feat
    print(f"Scene Features PASSED: Category='{feat['scene_type']}', Edge Alignment={feat['edge_alignment_fidelity']:.3f}, Texture Risk={feat['texture_copy_risk_index']:.3f}")

    rag_query_res = query_ai_depth_rag(query="Why is monocular depth uncalibrated without physical metric sensors?", prediction_id=indoor_id)
    assert rag_query_res["success"] is True
    assert "citation" in rag_query_res
    assert len(rag_query_res["answer"]) > 50
    print(f"RAG Query PASSED: Citation='{rag_query_res['citation']}', Answer Length={len(rag_query_res['answer'])} chars")

    print("\n" + "=" * 75)
    print("ALL 11 VERIFICATION TESTS COMPLETED AND PASSED!")
    print("Depth Anything V2 monocular depth estimation, caching, downloads,")
    print("Deep Learning Geometric Feature Extraction, CV RAG Knowledge Base,")
    print("EdgeFuse-DCR reconstruction pipeline, and scientific guards are verified.")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
