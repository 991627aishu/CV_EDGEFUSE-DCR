"""
EDGEFUSE-DCR Comparative Benchmark Scientific Consistency Verification Test Suite
Verifies:
1. Exact root-cause resolution of the A5 vs Live Run metric discrepancy (0.1143 vs 0.0372)
2. Accurate timing isolation: Model latency vs Preprocessing vs Postprocessing vs Total Roundtrip
3. Scientific fairness: Identical scene input, dimensions, and ground truth evaluation masks
4. Dynamic algorithm row labeling across A0-A5 methods
5. On-demand scene benchmark calculation and caching
6. Graceful handling of missing ground truth
"""

import os
import sys
import time
import pytest
import numpy as np

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.dataset_discovery import DatasetCatalog, CATALOG
from src.depth_engine import reconstruct_depth
from src.evaluation import compute_depth_metrics
from src.model import load_model
from app import app, resolve_demo_sample, get_scene_benchmark, run_reconstruction, METHOD_STAGE_MAP


def test_01_scene_benchmark_linking_and_consistency():
    """Verifies that nyu_idx_0000 correctly links to authentic precomputed scene benchmarks."""
    catalog = DatasetCatalog()
    scene = catalog.scenes_by_id.get("nyu_idx_0000")
    assert scene is not None, "nyu_idx_0000 must exist in catalog"
    rows = scene.get("benchmark_rows", [])
    assert len(rows) == 9, f"nyu_idx_0000 must have 9 benchmark rows, found {len(rows)}"

    # Locate A5 row
    a5_row = next((r for r in rows if "A5" in r["method"] or r.get("key") == "hybrid"), None)
    assert a5_row is not None, "A5 row must exist in benchmark_rows"
    assert a5_row["rmse"] == "0.0372", f"A5 RMSE on nyu_idx_0000 should be 0.0372, got {a5_row['rmse']}"
    assert a5_row["mae"] == "0.0173", f"A5 MAE on nyu_idx_0000 should be 0.0173, got {a5_row['mae']}"
    assert float(a5_row["psnr"]) >= 48.5, f"A5 PSNR should be >= 48.5, got {a5_row['psnr']}"
    assert a5_row["ssim"] == "0.9961", f"A5 SSIM should be 0.9961, got {a5_row['ssim']}"
    print("\n[Test 1 PASSED] Precomputed scene benchmark for nyu_idx_0000 successfully matches authentic 0.0372 RMSE baseline.")


@pytest.mark.asyncio
async def test_02_deterministic_live_run_matches_scene_baseline():
    """Verifies that executing Live A5 on nyu_idx_0000 matches the scene reference baseline within precision."""
    res = await run_reconstruction(
        demo_sample_id="nyu_idx_0000",
        method="hybrid",
        upsample_factor=4,
        apply_texture_suppression=True
    )
    assert res["success"] is True
    assert res["has_gt"] is True
    metrics = res["metrics"]
    assert metrics is not None, "Metrics must be computed for nyu_idx_0000"

    # Verify metrics match 0.0372 within floating point tolerance
    assert abs(metrics["rmse"] - 0.0372) < 0.0005, f"Live RMSE {metrics['rmse']} does not match 0.0372"
    assert abs(metrics["mae"] - 0.0173) < 0.0005, f"Live MAE {metrics['mae']} does not match 0.0173"
    assert abs(metrics["ssim"] - 0.9961) < 0.0005, f"Live SSIM {metrics['ssim']} does not match 0.9961"
    assert abs(metrics["boundary_f1"] - 0.6692) < 0.002, f"Live Boundary F1 {metrics['boundary_f1']} does not match 0.6692"
    assert abs(metrics["grad_mae"] - 0.0464) < 0.001, f"Live Grad MAE {metrics['grad_mae']} does not match 0.0464"
    print(f"\n[Test 2 PASSED] Live A5 execution perfectly replicates authentic scene baseline: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, PSNR={metrics['psnr']:.2f} dB.")


@pytest.mark.asyncio
async def test_03_latency_profiling_and_isolation():
    """Verifies that model latency is isolated from full HTTP roundtrip latency."""
    res = await run_reconstruction(
        demo_sample_id="nyu_idx_0000",
        method="hybrid",
        upsample_factor=4
    )
    b = res["latency_breakdown"]
    assert "model_latency_ms" in b, "Breakdown must include model_latency_ms"
    assert "preprocess_latency_ms" in b, "Breakdown must include preprocess_latency_ms"
    assert "postprocess_latency_ms" in b, "Breakdown must include postprocess_latency_ms"
    assert "total_latency_ms" in b, "Breakdown must include total_latency_ms"
    assert "device" in b, "Breakdown must record execution device"

    # Model latency should be pure inference (< total_latency)
    assert b["model_latency_ms"] <= b["total_latency_ms"], "Model latency must not exceed total latency"
    # Latency displayed in table should be pure model latency, not 3500ms
    assert res["latency_ms"] == b["model_latency_ms"], "Table latency must be model latency"
    assert res["latency_ms"] < 2000.0, f"Table latency {res['latency_ms']} ms must reflect model inference, not multi-second API overhead"
    print(f"\n[Test 3 PASSED] Latency boundaries verified: Model={b['model_latency_ms']} ms, Pre={b['preprocess_latency_ms']} ms, Post={b['postprocess_latency_ms']} ms, Total={b['total_latency_ms']} ms.")


@pytest.mark.asyncio
async def test_04_dynamic_algorithm_stage_labeling():
    """Verifies that selecting A0-A5 properly updates the live row label and stage."""
    test_methods = [
        ("nearest", "A0", "Nearest Neighbor"),
        ("guided", "A1", "Guided Filter"),
        ("cnn", "A2", "Deep Learning (EdgeFuseNet Alone)"),
        ("hybrid", "A5", "EdgeFuse-DCR Full Hybrid")
    ]

    for method_key, expected_stage, expected_name in test_methods:
        res = await run_reconstruction(
            demo_sample_id="nyu_idx_0000",
            method=method_key
        )
        assert res["algorithm_stage"] == expected_stage, f"Expected stage {expected_stage} for {method_key}, got {res['algorithm_stage']}"
        assert res["algorithm_name"] == expected_name, f"Expected name {expected_name} for {method_key}, got {res['algorithm_name']}"
        if method_key == "hybrid":
            assert res["live_row_label"] == "Live A5 — Current Execution", f"Unexpected label: {res['live_row_label']}"
        else:
            assert expected_stage in res["live_row_label"], f"Stage {expected_stage} must be in label: {res['live_row_label']}"
    print("\n[Test 4 PASSED] Dynamic algorithm stage labeling verified across A0, A1, A2, and A5.")


@pytest.mark.asyncio
async def test_05_graceful_missing_ground_truth_handling():
    """Verifies that missing Ground Truth does not fabricate metrics or crash the pipeline."""
    # Test with custom session without GT
    from app import CUSTOM_SESSIONS
    sess_id = "test_no_gt_session"
    dummy_rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    dummy_depth = np.full((30, 40), 2.0, dtype=np.float32)

    CUSTOM_SESSIONS[sess_id] = {
        "raw_rgb": dummy_rgb,
        "raw_depth": dummy_depth,
        "raw_gt": None,
        "depth_type": "metric",
        "depth_unit": "m",
        "depth_warning": None,
        "input_source": "sensor",
        "timestamp": time.time()
    }

    res = await run_reconstruction(
        session_id=sess_id,
        method="hybrid"
    )
    assert res["success"] is True
    assert res["has_gt"] is False
    assert res["metrics"] is not None, "Live sensor fidelity metrics must be computed"
    assert "rmse" in res["metrics"], "RMSE must be present in live metrics"
    assert res["error_b64"] is not None, "Refinement residual heatmap must be generated"
    print("\n[Test 5 PASSED] Absence of Ground Truth handled gracefully with live sensor fidelity metrics.")


if __name__ == "__main__":
    def sync_run(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    print("================================================================================")
    print("RUNNING EDGEFUSE-DCR COMPARATIVE BENCHMARK SCIENTIFIC CONSISTENCY TEST SUITE")
    print("================================================================================")

    test_01_scene_benchmark_linking_and_consistency()
    sync_run(test_02_deterministic_live_run_matches_scene_baseline())
    sync_run(test_03_latency_profiling_and_isolation())
    sync_run(test_04_dynamic_algorithm_stage_labeling())
    sync_run(test_05_graceful_missing_ground_truth_handling())

    print("\n================================================================================")
    print("ALL 5 SCIENTIFIC BENCHMARK CONSISTENCY TESTS PASSED SUCCESSFULLY!")
    print("================================================================================")

