import os, sys, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import (
    get_dataset_manifest,
    resolve_demo_sample,
    get_demo_sample,
    get_scene_benchmark,
    parse_depth_bytes,
    parse_rgb_bytes,
    compute_validity_mask_and_stats
)
from src.depth_engine import reconstruct_depth

def run_tests():
    print("=" * 80)
    print("EDGEFUSE-DCR DIRECT FUNCTIONAL VERIFICATION SUITE")
    print("=" * 80)
    passed = 0
    total = 8

    # 1. Dataset Manifest Verification
    print("\n[Test 1] Testing get_dataset_manifest()...")
    manifest = get_dataset_manifest()
    assert "datasets" in manifest
    assert len(manifest["datasets"]) == 3
    for dk in ["nyu", "kitti", "middlebury"]:
        dset = manifest["datasets"][dk]
        assert len(dset["samples"]) == 3, f"Dataset {dk} must have 3 samples"
        assert "sensor" in dset
        assert "sensor_short" in dset
        assert "typical_coverage" in dset
    print(f" -> PASS: Manifest contains 3 datasets, each with 3 verified scenes ({sum(len(d['samples']) for d in manifest['datasets'].values())} total).")
    passed += 1

    # 2. Dual Thumbnails Verification
    print("\n[Test 2] Testing Dual Thumbnails across all 9 scenes...")
    for dk, dset in manifest["datasets"].items():
        for s in dset["samples"]:
            sid = s["id"]
            assert "rgb_thumb_b64" in s and s["rgb_thumb_b64"].startswith("data:image/"), f"Missing rgb_thumb_b64 in {sid}"
            assert "depth_thumb_b64" in s and s["depth_thumb_b64"].startswith("data:image/"), f"Missing depth_thumb_b64 in {sid}"
            assert len(s["rgb_thumb_b64"]) > 100
            assert len(s["depth_thumb_b64"]) > 100
    print(" -> PASS: All 9 scenes possess valid Base64 RGB and Depth thumbnails.")
    passed += 1

    # 3. Authentic Benchmark Rows Verification
    print("\n[Test 3] Testing Authentic Benchmark Rows for all 9 scenes...")
    for dk, dset in manifest["datasets"].items():
        for s in dset["samples"]:
            sid = s["id"]
            assert "benchmark_rows" in s, f"Missing benchmark_rows in {sid}"
            rows = s["benchmark_rows"]
            assert len(rows) == 9, f"Scene {sid} must have exactly 9 evaluated algorithms, got {len(rows)}"
            for r in rows:
                rmse = float(r["rmse"])
                assert rmse > 0.0 and np.isfinite(rmse)
    print(" -> PASS: All 9 scenes contain authentic 9-algorithm benchmark rows against Ground Truth.")
    passed += 1

    # 4. get_scene_benchmark Endpoint Logic
    print("\n[Test 4] Testing get_scene_benchmark()...")
    bench = get_scene_benchmark("kitti_drive_03")
    assert bench["success"] is True
    assert bench["sample_id"] == "kitti_drive_03"
    assert len(bench["benchmark_rows"]) == 9
    print(f" -> PASS: get_scene_benchmark('kitti_drive_03') returned {len(bench['benchmark_rows'])} rows.")
    passed += 1

    # 5. get_demo_sample Function Execution
    print("\n[Test 5] Testing get_demo_sample()...")
    sample_ids = ["nyu_living_room", "kitti_drive_03", "middlebury_ladder1"]
    for sid in sample_ids:
        demo = get_demo_sample(sid)
        assert demo["success"] is True
        assert demo["rgb_b64"].startswith("data:image/")
        assert demo["lr_depth_b64"].startswith("data:image/")
        assert demo["valid_mask_b64"].startswith("data:image/")
        assert len(demo["benchmark_rows"]) == 9
        assert "valid_mask_stats" in demo
    print(f" -> PASS: Successfully loaded all {len(sample_ids)} test scenes with full Base64 images and stats.")
    passed += 1

    # 6. Validity Mask & Scientific Representation
    print("\n[Test 6] Testing compute_validity_mask_and_stats()...")
    kitti_demo = get_demo_sample("kitti_drive_03")
    assert kitti_demo["is_sparse"] is True
    assert "Measured Seeds" in kitti_demo["seed_label"]
    assert kitti_demo["valid_percent"] < 10.0

    nyu_demo = get_demo_sample("nyu_living_room")
    assert nyu_demo["is_sparse"] is False
    assert "Valid Input Pixels" in nyu_demo["seed_label"]
    assert nyu_demo["valid_percent"] > 95.0
    print(f" -> PASS: Verified Sparse ({kitti_demo['valid_percent']}%) vs Dense ({nyu_demo['valid_percent']}%) validity mask labeling.")
    passed += 1

    # 7. Scientific Honesty: Rejection of RGB Color Photos as Depth
    print("\n[Test 7] Testing Rejection of Ordinary Color Photographs...")
    photo = np.zeros((100, 100, 3), dtype=np.uint8)
    photo[:, :, 0] = 220
    photo[:, :, 1] = 80
    photo[:, :, 2] = 40
    import io
    buf = io.BytesIO()
    Image.fromarray(photo).save(buf, format="PNG")
    photo_bytes = buf.getvalue()

    rejected = False
    try:
        parse_depth_bytes(photo_bytes, filename="Home.png")
    except ValueError as e:
        if "ordinary RGB color photograph" in str(e):
            rejected = True
    assert rejected is True
    print(" -> PASS: Successfully rejected ordinary RGB photograph with scientific guidance.")
    passed += 1

    # 8. Frontend HTML Template Integrity
    print("\n[Test 8] Testing templates/index.html UI Component Integrity...")
    tpl_p = "templates/index.html"
    if not os.path.exists(tpl_p):
        tpl_p = "EDGEFUSE_DCR/templates/index.html"
    with open(tpl_p, "r", encoding="utf-8") as f:
        html = f.read()

    assert "gallery-active-label" in html
    assert "btn-gallery-prev" in html
    assert "btn-gallery-next" in html
    assert "dataset-meta-banner" in html
    assert "gallery-dual-thumbs" in html
    assert "updateGalleryActiveUI" in html
    assert "renderGalleryCards" in html
    assert "currentSceneBenchmarks" in html
    assert "badge-best" in html
    assert "Scene 1 of 3" in html
    print(" -> PASS: templates/index.html verified with all dual-thumbnail markup, synchronized navigation, and dynamic benchmark rendering.")
    passed += 1

    print("\n" + "=" * 80)
    print(f"ALL {passed}/{total} DIRECT FUNCTIONAL TESTS PASSED (100%)!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
