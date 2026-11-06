import sys
import os
import time

# Ensure repo root is on python path
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("=" * 70)
print("EDGEFUSE-DCR: UNLIMITED DATASET EXPLORER INTEGRATION TEST SUITE")
print("=" * 70)

# 1. Test Dataset Discovery Catalog
print("\n[Test 1/5] Testing Dataset Discovery Engine...")
from src.dataset_discovery import CATALOG

total_scenes = len(CATALOG.scenes)
print(f"Total Discovered Scenes: {total_scenes}")
assert total_scenes >= 2470, f"Expected at least 2,470 scenes, got {total_scenes}"

audit = CATALOG.inventory_audit
print(f"Inventory Audit Keys: {list(audit.keys())}")
assert "nyu" in audit and "kitti" in audit and "middlebury" in audit
print(f"NYU Complete Pairs: {audit['nyu']['complete_paired_scenes']}")
print(f"KITTI Complete Pairs: {audit['kitti']['complete_paired_scenes']}")
print(f"Middlebury Complete Pairs: {audit['middlebury']['complete_paired_scenes']}")

# 2. Test Pagination Queries
print("\n[Test 2/5] Testing Server-Side Pagination...")
# Test page 1 of all datasets (limit 12)
p1 = CATALOG.get_paginated_scenes("all", page=1, limit=12)
assert len(p1["scenes"]) == 12, f"Expected 12 scenes, got {len(p1['scenes'])}"
assert p1["total_scenes"] == total_scenes
print(f"Page 1 scenes returned: {len(p1['scenes'])}, Total: {p1['total_scenes']}, Total Pages: {p1['total_pages']}")

# Test page 2
p2 = CATALOG.get_paginated_scenes("all", page=2, limit=12)
assert len(p2["scenes"]) == 12
assert p1["scenes"][0]["id"] != p2["scenes"][0]["id"], "Page 1 and Page 2 should not start with same item"

# Test NYU specific pagination
p_nyu = CATALOG.get_paginated_scenes("nyu", page=1, limit=24)
assert len(p_nyu["scenes"]) == 24
assert all(s["dataset_key"] == "nyu" for s in p_nyu["scenes"])
print(f"NYU Page 1: {len(p_nyu['scenes'])} scenes, Total: {p_nyu['total_scenes']}")

# Test KITTI specific pagination
p_kitti = CATALOG.get_paginated_scenes("kitti", page=1, limit=24)
assert len(p_kitti["scenes"]) == 24
assert all(s["dataset_key"] == "kitti" for s in p_kitti["scenes"])
print(f"KITTI Page 1: {len(p_kitti['scenes'])} scenes, Total: {p_kitti['total_scenes']}")

# Test Middlebury specific pagination
p_mid = CATALOG.get_paginated_scenes("middlebury", page=1, limit=24)
assert len(p_mid["scenes"]) == 24
assert all(s["dataset_key"] == "middlebury" for s in p_mid["scenes"])
print(f"Middlebury Page 1: {len(p_mid['scenes'])} scenes, Total: {p_mid['total_scenes']}")

# 3. Test Search & Filters
print("\n[Test 3/5] Testing Search & Filter Functionality...")
search_res = CATALOG.get_paginated_scenes("all", page=1, limit=10, search="living")
assert len(search_res["scenes"]) > 0
print(f"Search 'living': Found {search_res['total_scenes']} matching scenes")

res_filtered = CATALOG.get_paginated_scenes("all", page=1, limit=10, filter_resolution="640")
assert len(res_filtered["scenes"]) > 0
assert all("640" in s["dimensions"] for s in res_filtered["scenes"])
print(f"Filter '640' resolution: Found {res_filtered['total_scenes']} matching scenes")

# 4. Test Arbitrary Scene Data Loading
print("\n[Test 4/5] Testing Raw Data Loading for Discovered Scenes...")
sample_ids_to_test = [
    "nyu_living_room",
    "nyu_idx_0050",
    "kitti_road_01",
    "kitti_2011_09_26_drive_0002_sync_image_0000000005_image_02",
    "middlebury_artroom1",
    "middlebury_chess1"
]

for sid in sample_ids_to_test:
    t0 = time.time()
    rgb, lr_depth, gt_depth, info = CATALOG.load_scene_data(sid)
    dur = round((time.time() - t0) * 1000, 1)
    assert rgb is not None, f"Failed RGB load for {sid}"
    assert lr_depth is not None, f"Failed Depth load for {sid}"
    gt_str = f"GT: {gt_depth.shape}" if gt_depth is not None else "No GT"
    print(f"  ✓ Loaded '{sid}' ({info['title']}) in {dur} ms | RGB: {rgb.shape} | Depth: {lr_depth.shape} | {gt_str}")

# 5. Test Full Reconstruction on Arbitrary Discovered Scene
print("\n[Test 5/5] Testing Reconstruction Pipeline on Arbitrary Scene...")
from src.depth_engine import reconstruct_depth
from src.evaluation import compute_depth_metrics
from app import MODEL, DEVICE

rgb, lr_depth, gt_depth, info = CATALOG.load_scene_data("nyu_idx_0050")
t0 = time.time()
out = reconstruct_depth(
    rgb_image=rgb,
    depth_input=lr_depth,
    factor=4,
    method="hybrid",
    cnn_model=MODEL,
    device=DEVICE,
    apply_suppression=True
)
dur = round(time.time() - t0, 3)
depth_recon = out["reconstructed_depth"]
assert depth_recon is not None
metrics = compute_depth_metrics(depth_recon, gt_depth, max_depth=10.0)
assert metrics["rmse"] is not None
print(f"  ✓ Arbitrary Scene 'nyu_idx_0050' reconstructed in {dur}s | RMSE: {metrics['rmse']} | PSNR: {metrics['psnr']} dB | SSIM: {metrics['ssim']}")

print("\n" + "=" * 70)
print("ALL TESTS PASSED PERFECTLY! UNLIMITED DATASET EXPLORER READY FOR DEPLOYMENT.")
print("=" * 70)
