"""
Dedicated Test Suite for EDGEFUSE-DCR
Covers:
  - Test Group A: Image Rendering (Hero comparison, thumbnails, endpoints, content types)
  - Test Group B: Flexible Pagination (All 10 options: 3, 6, 9, 12, 15, 18, 24, 30, 48, 96; formula, filter resets)
  - Test Group C: Dataset Image Uniqueness (KITTI, NYU, Middlebury hashes, distinct scene imagery)
"""

import os
import sys
import json
import math
import hashlib
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import get_hero_comparison, get_dataset_scenes, get_dataset_integrity, get_demo_sample, CATALOG

def test_group_a_image_rendering():
    print("\n" + "=" * 70)
    print("TEST GROUP A: Image Rendering & Static Asset Validation")
    print("=" * 70)

    # 1. Hero comparison endpoint
    hero = get_hero_comparison()
    assert "input_image_url" in hero
    assert "refined_image_url" in hero
    assert "rmse" in hero
    print(f" -> PASS: Hero comparison metadata verified (RMSE: {hero['rmse']})")

    # 2. Both hero images load and have correct dimensions
    input_file = hero["input_image_url"].lstrip("/")
    refined_file = hero["refined_image_url"].lstrip("/")
    assert os.path.exists(input_file), f"Missing input hero image: {input_file}"
    assert os.path.exists(refined_file), f"Missing refined hero image: {refined_file}"

    img_in = Image.open(input_file)
    img_ref = Image.open(refined_file)
    print(f" -> PASS: Hero raw input depth image loaded successfully ({img_in.size[0]}×{img_in.size[1]})")
    print(f" -> PASS: Hero refined depth image loaded successfully ({img_ref.size[0]}×{img_ref.size[1]})")

    # 3. Compatible dimensions assertion
    assert img_in.size == img_ref.size, f"Hero comparison image dimensions mismatch: {img_in.size} vs {img_ref.size}"
    assert img_in.size == (960, 540), f"Expected 960x540, got {img_in.size}"
    print(f" -> PASS: Both hero comparison images have matching 16:9 aspect ratio and dimensions ({img_in.size})")

    # 4. Dataset thumbnails check across all 3 datasets
    test_scenes = [
        "middlebury_artroom1", 
        "kitti_road_01", 
        "nyu_living_room", 
        "kitti_2011_09_26_drive_0002_sync_image_0000000005_image_02", 
        "nyu_idx_0001",
        "mid_chess1"
    ]
    for sid in test_scenes:
        assert sid in CATALOG.scenes_by_id, f"Scene {sid} missing from catalog"
        sc = CATALOG.scenes_by_id[sid]
        rgb_url = sc.get("rgb_thumb_url")
        depth_url = sc.get("depth_thumb_url")
        assert rgb_url, f"Missing rgb_thumb_url for {sid}"
        assert depth_url, f"Missing depth_thumb_url for {sid}"
        
        rgb_p = rgb_url.lstrip("/")
        depth_p = depth_url.lstrip("/")
        assert os.path.exists(rgb_p), f"Thumbnail file not found: {rgb_p}"
        assert os.path.exists(depth_p), f"Thumbnail file not found: {depth_p}"
        
        # Verify valid image decoding
        im_rgb = Image.open(rgb_p)
        im_d = Image.open(depth_p)
        assert im_rgb.size[0] > 0 and im_d.size[0] > 0

    print(f" -> PASS: Dataset thumbnails decoded and verified across NYU, KITTI, and Middlebury sample scenes.")


def test_group_b_pagination():
    print("\n" + "=" * 70)
    print("TEST GROUP B: Flexible Dataset Pagination (All 10 Page Sizes)")
    print("=" * 70)

    page_sizes = [3, 6, 9, 12, 15, 18, 24, 30, 48, 96]
    total_scenes = len(CATALOG.scenes)
    assert total_scenes >= 2479, f"Expected >=2479 scenes in catalog, found {total_scenes}"

    for size in page_sizes:
        data = get_dataset_scenes("all", page=1, limit=size)
        rendered_count = len(data["scenes"])
        assert rendered_count == size, f"Expected {size} scenes for limit={size}, got {rendered_count}"
        
        expected_total_pages = math.ceil(total_scenes / size)
        assert data["total_pages"] == expected_total_pages, f"Expected {expected_total_pages} total pages, got {data['total_pages']}"
        print(f" -> PASS: Limit = {size:2d} | Rendered exactly {rendered_count:2d} cards | Total Pages: {data['total_pages']:4d}")

    # Test pagination navigation (Page 2, 3, last page)
    data_p2 = get_dataset_scenes("all", page=2, limit=9)
    assert data_p2["page"] == 2
    assert len(data_p2["scenes"]) == 9
    assert data_p2["has_prev"] is True
    assert data_p2["has_next"] is True
    print(" -> PASS: Page 2 navigation with limit=9 verified.")

    # Test dataset filter integration + page reset
    data_kitti = get_dataset_scenes("kitti", page=1, limit=9)
    assert len(data_kitti["scenes"]) == 9
    assert all(s["dataset_key"] == "kitti" for s in data_kitti["scenes"])
    kitti_total = data_kitti["total_scenes"]
    assert kitti_total >= 1000, f"Expected at least 1000 KITTI scenes, got {kitti_total}"
    print(f" -> PASS: KITTI filter synchronization: 9 scenes rendered out of {kitti_total} total matching scenes.")

    # Test NYU filter
    data_nyu = get_dataset_scenes("nyu", page=1, limit=9)
    assert len(data_nyu["scenes"]) == 9
    assert all(s["dataset_key"] == "nyu" for s in data_nyu["scenes"])
    print(f" -> PASS: NYU filter synchronization: 9 scenes rendered out of {data_nyu['total_scenes']} total matching scenes.")

    # Test Middlebury filter
    data_mid = get_dataset_scenes("middlebury", page=1, limit=9)
    assert len(data_mid["scenes"]) == 9
    assert all(s["dataset_key"] == "middlebury" for s in data_mid["scenes"])
    print(f" -> PASS: Middlebury filter synchronization: 9 scenes rendered out of {data_mid['total_scenes']} total matching scenes.")

    # Test search filter synchronization
    data_search = get_dataset_scenes("all", page=1, limit=9, search="artroom")
    assert data_search["total_scenes"] >= 1
    assert any("artroom" in s["title"].lower() or "artroom" in s["id"].lower() for s in data_search["scenes"])
    print(f" -> PASS: Search filter synchronization: {data_search['total_scenes']} matching scenes for 'artroom'.")


def test_group_c_dataset_image_uniqueness():
    print("\n" + "=" * 70)
    print("TEST GROUP C: Dataset Image Integrity & Uniqueness Audit")
    print("=" * 70)

    # 1. Check dataset_image_integrity_report.json
    rep = get_dataset_integrity()
    print(f" -> Integrity Report: {rep['total_indexed_scenes']} indexed scenes audited.")
    print(f" -> Unique Depth Thumbnails: {rep['authenticity']['unique_depth_thumbnails']} ({rep['authenticity']['percentage_unique_depth']}%)")
    print(f" -> Unique RGB Thumbnails: {rep['authenticity']['unique_rgb_thumbnails']} ({rep['authenticity']['percentage_unique_rgb']}%)")
    assert rep['authenticity']['percentage_unique_depth'] >= 99.0
    assert rep['authenticity']['percentage_unique_rgb'] >= 99.0

    # 2. Inspect 10 different KITTI frames
    kitti_scenes = [s for s in CATALOG.scenes if s["dataset_key"] == "kitti" and not s.get("is_demo", False)]
    assert len(kitti_scenes) >= 10
    kitti_sample_10 = kitti_scenes[:10]
    kitti_rgb_hashes = set()
    kitti_depth_hashes = set()
    for s in kitti_sample_10:
        rgb_url = s["rgb_thumb_url"]
        depth_url = s["depth_thumb_url"]
        assert rgb_url and depth_url
        rgb_p = rgb_url.lstrip("/")
        depth_p = depth_url.lstrip("/")
        assert os.path.exists(rgb_p), f"Missing {rgb_p}"
        assert os.path.exists(depth_p), f"Missing {depth_p}"
        kitti_rgb_hashes.add(hashlib.md5(open(rgb_p, "rb").read()).hexdigest())
        kitti_depth_hashes.add(hashlib.md5(open(depth_p, "rb").read()).hexdigest())
    print(f" -> PASS: 10 inspected non-demo KITTI frames have {len(kitti_depth_hashes)} distinct depth map hashes.")
    assert len(kitti_depth_hashes) == len(kitti_sample_10), "Detected duplicate depth thumbnails across KITTI frames!"

    # 3. Inspect 10 different NYU scenes
    nyu_scenes = [s for s in CATALOG.scenes if s["dataset_key"] == "nyu" and not s.get("is_demo", False)]
    assert len(nyu_scenes) >= 10
    nyu_sample_10 = nyu_scenes[:10]
    nyu_rgb_hashes = set()
    nyu_depth_hashes = set()
    for s in nyu_sample_10:
        rgb_p = s["rgb_thumb_url"].lstrip("/")
        depth_p = s["depth_thumb_url"].lstrip("/")
        assert os.path.exists(rgb_p), f"Missing {rgb_p}"
        assert os.path.exists(depth_p), f"Missing {depth_p}"
        nyu_rgb_hashes.add(hashlib.md5(open(rgb_p, "rb").read()).hexdigest())
        nyu_depth_hashes.add(hashlib.md5(open(depth_p, "rb").read()).hexdigest())
    print(f" -> PASS: 10 inspected non-demo NYU scenes have {len(nyu_rgb_hashes)} distinct RGB hashes and {len(nyu_depth_hashes)} distinct depth hashes.")
    assert len(nyu_rgb_hashes) == len(nyu_sample_10)

    # 4. Inspect Middlebury scenes
    mid_scenes = [s for s in CATALOG.scenes if s["dataset_key"] == "middlebury" and not s.get("is_demo", False)]
    assert len(mid_scenes) >= 20
    mid_rgb_hashes = set()
    mid_depth_hashes = set()
    for s in mid_scenes:
        rgb_p = s["rgb_thumb_url"].lstrip("/")
        depth_p = s["depth_thumb_url"].lstrip("/")
        assert os.path.exists(rgb_p), f"Missing {rgb_p}"
        assert os.path.exists(depth_p), f"Missing {depth_p}"
        mid_rgb_hashes.add(hashlib.md5(open(rgb_p, "rb").read()).hexdigest())
        mid_depth_hashes.add(hashlib.md5(open(depth_p, "rb").read()).hexdigest())
    print(f" -> PASS: All {len(mid_scenes)} Middlebury scenes have {len(mid_rgb_hashes)} distinct RGB hashes and {len(mid_depth_hashes)} distinct depth hashes.")
    assert len(mid_rgb_hashes) == len(mid_scenes)
    assert len(mid_depth_hashes) == len(mid_scenes)

    print("\n" + "=" * 70)
    print("ALL TEST GROUPS (A, B, C) PASSED WITH ZERO FAILURES!")
    print("=" * 70)

if __name__ == "__main__":
    test_group_a_image_rendering()
    test_group_b_pagination()
    test_group_c_dataset_image_uniqueness()
