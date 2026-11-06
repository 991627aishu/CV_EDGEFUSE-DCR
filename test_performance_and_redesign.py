import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import (
    get_dataset_manifest,
    get_demo_sample,
    run_reconstruction,
    encode_image_base64
)

def run_performance_audit():
    print("=" * 80)
    print("EDGEFUSE-DCR ULTIMATE PERFORMANCE & REDESIGN VERIFICATION")
    print("=" * 80)

    # 1. Image Encoding Compression & Speed Benchmark
    print("\n[Benchmark 1] Image Encoding Throughput & Payload Reduction...")
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test auto (JPEG quality 88)
    t0 = time.perf_counter()
    b64_auto = encode_image_base64(test_img, format="auto", quality=88)
    t_auto = (time.perf_counter() - t0) * 1000.0

    # Test legacy PNG
    t0 = time.perf_counter()
    b64_png = encode_image_base64(test_img, format="PNG")
    t_png = (time.perf_counter() - t0) * 1000.0

    reduction = (1.0 - len(b64_auto) / max(1, len(b64_png))) * 100.0
    speedup = t_png / max(0.001, t_auto)
    print(f" -> Legacy PNG: {len(b64_png)/1024:.1f} KB in {t_png:.1f} ms")
    print(f" -> Optimized JPEG (88%): {len(b64_auto)/1024:.1f} KB in {t_auto:.1f} ms")
    print(f" -> PayLoad Reduction: {reduction:.1f}% smaller, {speedup:.1f}× faster encoding!")
    assert reduction > 50.0, "JPEG payload should be at least 50% smaller than PNG"

    # 2. Demo Sample Caching Benchmark
    print("\n[Benchmark 2] In-Memory Demo Sample Caching...")
    # First call: cold
    t0 = time.perf_counter()
    demo1 = get_demo_sample("nyu_living_room")
    t_cold = (time.perf_counter() - t0) * 1000.0

    # Second call: warm cache
    t0 = time.perf_counter()
    demo2 = get_demo_sample("nyu_living_room")
    t_warm = (time.perf_counter() - t0) * 1000.0

    print(f" -> Cold Sample Load: {t_cold:.2f} ms")
    print(f" -> Warm Cached Load: {t_warm:.3f} ms ({t_cold / max(0.0001, t_warm):.1f}× faster)")
    assert demo1["sample_id"] == demo2["sample_id"]
    assert t_warm < 5.0, "Warm cached sample should return in under 5ms"

    # 3. HTML Payload Audit
    print("\n[Benchmark 3] HTML Template Payload & Integrity Audit...")
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    size_kb = len(html.encode("utf-8")) / 1024.0
    print(f" -> Template size: {size_kb:.1f} KB")
    
    # Assert key architectural components
    checks = [
      ("Editorial Canvas", "Reconstructing the Missing"),
      ("Split Comparison Slider", "hero-slider-container"),
      ("Why Depth Matters Storytelling", "Why Depth Completion Matters"),
      ("8-Step Workflow", "How to Use EDGEFUSE-DCR"),
      ("Dataset Explorer Section", "section-explorer"),
      ("Scene Viewer Lightbox", "scene-viewer-modal"),
      ("Reconstruction Workspace", "section-reconstruction"),
      ("KPI Metrics", "kpi-metric-bar"),
      ("Comparative Benchmarks", "section-benchmark"),
      ("6-Card Research Exports", "section-exports"),
      ("Guided Spotlight Tour", "startGuidedTour"),
      ("Dual Thumbnails Class", "gallery-dual-thumbs"),
    ]

    for label, substr in checks:
        assert substr in html, f"Missing critical component: {label} ({substr})"
        print(f" -> PASS: {label} verified in DOM template.")

    print("\n" + "=" * 80)
    print("ALL PERFORMANCE & ARCHITECTURAL BENCHMARKS PASSED (100%)!")
    print("=" * 80)

if __name__ == "__main__":
    run_performance_audit()
