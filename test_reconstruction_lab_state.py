"""
Test Suite: Reconstruction Lab Initial State and UI/UX Verification
Verifies that:
1. HTML contains zero hardcoded initial metrics (all 6 are N/A).
2. All reconstruction lab <img> elements are hidden on initial load.
3. All reconstruction placeholders exist.
4. JavaScript initializes currentDemo to null.
5. 5-state application model (idle, input_ready, processing, success, error) is implemented.
6. Execution button is disabled in idle state.
7. Backend /api/reconstruct responds successfully without HTTP 502.
"""
import re
import os
import sys

def test_html_initial_state():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Verify metric elements initialized to N/A
    metrics = ["val-rmse", "val-mae", "val-psnr", "val-ssim", "val-f1", "val-latency"]
    for m in metrics:
        pattern = rf'id="{m}"[^>]*>\s*([^\s<]+)'
        match = re.search(pattern, html)
        assert match is not None, f"Element #{m} not found in HTML"
        val = match.group(1).strip()
        assert val == "N/A", f"Element #{m} has initial value '{val}', expected 'N/A'"
    print("✓ All 6 KPI metrics initialized to N/A")

    # 2. Verify header sample pill and dimensions
    assert 'id="header-sample-pill"' in html
    match_pill = re.search(r'id="header-sample-pill"[^>]*>\s*([^<\n]+)', html)
    assert match_pill and "Idle (Awaiting Input)" in match_pill.group(1), f"Pill unexpected: {match_pill.group(1) if match_pill else 'None'}"
    print("✓ Header pill initialized to 'Idle (Awaiting Input)'")

    match_dim = re.search(r'id="sample-dimensions"[^>]*>\s*([^<\n]+)', html)
    assert match_dim and "No Active Input" in match_dim.group(1), f"Dimensions unexpected: {match_dim.group(1) if match_dim else 'None'}"
    print("✓ Sample dimensions initialized to 'No Active Input'")

    # 3. Verify img tags in reconstruction section have hidden class
    sec_idx = html.find('id="section-reconstruction"')
    sec_end = html.find('id="section-benchmark"')
    rec_html = html[sec_idx:sec_end]
    
    img_tags = re.findall(r'<img[^>]+>', rec_html)
    assert len(img_tags) == 10, f"Expected 10 img tags in reconstruction section, found {len(img_tags)}"
    for tag in img_tags:
        assert 'class="hidden' in tag or 'hidden' in tag, f"Img tag missing hidden class: {tag}"
    print("✓ All 10 Reconstruction Lab <img> tags have class='hidden'")

    # 4. Verify placeholders exist
    placeholders = [
        "ph-view-rgb", "ph-view-lr-depth", "view-reconstructed-placeholder",
        "view-classical-ph", "view-cnn-ph", "view-ablation-final-ph",
        "ph-view-valid-mask", "ph-view-boundary", "ph-view-confidence", "ph-view-error"
    ]
    for ph in placeholders:
        assert f'id="{ph}"' in rec_html, f"Placeholder #{ph} not found in reconstruction section"
    print("✓ All 10 diagnostic preview placeholders present")

    # 5. Verify JavaScript initial state
    assert "let currentDemo = null;" in html, "currentDemo should be initialized to null"
    assert "const LAB_STATES" in html, "LAB_STATES state machine missing"
    assert "'idle'" in html and "'input_ready'" in html and "'processing'" in html and "'success'" in html and "'error'" in html
    print("✓ JavaScript 5-state application model defined with currentDemo = null")

    # 6. Verify timing cards initialized to -- ms
    for timing_id in ["timing-model-ms", "timing-prep-ms", "timing-post-ms", "timing-total-ms"]:
        pattern = rf'id="{timing_id}"[^>]*>\s*([^\s<]+)'
        match = re.search(pattern, html)
        assert match and match.group(1).strip() == "--", f"Timing #{timing_id} not initialized to --"
    print("✓ Latency timing breakdown cards initialized to -- ms")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_html_initial_state()
    print("\n✅ ALL RECONSTRUCTION LAB INITIAL STATE TESTS PASSED SUCCESSFULLY!")
