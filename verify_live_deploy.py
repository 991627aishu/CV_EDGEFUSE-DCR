import urllib.request
import re

url = "https://cv-edgefuse-dcr.onrender.com/"
print(f"Fetching {url} ...")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    html = resp.read().decode("utf-8")

print(f"HTTP Status: {resp.status} (Length: {len(html)} bytes)")

# 1. Check metric values
metrics = ["val-rmse", "val-mae", "val-psnr", "val-ssim", "val-f1", "val-latency"]
for m in metrics:
    pattern = rf'id="{m}"[^>]*>\s*([^\s<]+)'
    match = re.search(pattern, html)
    assert match is not None, f"Element #{m} not found in HTML"
    val = match.group(1).strip()
    assert val == "N/A", f"Live element #{m} has initial value '{val}', expected 'N/A'"
    print(f"✓ #{m} is initialized to N/A")

# 2. Check header sample pill
match_pill = re.search(r'id="header-sample-pill"[^>]*>\s*([^<\n]+)', html)
assert match_pill and "Idle (Awaiting Input)" in match_pill.group(1), f"Unexpected pill: {match_pill.group(1)}"
print(f"✓ Header pill: '{match_pill.group(1).strip()}'")

# 3. Check sample dimensions
match_dim = re.search(r'id="sample-dimensions"[^>]*>\s*([^<\n]+)', html)
assert match_dim and "No Active Input" in match_dim.group(1), f"Unexpected dim: {match_dim.group(1)}"
print(f"✓ Sample dimensions: '{match_dim.group(1).strip()}'")

# 4. Check JavaScript state definitions
assert "let currentDemo = null;" in html, "currentDemo should be null in live JS"
print("✓ JavaScript currentDemo is initialized to null")

assert "const LAB_STATES" in html, "LAB_STATES missing in live JS"
print("✓ JavaScript 5-state application model (LAB_STATES) is live")

# 5. Check images have hidden class
sec_idx = html.find('id="section-reconstruction"')
sec_end = html.find('id="section-benchmark"')
rec_html = html[sec_idx:sec_end]
img_tags = re.findall(r'<img[^>]+>', rec_html)
assert len(img_tags) == 10, f"Expected 10 img tags, found {len(img_tags)}"
for tag in img_tags:
    assert "class=\"hidden" in tag or "hidden" in tag, f"Img tag missing hidden: {tag}"
print(f"✓ All {len(img_tags)} reconstruction images have class='hidden'")

print("\n🎉 ALL LIVE DEPLOYMENT CHECKS PASSED PERFECTLY!")

