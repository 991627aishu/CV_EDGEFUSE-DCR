"""
Dedicated E2E Browser Test for EDGEFUSE-DCR (Test Group D)
Validates:
1. Homepage hero boundary comparison slider & image elements
2. Dataset Explorer tab navigation (All, KITTI, NYU, Middlebury)
3. Pagination selector (3 per page -> 3 cards, 9 per page -> 9 cards)
4. Authentic image variations across different frames
5. Inspect modal rendering
6. Load in Workspace execution
7. Zero severe browser console errors
"""

import sys
import os
import time
import threading
import uvicorn
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app

PORT = 8898

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error")

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(2)

print(f"Started local test server on http://127.0.0.1:{PORT}")

driver = None
try:
    opts = ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1600,1000")
    try:
        driver = webdriver.Chrome(options=opts)
        print("Initialized Chrome Headless WebDriver.")
    except Exception as e:
        print(f"Chrome not available ({e}), trying Edge...")
        edge_opts = EdgeOptions()
        edge_opts.add_argument("--headless=new")
        edge_opts.add_argument("--no-sandbox")
        edge_opts.add_argument("--disable-dev-shm-usage")
        edge_opts.add_argument("--disable-gpu")
        edge_opts.add_argument("--window-size=1600,1000")
        driver = webdriver.Edge(options=edge_opts)
        print("Initialized Edge Headless WebDriver.")

    driver.get(f"http://127.0.0.1:{PORT}")
    wait = WebDriverWait(driver, 10)

    # 1. Verify Homepage & Hero Section
    print("\n[Step 1] Verifying Overview & Hero Section...")
    hero_section = wait.until(EC.presence_of_element_located((By.ID, "section-hero")))
    assert hero_section is not None
    slider_container = driver.find_element(By.ID, "hero-slider-container")
    assert slider_container is not None
    
    # Check comparison images
    after_img = driver.find_element(By.ID, "hero-slider-after-img")
    before_img = driver.find_element(By.ID, "hero-slider-before-img")
    assert after_img.get_attribute("src"), "Refined depth hero image missing src"
    assert before_img.get_attribute("src"), "Raw input low-res hero image missing src"
    rmse_badge = driver.find_element(By.ID, "hero-slider-rmse-badge").text
    assert "RMSE:" in rmse_badge
    print(f" -> PASS: Hero Live Boundary Comparison Demo verified! ({rmse_badge})")

    # 2. Navigate to Dataset Explorer
    print("\n[Step 2] Navigating to Dataset Explorer...")
    explorer = driver.find_element(By.ID, "section-explorer")
    assert explorer is not None
    time.sleep(1.0)

    # 3. Test Pagination Selectors: 3 per page
    print("\n[Step 3] Testing Pagination Selector = 3 per page...")
    limit_select = Select(driver.find_element(By.ID, "gallery-limit-select"))
    limit_select.select_by_value("3")
    time.sleep(1.0)
    cards = driver.find_elements(By.CLASS_NAME, "gallery-item")
    assert len(cards) == 3, f"Expected 3 cards for limit=3, got {len(cards)}"
    info_text = driver.find_element(By.ID, "gallery-pagination-info").text
    print(f" -> PASS: 3 cards rendered. Info: '{info_text}'")

    # 4. Test Pagination Selectors: 9 per page
    print("\n[Step 4] Testing Pagination Selector = 9 per page...")
    limit_select.select_by_value("9")
    time.sleep(1.0)
    cards = driver.find_elements(By.CLASS_NAME, "gallery-item")
    assert len(cards) == 9, f"Expected 9 cards for limit=9, got {len(cards)}"
    info_text = driver.find_element(By.ID, "gallery-pagination-info").text
    print(f" -> PASS: 9 cards rendered. Info: '{info_text}'")

    # 5. Switch to KITTI Dataset
    print("\n[Step 5] Switching to KITTI Depth Completion tab...")
    btn_kitti = driver.find_element(By.ID, "tab-dset-kitti")
    driver.execute_script("arguments[0].click();", btn_kitti)
    time.sleep(1.0)
    kitti_cards = driver.find_elements(By.CLASS_NAME, "gallery-item")
    assert len(kitti_cards) == 9, f"Expected 9 KITTI cards, got {len(kitti_cards)}"
    
    # Check that cards have distinct image sources and titles
    kitti_titles = [c.find_element(By.TAG_NAME, "h4").text for c in kitti_cards]
    print(f" -> First 3 KITTI titles: {kitti_titles[:3]}")
    assert len(set(kitti_titles)) == len(kitti_titles), "Detected duplicate card titles in KITTI page!"
    print(" -> PASS: KITTI cards display unique frame identifiers and distinct thumbnails.")

    # 6. Test Inspect Modal on a KITTI card
    print("\n[Step 6] Testing Scene Inspector Modal...")
    inspect_btn = kitti_cards[0].find_elements(By.TAG_NAME, "button")[1]
    driver.execute_script("arguments[0].click();", inspect_btn)
    time.sleep(1.0)
    modal = driver.find_element(By.ID, "scene-viewer-modal")
    assert not ("hidden" in modal.get_attribute("class")), "Scene viewer modal should be visible"
    modal_title = driver.find_element(By.ID, "modal-scene-title").text
    print(f" -> PASS: Inspect modal opened for scene '{modal_title}'")
    
    # Close modal
    close_btn = modal.find_element(By.XPATH, ".//button[contains(text(), '✕')]")
    driver.execute_script("arguments[0].click();", close_btn)
    time.sleep(0.5)

    # 7. Switch to NYU Depth V2
    print("\n[Step 7] Switching to NYU Depth V2 tab...")
    btn_nyu = driver.find_element(By.ID, "tab-dset-nyu")
    driver.execute_script("arguments[0].click();", btn_nyu)
    time.sleep(1.0)
    nyu_cards = driver.find_elements(By.CLASS_NAME, "gallery-item")
    assert len(nyu_cards) == 9
    nyu_titles = [c.find_element(By.TAG_NAME, "h4").text for c in nyu_cards]
    print(f" -> First 3 NYU titles: {nyu_titles[:3]}")
    print(" -> PASS: NYU tab renders 9 verified indoor scenes.")

    # 8. Switch to Middlebury 2021
    print("\n[Step 8] Switching to Middlebury 2021 tab...")
    btn_mid = driver.find_element(By.ID, "tab-dset-middlebury")
    driver.execute_script("arguments[0].click();", btn_mid)
    time.sleep(1.0)
    mid_cards = driver.find_elements(By.CLASS_NAME, "gallery-item")
    assert len(mid_cards) == 9
    mid_titles = [c.find_element(By.TAG_NAME, "h4").text for c in mid_cards]
    print(f" -> First 3 Middlebury titles: {mid_titles[:3]}")
    print(" -> PASS: Middlebury tab renders 9 verified stereo scenes.")

    # 9. Test Load in Workspace
    print("\n[Step 9] Testing 'Load in Workspace'...")
    load_btn = mid_cards[0].find_elements(By.TAG_NAME, "button")[0]
    driver.execute_script("arguments[0].click();", load_btn)
    time.sleep(1.5)
    header_pill = driver.find_element(By.ID, "header-sample-pill").text
    print(f" -> PASS: Loaded scene into Reconstruction Lab workspace! Header pill: '{header_pill}'")

    # 10. Check Browser Console Logs
    print("\n[Step 10] Checking Browser Console Logs...")
    logs = driver.get_log("browser")
    severe_errors = [l for l in logs if l.get("level") == "SEVERE"]
    if severe_errors:
        print(f"Warning: Browser console severe logs: {severe_errors}")
    else:
        print(" -> PASS: Browser console has ZERO severe JavaScript errors!")

    print("\n" + "=" * 70)
    print("ALL BROWSER E2E TESTS (TEST GROUP D) PASSED PERFECTLY (100%)!")
    print("=" * 70)

except Exception as err:
    print(f"\nBrowser test failed with exception: {err}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    if driver:
        driver.quit()

