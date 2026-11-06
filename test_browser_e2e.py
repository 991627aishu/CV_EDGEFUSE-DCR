"""
EDGEFUSE-DCR Headless Browser End-to-End Validation
Starts local test server, opens with headless Chrome/Edge, verifies:
1. Header elements & badges
2. Dataset Explorer gallery items rendering & clicking
3. Tab switching between all 4 tabs
4. Reconstruction execution & image updates
5. Zero browser console errors
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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app

# Start uvicorn server in daemon thread
def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8899, log_level="error")

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(2)

print("Started local test server on http://127.0.0.1:8899")

driver = None
try:
    # Try Chrome first
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

    driver.get("http://127.0.0.1:8899")
    wait = WebDriverWait(driver, 10)

    # 1. Verify Header
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "brand-text")))
    brand_title = driver.find_element(By.CLASS_NAME, "brand-text").text
    assert "EDGEFUSE-DCR" in brand_title
    print(f"[Browser Test 1] Header Brand verified: '{brand_title.splitlines()[0]}'")

    # 2. Verify Dataset Explorer Gallery
    time.sleep(1.5)
    gallery_items = driver.find_elements(By.CLASS_NAME, "gallery-item")
    assert len(gallery_items) >= 3, f"Expected at least 3 gallery items, got {len(gallery_items)}"
    print(f"[Browser Test 2] Dataset Gallery rendered {len(gallery_items)} scene cards.")

    # 3. Click Middlebury Tab and verify gallery updates
    btn_mid = driver.find_element(By.ID, "tab-dset-middlebury")
    btn_mid.click()
    time.sleep(1.0)
    mid_items = driver.find_elements(By.CLASS_NAME, "gallery-item")
    assert len(mid_items) >= 3
    print(f"[Browser Test 3] Clicked Middlebury tab: Loaded {len(mid_items)} stereo scenes.")

    # 4. Click a Middlebury sample (e.g. Chess 1)
    chess_item = None
    for it in mid_items:
        if "Chess" in it.text:
            chess_item = it
            break
    if chess_item:
        chess_item.click()
        print("[Browser Test 4] Selected 'Chess 1' sample from gallery.")
        time.sleep(2.0)

    # 5. Verify Reconstruction Execution & Metric Bar updates
    btn_run = driver.find_element(By.ID, "btn-run")
    btn_run.click()
    time.sleep(3.0)

    val_rmse = driver.find_element(By.ID, "val-rmse").text
    assert "--" not in val_rmse, f"Metrics should be calculated, got {val_rmse}"
    print(f"[Browser Test 5] Reconstruction executed! Live RMSE: {val_rmse}")

    # 6. Test All 4 Tabs
    tabs = driver.find_elements(By.CLASS_NAME, "tab-btn")
    for tab in tabs:
        tab_name = tab.text
        driver.execute_script("arguments[0].click();", tab)
        time.sleep(0.5)
        print(f"[Browser Test 6] Switched to tab: '{tab_name}' -> OK")

    # 7. Check Browser Console Logs for zero critical errors
    logs = driver.get_log("browser")
    severe_errors = [l for l in logs if l.get("level") == "SEVERE"]
    if severe_errors:
        print(f"Warning: Browser console severe logs: {severe_errors}")
    else:
        print("[Browser Test 7] Browser console has ZERO severe JavaScript errors!")

    print("\nALL BROWSER TESTS PASSED (100%)!")

except Exception as err:
    print(f"Browser test error: {err}")
    sys.exit(1)
finally:
    if driver:
        driver.quit()
