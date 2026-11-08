import time
import urllib.request
import re

for i in range(15):
    try:
        req = urllib.request.Request(
            'https://cv-edgefuse-dcr.onrender.com/',
            headers={'User-Agent': 'Mozilla/5.0', 'Cache-Control': 'no-cache'}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode('utf-8')
            match = re.search(r'id="val-rmse"[^>]*>\s*([^\s<]+)', html)
            val = match.group(1).strip() if match else 'None'
            print(f"Poll {i+1}: val-rmse = '{val}'")
            if val == 'N/A':
                print("🚀 SUCCESS: NEW DEPLOYMENT IS LIVE ON RENDER!")
                exit(0)
    except Exception as e:
        print(f"Poll {i+1}: {e}")
    time.sleep(12)

print("Timed out waiting for Render deploy.")
exit(1)

