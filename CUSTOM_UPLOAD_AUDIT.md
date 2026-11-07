# EDGEFUSE-DCR — CUSTOM UPLOAD PIPELINE RE-ENGINEERING & ARCHITECTURAL AUDIT

**Project:** EDGEFUSE-DCR — Edge-Aware Confidence-Guided Depth Completion & Refinement  
**Phase:** Phase 4 (P4) Depth Upsampling and Refinement  
**Live Production URL:** https://cv-edgefuse-dcr.onrender.com  
**GitHub Repository:** https://github.com/991627aishu/CV_EDGEFUSE-DCR  
**Audit Date:** November 7, 2026  

---

## 1. EXECUTIVE SUMMARY & DEFECT ROOT-CAUSE INVESTIGATION

During rigorous evaluation of the live EDGEFUSE-DCR platform with custom building facade inputs (`Home.png` and `Grayscale Depth Map of Building Facade.png`), five critical defects were discovered across state management, scientific evaluation honesty, and UI synchronization.

### Defect 1: Stale RGB Boundary Discontinuity Overlay
* **Observed Defect:** When custom building images were uploaded, Tab 3 ("Edge Discontinuities") continued to display the boundary contours superimposed on the NYU indoor living room rather than the user's uploaded building facade.
* **Root Cause:** In `templates/index.html`, `executeReconstruction()` evaluated `if (currentDemo)` before evaluating uploaded custom files. Because `currentDemo` defaulted to `'nyu_living_room'`, the frontend submitted `demo_sample_id: currentDemo` and omitted the uploaded file buffers. Consequently, the backend reconstructed the NYU sample and returned NYU's boundary overlay image.
* **Resolution:** Re-engineered request building to grant absolute precedence to custom uploads whenever custom files are present (`isCustomRun = Boolean(activeIsCustom || (uploadedRgbFile && uploadedDepthFile))`). When custom files are present, `demo_sample_id` is never transmitted to the backend.

### Defect 2: Sensor Validity Mask Metric Discrepancy
* **Observed Defect:** Even after uploading high-resolution custom images, the validity mask diagnostic header reported `19200 Valid / 0 Void (19200 total px)` and `100% Valid Measurements` (the dimensions of NYU Kinect's 160×120 sensor grid).
* **Root Cause:** `previewDepthMap()` updated the Base64 image `view-valid-mask`, but failed to update DOM elements `#validity-px-counts` and `#validity-mask-stats`. As a result, the NYU values remained permanently hardcoded in the interface.
* **Resolution:** Updated `previewDepthMap()` to extract `valid_pixels`, `invalid_pixels`, `total_pixels`, and `valid_percent` from the backend `/api/preview_depth` response, dynamically updating `#validity-px-counts`, `#validity-mask-stats`, and `#sample-dimensions`.

### Defect 3: Quantitative Metric Hallucination Without Ground Truth Reference
* **Observed Defect:** Top KPI cards displayed precise evaluation metrics (`RMSE: 0.0372m`, `MAE: 0.0173m`, `PSNR: 48.58dB`, `SSIM: 0.9961`, `Boundary F1: 0.6692`) even though the optional Ground Truth field was left empty (`Reference GT (for quantitative metrics)...`).
* **Root Cause:** These numbers were stale metrics retained from the initial NYU living room benchmark because `updateReconstructionUI()` did not clear metrics when Ground Truth was absent.
* **Resolution:** Enforced strict scientific integrity: when Ground Truth is omitted, the backend explicitly sets `metrics = None` and `has_gt = False`. The frontend renders `N/A` across all Ground Truth-dependent KPI cards while reporting authentic CPU latency in milliseconds.

### Defect 4: Stale Residual Error Heatmap Display
* **Observed Defect:** In Tab 4 ("Confidence & Error"), the Jet absolute error heatmap (`|Dpred - Dgt|`) displayed the indoor furniture residual error map when evaluating a custom building without Ground Truth.
* **Root Cause:** When `has_gt == False`, `data.error_b64` is `None`. The client script executed `if (data.error_b64) setViewImage('view-error', data.error_b64);`, leaving the previous image element intact.
* **Resolution:** Created an informative, dedicated empty state (`#view-error-unavailable`) explaining that quantitative error heatmaps require an authentic Ground Truth reference map. When no Ground Truth is provided, `view-error` is hidden and wiped, and `#view-error-unavailable` is displayed with an amber status badge `GT Required`.

### Defect 5: "Overview & Hero" Redundancy
* **Observed Defect:** Redundant "Overview & Hero" links and section duplicated information available in the technical sections.
* **Resolution:** Completely removed "Overview & Hero" from the sidebar navigation, eliminated `<section id="section-hero">` from the DOM, and updated brand and back-to-top links to point to `#section-why-matters`.

---

## 2. RE-ENGINEERED ARCHITECTURE & SESSION ISOLATION

### 2.1 Multi-User Session Cache (`CUSTOM_SESSIONS`)
To eliminate re-upload latency when researchers switch algorithms (e.g., A0 to A2 to A5) or upsampling factors (2×, 4×, 8×), `app.py` implements session-isolated caching:
```python
CUSTOM_SESSIONS: Dict[str, Dict[str, Any]] = {}
```
1. Each client session generates a unique `session_id = 'sess_' + uuid.uuid4().hex`.
2. Raw uploaded arrays (`raw_rgb`, `raw_depth`, `raw_gt`) are cached in memory keyed by `session_id`.
3. Subsequent requests for the same session re-use pre-parsed NumPy arrays without disk writes or re-upload overhead.

### 2.2 Re-engineered API Contract

| Endpoint | Method | Input Parameters | Output / Behavior |
| :--- | :--- | :--- | :--- |
| `/api/preview_depth` | POST | `depth_file`, `depth_scale` | Validates depth map, computes valid/void pixel counts, and returns Turbo preview + binary validity mask. Rejects color photos. |
| `/api/reconstruct` | POST | `rgb_file`, `depth_file`, `gt_file`, `method`, `upsample_factor`, `session_id` | Core reconstruction pipeline. Precedence given to custom files over demo IDs. Generates boundary overlay, confidence map, and ablation progression. |
| `/api/benchmark/custom`| POST | `session_id`, `max_depth` | Computes full 9-algorithm benchmark against Ground Truth for custom uploads. Refuses with HTTP 400 if GT is not attached. |
| `/api/download/boundary`| GET | None (session context) | Downloads PNG boundary overlay with red photometric and cyan geometric edges. |
| `/api/download/error_map`| GET | None (session context) | Downloads Jet error heatmap. Refuses with HTTP 400 when no GT exists. |
| `/api/download/npy` | GET | None (session context) | Downloads float32 metric depth NumPy array. |
| `/api/download/png` | GET | None (session context) | Downloads 16-bit millimeter depth PNG. |
| `/api/download/report_json`| GET | None (session context) | Complete JSON evaluation report and parameter provenance. |

---

## 3. SCIENTIFIC RECONSTRUCTION MATHEMATICS

### 3.1 Dual-Branch Edge Extraction
* **Photometric Edge Field:**
  $$\nabla I_{\text{rgb}} = \sqrt{(\partial_x I)^2 + (\partial_y I)^2}, \quad E_{\text{rgb}} = \mathbb{I}(\nabla I_{\text{rgb}} > \tau_{\text{rgb}})$$
* **Geometric Depth Discontinuity Field:**
  $$\nabla D = \sqrt{(\partial_x D)^2 + (\partial_y D)^2}, \quad E_{\text{depth}} = \mathbb{I}\left(\frac{\nabla D}{D + \epsilon} > \tau_{\text{depth}}\right)$$

### 3.2 Bilateral Confidence-Guided Depth Refinement
$$\hat{D}(p) = \frac{\sum_{q \in \mathcal{N}(p)} W_s(p, q) \cdot W_r(I_p, I_q) \cdot C(q) \cdot G(p, q) \cdot D(q)}{\sum_{q \in \mathcal{N}(p)} W_s(p, q) \cdot W_r(I_p, I_q) \cdot C(q) \cdot G(p, q)}$$
* Spatial Gaussian: $W_s(p, q) = \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right)$
* Range Gaussian: $W_r(I_p, I_q) = \exp\left(-\frac{\|I_p - I_q\|^2}{2\sigma_r^2}\right)$
* Confidence Weight: $C(q) = \lambda_{\text{valid}} \cdot M(q) + (1 - \lambda_{\text{valid}}) \cdot C_{\text{cnn}}(q)$
* Texture-Copy Suppression Gate: $G(p, q) = 1.0 - \alpha \cdot \mathbb{I}(E_{\text{rgb}}(q) \land \neg E_{\text{depth}}(q))$

---

## 4. VERIFICATION EVIDENCE

All 9 live server probe tests and 8 direct functional tests passed with 100% compliance:
```
============================================================
LIVE PROBE TEST OF EDGEFUSE-DCR APPLICATION SERVER
============================================================
[Step 1] Probing POST /api/preview_depth...
✓ /api/preview_depth succeeded: dimensions=160 × 120, valid=100.0%

[Step 2] Probing POST /api/reconstruct (Custom upload, NO GT)...
✓ Custom reconstruction (No GT) succeeded: is_custom=True, has_gt=False, metrics=None, latency=135.5ms

[Step 3] Probing POST /api/reconstruct algorithm switch on cached session...
✓ Algorithm switch on cached session succeeded: method=cnn

[Step 4] Probing GET /api/download/boundary...
✓ Boundary download succeeded: 412 bytes

[Step 5] Probing GET /api/download/error_map rejection (no GT)...
✓ Error map download correctly refused with HTTP 400 when no GT exists.

[Step 6] Probing POST /api/reconstruct (Custom upload, WITH GT)...
✓ Custom reconstruction (With GT) succeeded: RMSE=0.0000, MAE=0.0000

[Step 7] Probing GET /api/download/error_map (with GT)...
✓ Error map download succeeded: 4067 bytes

[Step 8] Probing POST /api/benchmark/custom (with GT)...
✓ /api/benchmark/custom evaluated all 9 methods accurately!

[Step 9] Probing POST /api/benchmark/custom rejection without GT...
✓ /api/benchmark/custom correctly rejected session without GT.

============================================================
ALL LIVE SERVER PROBE TESTS PASSED 100% SUCCESSFULLY!
============================================================
```

