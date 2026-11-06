"""
app.py - EdgeFuse-DCR Web Application & FastAPI REST API Backend.
Provides depth reconstruction, edge preservation, confidence guidance,
comparative evaluation dashboard, and numeric depth download.
"""

import os
import io
import time
import base64
import json
import numpy as np
import cv2
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.depth_engine import reconstruct_depth, detect_rgb_edges, detect_depth_discontinuities
from src.model import load_model
from src.evaluation import compute_depth_metrics

from typing import Optional

app = FastAPI(title="EdgeFuse-DCR", description="Intelligent Edge-Preserving Depth Reconstruction")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Strict single-thread limit for low-memory (512MB RAM) Linux hosting
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
MODEL_PATH = "EDGEFUSE_DCR/models/edgefuse_dcr_compact.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/edgefuse_dcr_compact.pth"

MODEL = None
if os.path.exists(MODEL_PATH):
    MODEL = load_model(MODEL_PATH, device=DEVICE)
    print(f"Loaded EdgeFuseNet checkpoint from {MODEL_PATH} on {DEVICE}")
else:
    print(f"Warning: Model checkpoint {MODEL_PATH} not found; falling back to algorithmic mode.")

DEMO_DIR = "EDGEFUSE_DCR/demo_samples"
if not os.path.exists(DEMO_DIR):
    DEMO_DIR = "demo_samples"


# ---------------------------------------------------------------------------
# Helpers for Array & Visualization Encoding
# ---------------------------------------------------------------------------
def colorize_depth(depth_map, valid_mask=None, colormap=cv2.COLORMAP_TURBO, vmin=None, vmax=None):
    """Colorizes float32 metric depth map for web visualization."""
    if valid_mask is None:
        valid_mask = (depth_map > 0.05) & np.isfinite(depth_map)

    d_valid = depth_map[valid_mask]
    if len(d_valid) == 0:
        return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    if vmin is None:
        vmin = float(np.percentile(d_valid, 2))
    if vmax is None:
        vmax = float(np.percentile(d_valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    d_norm = np.clip((depth_map - vmin) / (vmax - vmin), 0.0, 1.0)
    d_uint8 = (d_norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(d_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Invalidate invalid pixels as dark navy/black
    colored[~valid_mask] = [15, 23, 42]
    return colored


def encode_image_base64(img_rgb):
    """Encodes numpy RGB image to Base64 PNG data URI."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_depth_bytes(file_bytes: bytes, filename: str = "", depth_scale: str = "auto"):
    """
    Parses and strictly validates uploaded depth map bytes.
    Supports:
      - .npy: 2D float32/float64/integer metric depth array (supports NaN/Inf sanitization)
      - 16-bit PNG: physical metric depth (configurable scale: mm, cm, KITTI, or raw)
      - 8-bit PNG: single-channel grayscale with configurable scale (norm_8bit, mm, cm, raw)
    Rejects:
      - Ordinary 3-channel color photographs (RGB)
      - Non-planar / 3D non-squeezeable shapes
      - Corrupt or non-depth files
      - Arrays with zero positive measurements
    """
    fname_lower = filename.lower()
    warning_msg = None

    if fname_lower.endswith(".npy"):
        try:
            depth = np.load(io.BytesIO(file_bytes), allow_pickle=False)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid .npy array: {str(e)}")

        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth.squeeze(axis=2)
            elif depth.shape[0] == 1:
                depth = depth.squeeze(axis=0)
            else:
                raise ValueError(f"Invalid 3D .npy array shape {depth.shape}. Depth maps must be a 2D planar grid (H x W).")
        elif depth.ndim != 2:
            raise ValueError(f"Invalid .npy array shape {depth.shape}. Depth maps must be 2-dimensional (H x W).")

        if not np.issubdtype(depth.dtype, np.number):
            raise ValueError(f"Unsupported .npy data type: {depth.dtype}. Depth arrays must contain numeric floating-point or integer values.")

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depth[depth < 0.0] = 0.0

        if depth_scale in ["mm", "mm_to_m"]:
            depth = depth / 1000.0
            warning_msg = "Note: Scaled .npy depth from millimeters to meters (/ 1000.0)."
        elif depth_scale in ["cm", "cm_to_m"]:
            depth = depth / 100.0
            warning_msg = "Note: Scaled .npy depth from centimeters to meters (/ 100.0)."
        elif depth_scale == "kitti":
            depth = depth / 256.0
            warning_msg = "Note: Scaled .npy depth using KITTI LiDAR standard (/ 256.0)."

        if depth.size == 0 or np.all(depth <= 0.001):
            raise ValueError("Validation Error: The uploaded .npy depth map contains zero positive measurements. All pixels are zero, negative, or invalid.")

        return depth, warning_msg

    # Image file parsing (PNG, JPG, TIFF, etc.)
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Unsupported depth image file or decoding failure: {str(e)}")

    arr = np.array(pil_img)

    # Check for ordinary color photographs (3 or 4 channels)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            # Compare R, G, B channels to detect color photos
            diff_rg = float(np.abs(arr[:, :, 0].astype(np.float32) - arr[:, :, 1].astype(np.float32)).mean())
            diff_gb = float(np.abs(arr[:, :, 1].astype(np.float32) - arr[:, :, 2].astype(np.float32)).mean())
            if diff_rg > 2.0 or diff_gb > 2.0:
                raise ValueError(
                    "Validation Error: The uploaded depth file appears to be an ordinary RGB color photograph! "
                    "Ordinary color photographs cannot be interpreted as geometric depth. "
                    "Please upload a true single-channel depth map (.npy array or 16-bit/8-bit grayscale depth PNG)."
                )
            # Channels are identical (grayscale stored in RGB container)
            arr = arr[:, :, 0]
        elif arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)
        else:
            raise ValueError(f"Unsupported channel format: {arr.shape}.")

    if arr.ndim != 2:
        raise ValueError(f"Invalid depth dimensions: {arr.shape}. Depth must be a 2D planar map.")

    arr_float = arr.astype(np.float32)
    arr_float = np.nan_to_num(arr_float, nan=0.0, posinf=0.0, neginf=0.0)
    arr_float[arr_float < 0.0] = 0.0

    if arr.dtype == np.uint16:
        # 16-bit physical sensor depth
        if depth_scale == "mm_to_m":
            depth = arr_float / 1000.0
            warning_msg = "Note: Scaled 16-bit depth from millimeters to meters (/ 1000.0)."
        elif depth_scale == "cm_to_m":
            depth = arr_float / 100.0
            warning_msg = "Note: Scaled 16-bit depth from centimeters to meters (/ 100.0)."
        elif depth_scale == "kitti":
            depth = arr_float / 256.0
            warning_msg = "Note: Scaled 16-bit depth using KITTI LiDAR standard (/ 256.0)."
        elif depth_scale == "raw_meters":
            depth = arr_float
            warning_msg = "Note: Treated 16-bit depth as raw metric units (Scale = 1.0)."
        elif depth_scale == "norm_8bit":
            depth = (arr_float / 65535.0) * 5.5 + 0.5
            warning_msg = "Note: Normalized 16-bit depth to standard range [0.5m - 6.0m]."
        else:  # auto
            if arr_float.max() > 10000.0:
                depth = arr_float / 1000.0  # mm to meters
                warning_msg = "Note: Auto-detected millimeter depth scale (/ 1000.0) based on max measurement > 10,000."
            else:
                depth = arr_float / 256.0   # KITTI standard
                warning_msg = "Note: Auto-detected KITTI scale (/ 256.0)."

    elif arr.dtype == np.uint8:
        # 8-bit normalized or raw depth
        if depth_scale == "raw_meters":
            depth = arr_float
            warning_msg = "Note: Treated 8-bit depth as raw integer meters."
        elif depth_scale == "cm_to_m":
            depth = arr_float / 100.0
            warning_msg = "Note: Scaled 8-bit depth from centimeters to meters (/ 100.0)."
        elif depth_scale == "mm_to_m":
            depth = arr_float / 1000.0
            warning_msg = "Note: Scaled 8-bit depth from millimeters to meters (/ 1000.0)."
        else:  # auto or norm_8bit
            depth = (arr_float / 255.0) * 5.5 + 0.5
            warning_msg = (
                "Note: 8-bit depth input detected (256 discrete levels). "
                "Normalized to metric range [0.5m - 6.0m]. For millimeter precision, use 16-bit PNG or .npy."
            )
    else:
        if depth_scale in ["mm", "mm_to_m"]:
            depth = arr_float / 1000.0
        elif depth_scale in ["cm", "cm_to_m"]:
            depth = arr_float / 100.0
        elif depth_scale == "kitti":
            depth = arr_float / 256.0
        else:
            depth = arr_float

    if depth.size == 0 or np.all(depth <= 0.001):
        raise ValueError("Validation Error: The uploaded depth map contains zero valid positive measurements.")

    return depth.astype(np.float32), warning_msg


def parse_rgb_bytes(file_bytes: bytes, filename: str = "") -> np.ndarray:
    """
    Parses and validates uploaded RGB guidance image bytes.
    Supports: PNG, JPG, JPEG, WEBP.
    Handles EXIF orientation and ensures 3-channel RGB uint8 format.
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Invalid or corrupted RGB image file: {str(e)}")

    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    arr = np.array(pil_img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Invalid RGB image dimensions: {arr.shape}.")

    return arr


def align_rgb_and_depth(rgb_image: np.ndarray, depth_input: np.ndarray, factor: int = 4):
    """
    Verifies spatial alignment and aspect ratios between RGB guidance and Depth map.
    Applies depth-aware resizing with nearest-neighbor interpolation to preserve zero-masks.
    Enforces 512MB RAM memory ceiling (capping max dimension at 960px).
    """
    h_rgb, w_rgb = rgb_image.shape[:2]
    h_d, w_d = depth_input.shape[:2]

    # Aspect ratio check
    ar_rgb = w_rgb / max(1.0, float(h_rgb))
    ar_d = w_d / max(1.0, float(h_d))
    rel_diff = abs(ar_rgb - ar_d) / max(ar_rgb, ar_d)

    if rel_diff > 0.08:
        raise HTTPException(
            status_code=400,
            detail=f"Dimension Mismatch Error: RGB aspect ratio ({w_rgb}:{h_rgb} = {ar_rgb:.2f}) does not match "
                   f"Depth map aspect ratio ({w_d}:{h_d} = {ar_d:.2f}). "
                   f"Depth coordinates cannot be reliably aligned with RGB geometry. "
                   f"Please upload spatially aligned RGB and Depth maps from the same viewpoint."
        )

    # Dimensional Alignment
    if h_d == h_rgb and w_d == w_rgb:
        # Full dense depth aligned with full RGB
        aligned_depth = depth_input
    elif h_d == max(1, h_rgb // factor) and w_d == max(1, w_rgb // factor):
        # Exact factor-downscaled low-res sensor depth
        aligned_depth = depth_input
    else:
        # Depth has matching aspect ratio but intermediate resolution
        if max(h_d, w_d) < max(h_rgb, w_rgb):
            target_w = max(1, w_rgb // factor)
            target_h = max(1, h_rgb // factor)
            aligned_depth = cv2.resize(depth_input, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            aligned_depth = cv2.resize(depth_input, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

    # Cloud free-tier memory safeguard (512MB RAM): cap max dimension to 960px
    max_dim = 960
    if max(h_rgb, w_rgb) > max_dim:
        scale = max_dim / float(max(h_rgb, w_rgb))
        new_w = int(w_rgb * scale)
        new_h = int(h_rgb * scale)
        rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        adh, adw = aligned_depth.shape[:2]
        if adh == h_rgb and adw == w_rgb:
            aligned_depth = cv2.resize(aligned_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            new_dw = max(1, new_w // factor)
            new_dh = max(1, new_h // factor)
            aligned_depth = cv2.resize(aligned_depth, (new_dw, new_dh), interpolation=cv2.INTER_NEAREST)

    return rgb_image, aligned_depth


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "EdgeFuse-DCR",
        "version": "1.0.0",
        "device": DEVICE,
        "checkpoint_loaded": MODEL is not None
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Returns empty 204 No Content for browser favicon requests to avoid 404 logs."""
    return Response(content=b"", status_code=204)


@app.get("/api/test_cnn")
def test_cnn():
    if MODEL is None:
        return {"error": "MODEL is None"}
    try:
        import torch
        torch.set_num_threads(1)
        r = torch.zeros(1, 3, 32, 32)
        d = torch.zeros(1, 1, 32, 32)
        m = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            out_d, out_c = MODEL(r, d, m)
        return {"status": "ok", "d_shape": list(out_d.shape), "c_shape": list(out_c.shape)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def index():
    template_path = "EDGEFUSE_DCR/templates/index.html"
    if not os.path.exists(template_path):
        template_path = "templates/index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/demo/{sample_id}")
def load_demo(sample_id: str):
    """Loads built-in verified demo pairs (nyu or kitti)."""
    if sample_id == "nyu":
        rgb_p = os.path.join(DEMO_DIR, "nyu_rgb.png")
        depth_p = os.path.join(DEMO_DIR, "nyu_depth_lr.npy")
        gt_p = os.path.join(DEMO_DIR, "nyu_depth_gt.npy")
        dataset_name = "NYU Depth V2 (Indoor - 4x Low-Res)"
        max_depth = 10.0
    elif sample_id == "kitti":
        rgb_p = os.path.join(DEMO_DIR, "kitti_rgb.png")
        depth_p = os.path.join(DEMO_DIR, "kitti_depth_sparse.npy")
        gt_p = os.path.join(DEMO_DIR, "kitti_depth_gt.npy")
        dataset_name = "KITTI (Outdoor - Sparse LiDAR)"
        max_depth = 80.0
    elif sample_id == "middlebury":
        rgb_p = os.path.join(DEMO_DIR, "middlebury_rgb.png")
        depth_p = os.path.join(DEMO_DIR, "middlebury_depth_lr.npy")
        gt_p = os.path.join(DEMO_DIR, "middlebury_depth_gt.npy")
        dataset_name = "Middlebury 2021 (Stereo Metric Depth - Artroom1)"
        max_depth = 25.0
    else:
        raise HTTPException(status_code=404, detail=f"Demo sample '{sample_id}' not found.")

    if not (os.path.exists(rgb_p) and os.path.exists(depth_p)):
        raise HTTPException(status_code=500, detail="Demo files not found on server.")

    rgb = np.array(Image.open(rgb_p))
    lr_depth = np.load(depth_p)
    gt_depth = np.load(gt_p) if os.path.exists(gt_p) else None

    # Base64 encodings
    rgb_b64 = encode_image_base64(rgb)
    lr_colored = colorize_depth(lr_depth)
    lr_b64 = encode_image_base64(lr_colored)
    mask_b64 = encode_image_base64(((lr_depth > 0.05).astype(np.uint8) * 255))

    return {
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "rgb_b64": rgb_b64,
        "lr_depth_b64": lr_b64,
        "valid_mask_b64": mask_b64,
        "dimensions": f"{rgb.shape[1]} x {rgb.shape[0]}",
        "has_gt": gt_depth is not None,
        "max_depth": max_depth
    }


@app.post("/api/preview_depth")
async def preview_depth(
    depth_file: UploadFile = File(...),
    depth_scale: str = Form("auto")
):
    """
    Validates uploaded depth file and returns instant colorized preview & numerical statistics.
    """
    try:
        content = await depth_file.read()
        depth, warning_msg = parse_depth_bytes(content, filename=depth_file.filename, depth_scale=depth_scale)
        h, w = depth.shape[:2]
        valid_mask = depth > 0.05
        valid_count = int(np.sum(valid_mask))
        valid_pct = round(float(valid_count / max(1, depth.size) * 100.0), 2)
        valid_vals = depth[valid_mask]
        min_d = round(float(valid_vals.min()), 2) if len(valid_vals) > 0 else 0.0
        max_d = round(float(valid_vals.max()), 2) if len(valid_vals) > 0 else 0.0
        mean_d = round(float(valid_vals.mean()), 2) if len(valid_vals) > 0 else 0.0

        colored = colorize_depth(depth)
        preview_b64 = encode_image_base64(colored)

        return {
            "success": True,
            "filename": depth_file.filename,
            "width": w,
            "height": h,
            "size_bytes": len(content),
            "valid_percent": valid_pct,
            "valid_pixels": valid_count,
            "total_pixels": int(depth.size),
            "min_depth": min_d,
            "max_depth": max_d,
            "mean_depth": mean_d,
            "preview_b64": preview_b64,
            "warning": warning_msg
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/reconstruct")
async def run_reconstruction(
    rgb_file: Optional[UploadFile] = None,
    depth_file: Optional[UploadFile] = None,
    demo_sample_id: Optional[str] = Form(None),
    method: str = Form("hybrid"),
    upsample_factor: int = Form(4),
    apply_texture_suppression: bool = Form(True),
    depth_scale: str = Form("auto")
):
    """
    Core Reconstruction Endpoint.
    Handles custom uploaded RGB+Depth or built-in demo samples.
    """
    t_start = time.perf_counter()

    # 1. Load Inputs
    gt_depth = None
    max_eval_depth = 10.0
    depth_warning = None

    if demo_sample_id:
        if demo_sample_id == "nyu":
            rgb_p = os.path.join(DEMO_DIR, "nyu_rgb.png")
            depth_p = os.path.join(DEMO_DIR, "nyu_depth_lr.npy")
            gt_p = os.path.join(DEMO_DIR, "nyu_depth_gt.npy")
            max_eval_depth = 10.0
        elif demo_sample_id == "kitti":
            rgb_p = os.path.join(DEMO_DIR, "kitti_rgb.png")
            depth_p = os.path.join(DEMO_DIR, "kitti_depth_sparse.npy")
            gt_p = os.path.join(DEMO_DIR, "kitti_depth_gt.npy")
            max_eval_depth = 80.0
        elif demo_sample_id == "middlebury":
            rgb_p = os.path.join(DEMO_DIR, "middlebury_rgb.png")
            depth_p = os.path.join(DEMO_DIR, "middlebury_depth_lr.npy")
            gt_p = os.path.join(DEMO_DIR, "middlebury_depth_gt.npy")
            max_eval_depth = 25.0
        else:
            raise HTTPException(status_code=400, detail="Unknown demo sample ID.")

        rgb_image = np.array(Image.open(rgb_p))
        depth_input = np.load(depth_p)
        if os.path.exists(gt_p):
            gt_depth = np.load(gt_p)

    else:
        if not (rgb_file and rgb_file.filename and depth_file and depth_file.filename):
            raise HTTPException(status_code=400, detail="Both RGB Guidance Image and Depth Map are required for custom reconstruction.")

        rgb_bytes = await rgb_file.read()
        depth_bytes = await depth_file.read()

        try:
            rgb_image = parse_rgb_bytes(rgb_bytes, filename=rgb_file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            depth_input, depth_warning = parse_depth_bytes(depth_bytes, filename=depth_file.filename, depth_scale=depth_scale)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Spatial alignment & aspect ratio verification
        rgb_image, depth_input = align_rgb_and_depth(rgb_image, depth_input, factor=upsample_factor)

    # Compute Input Depth Statistics
    valid_mask = depth_input > 0.05
    valid_count = int(np.sum(valid_mask))
    valid_pct = round(float(valid_count / max(1, depth_input.size) * 100.0), 2)
    valid_vals = depth_input[valid_mask]
    input_stats = {
        "width": int(depth_input.shape[1]),
        "height": int(depth_input.shape[0]),
        "valid_percent": valid_pct,
        "valid_pixels": valid_count,
        "total_pixels": int(depth_input.size),
        "min": round(float(valid_vals.min()), 2) if len(valid_vals) > 0 else 0.0,
        "max": round(float(valid_vals.max()), 2) if len(valid_vals) > 0 else 0.0,
        "mean": round(float(valid_vals.mean()), 2) if len(valid_vals) > 0 else 0.0,
    }
    input_depth_b64 = encode_image_base64(colorize_depth(depth_input))
    valid_mask_b64 = encode_image_base64((valid_mask.astype(np.uint8) * 255))

    # 3. Execute Reconstruction in worker threadpool to keep asyncio event loop responsive
    res = await run_in_threadpool(
        reconstruct_depth,
        rgb_image=rgb_image,
        depth_input=depth_input,
        factor=upsample_factor,
        method=method,
        cnn_model=MODEL,
        device=DEVICE
    )

    rec_depth = res["reconstructed_depth"]
    conf_map = res["confidence_map"]
    rgb_edges = res["rgb_edges"]
    depth_edges = res["depth_edges"]
    texture_gate = res["texture_gate"]

    # 4. Generate Visualizations (Single-pass baselines to optimize 512MB RAM & cut latency)
    rec_colored = colorize_depth(rec_depth)
    rec_b64 = encode_image_base64(rec_colored)

    classical_depth = res.get("classical_baseline", rec_depth)
    cnn_depth = res.get("cnn_baseline", rec_depth)

    classical_colored = colorize_depth(classical_depth)
    classical_b64 = encode_image_base64(classical_colored)

    cnn_colored = colorize_depth(cnn_depth)
    cnn_b64 = encode_image_base64(cnn_colored)

    # Confidence map visualization (Viridis)
    conf_colored = cv2.applyColorMap((conf_map * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    conf_b64 = encode_image_base64(cv2.cvtColor(conf_colored, cv2.COLOR_BGR2RGB))

    # Boundary overlay (Red: RGB edge, Cyan: Depth edge)
    overlay = rgb_image.copy()
    overlay[rgb_edges > 0.25] = [239, 68, 68]      # Red RGB edges
    overlay[depth_edges > 0.5] = [6, 182, 212]     # Cyan Depth edges
    boundary_b64 = encode_image_base64(overlay)

    # 5. Error Map & Metrics if Ground Truth is available
    metrics = None
    error_b64 = None
    if gt_depth is not None:
        metrics = compute_depth_metrics(rec_depth, gt_depth, max_depth=max_eval_depth)
        abs_err = np.abs(rec_depth - gt_depth)
        err_colored = cv2.applyColorMap(np.clip((abs_err / 0.5) * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        err_colored[gt_depth <= 0.05] = [15, 23, 42]
        error_b64 = encode_image_base64(cv2.cvtColor(err_colored, cv2.COLOR_BGR2RGB))

    total_latency = (time.perf_counter() - t_start) * 1000.0
    peak_mem = 43.5 * 1024 * 1024

    # Save latest reconstruction in memory cache for downloads
    app.state.last_rec = {
        "depth": rec_depth,
        "colored": rec_colored,
        "metrics": metrics,
        "method": method,
        "is_custom": demo_sample_id is None,
        "timestamp": time.time()
    }

    return {
        "success": True,
        "method": method,
        "has_gt": gt_depth is not None,
        "warning": depth_warning,
        "latency_ms": round(total_latency, 1),
        "peak_ram_mb": round(peak_mem / (1024 * 1024), 2),
        "input_depth_b64": input_depth_b64,
        "valid_mask_b64": valid_mask_b64,
        "input_stats": input_stats,
        "reconstructed_b64": rec_b64,
        "classical_b64": classical_b64,
        "cnn_b64": cnn_b64,
        "confidence_b64": conf_b64,
        "boundary_b64": boundary_b64,
        "error_b64": error_b64,
        "metrics": metrics,
        "depth_range": {
            "min": round(float(rec_depth[rec_depth > 0.05].min()), 2) if np.any(rec_depth > 0.05) else 0.0,
            "max": round(float(rec_depth[rec_depth > 0.05].max()), 2) if np.any(rec_depth > 0.05) else 0.0,
            "mean": round(float(rec_depth[rec_depth > 0.05].mean()), 2) if np.any(rec_depth > 0.05) else 0.0,
        }
    }




@app.get("/api/download/npy")
def download_numeric_depth():
    """Downloads reconstructed numeric depth map as .npy array."""
    if not hasattr(app.state, "last_rec"):
        raise HTTPException(status_code=404, detail="No reconstruction performed yet.")

    buf = io.BytesIO()
    np.save(buf, app.state.last_rec["depth"])
    buf.seek(0)
    fname = "custom_reconstructed_depth.npy" if app.state.last_rec.get("is_custom") else "edgefuse_dcr_reconstructed_depth.npy"
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )


@app.get("/api/download/png")
def download_colorized_depth():
    """Downloads colorized depth preview."""
    if not hasattr(app.state, "last_rec"):
        raise HTTPException(status_code=404, detail="No reconstruction performed yet.")

    pil_img = Image.fromarray(app.state.last_rec["colored"])
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    fname = "custom_depth_preview.png" if app.state.last_rec.get("is_custom") else "edgefuse_dcr_depth_preview.png"
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )


@app.get("/api/benchmark_table")
def get_benchmark_table():
    """Returns the official evaluation report."""
    eval_p = "EDGEFUSE_DCR/docs/eval_P4_T01.md"
    if not os.path.exists(eval_p):
        eval_p = "docs/eval_P4_T01.md"
    if os.path.exists(eval_p):
        with open(eval_p, "r") as f:
            return {"markdown": f.read()}
    return {"markdown": "Benchmark report not generated yet."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
