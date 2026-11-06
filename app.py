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
import tracemalloc
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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


def parse_depth_bytes(file_bytes, filename=""):
    """
    Parses uploaded depth map bytes.
    Supports:
      - .npy: float32 / float64 numpy array
      - 16-bit PNG: (if values > 255, assumes mm / 1000.0 or KITTI / 256.0)
      - 8-bit PNG/JPG
    """
    if filename.endswith(".npy"):
        depth = np.load(io.BytesIO(file_bytes))
        return depth.astype(np.float32)

    # Image format
    pil_img = Image.open(io.BytesIO(file_bytes))
    arr = np.array(pil_img)

    if len(arr.shape) == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if arr.dtype == np.uint16:
        # Check if KITTI scale (max > 5000) or NYU mm scale
        if arr.max() > 10000:
            depth = arr.astype(np.float32) / 1000.0  # mm to meters
        else:
            depth = arr.astype(np.float32) / 256.0   # KITTI standard
    else:
        # 8-bit normalized depth: scale to 0.5 - 6.0 meters
        depth = (arr.astype(np.float32) / 255.0) * 5.5 + 0.5

    return depth.astype(np.float32)


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


@app.post("/api/reconstruct")
async def run_reconstruction(
    rgb_file: Optional[UploadFile] = None,
    depth_file: Optional[UploadFile] = None,
    demo_sample_id: Optional[str] = Form(None),
    method: str = Form("hybrid"),
    upsample_factor: int = Form(4),
    apply_texture_suppression: bool = Form(True)
):
    """
    Core Reconstruction Endpoint.
    Handles custom uploaded RGB+Depth or built-in demo samples.
    """
    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Load Inputs
    gt_depth = None
    max_eval_depth = 10.0

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
        else:
            raise HTTPException(status_code=400, detail="Unknown demo sample ID.")

        rgb_image = np.array(Image.open(rgb_p))
        depth_input = np.load(depth_p)
        if os.path.exists(gt_p):
            gt_depth = np.load(gt_p)

    else:
        if not (rgb_file and rgb_file.filename and depth_file and depth_file.filename):
            raise HTTPException(status_code=400, detail="Both RGB image and Depth Map are required.")

        rgb_bytes = await rgb_file.read()
        depth_bytes = await depth_file.read()

        try:
            rgb_image = np.array(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid RGB image file: {str(e)}")

        try:
            depth_input = parse_depth_bytes(depth_bytes, filename=depth_file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Depth Map file: {str(e)}")

    # 2. Validate Inputs
    if depth_input.size == 0 or np.all(depth_input <= 0):
        raise HTTPException(status_code=400, detail="Depth map contains no valid positive measurements.")

    # 3. Execute Reconstruction
    res = reconstruct_depth(
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

    # 4. Generate Visualizations
    rec_colored = colorize_depth(rec_depth)
    rec_b64 = encode_image_base64(rec_colored)

    # Classical baseline for visual comparison (Bilinear)
    classical_res = reconstruct_depth(rgb_image, depth_input, factor=upsample_factor, method="bilinear")
    classical_colored = colorize_depth(classical_res["reconstructed_depth"])
    classical_b64 = encode_image_base64(classical_colored)

    # Deep learning alone baseline
    cnn_res = reconstruct_depth(rgb_image, depth_input, factor=upsample_factor, method="cnn", cnn_model=MODEL, device=DEVICE)
    cnn_colored = colorize_depth(cnn_res["reconstructed_depth"])
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
        # Error map
        abs_err = np.abs(rec_depth - gt_depth)
        err_colored = cv2.applyColorMap(np.clip((abs_err / 0.5) * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        err_colored[gt_depth <= 0.05] = [15, 23, 42]
        error_b64 = encode_image_base64(cv2.cvtColor(err_colored, cv2.COLOR_BGR2RGB))

    total_latency = (time.perf_counter() - t_start) * 1000.0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Save latest reconstruction in memory cache for downloads
    app.state.last_rec = {
        "depth": rec_depth,
        "colored": rec_colored,
        "metrics": metrics,
        "method": method
    }

    return {
        "success": True,
        "method": method,
        "latency_ms": round(total_latency, 1),
        "peak_ram_mb": round(peak_mem / (1024 * 1024), 2),
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
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=edgefuse_dcr_reconstructed_depth.npy"}
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
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=edgefuse_dcr_depth_preview.png"}
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
