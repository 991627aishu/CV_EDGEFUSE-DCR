"""
Compute authentic scene benchmarks and dual thumbnails for all 9 scenes.
Updates demo_samples/manifest.json.
"""

import os
import sys
import io
import json
import base64
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import MODEL, DEVICE, colorize_depth
from src.depth_engine import reconstruct_depth
from src.evaluation import compute_depth_metrics

DEMO_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "demo_samples")
MANIFEST_PATH = os.path.join(DEMO_DIR, "manifest.json")

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

def make_thumb_b64(img_arr, is_depth=False, valid_mask=None, max_depth=10.0):
    if is_depth:
        colored = colorize_depth(img_arr, valid_mask=valid_mask, vmax=max_depth)
        pil = Image.fromarray(colored)
    else:
        pil = Image.fromarray(img_arr)
    # Downscale to 90x60 thumbnail
    pil.thumbnail((90, 60), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    if is_depth:
        pil.save(buf, format="PNG")
        mime = "image/png"
    else:
        pil.save(buf, format="JPEG", quality=85)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

algorithms = [
    ("A0: Nearest Neighbor", "nearest"),
    ("A0: Bilinear Interpolation", "bilinear"),
    ("A0: Bicubic Interpolation", "bicubic"),
    ("A1: Joint Bilateral Filter", "joint_bilateral"),
    ("A1: Guided Filter", "guided"),
    ("A2: Deep Learning (EdgeFuseNet Alone)", "cnn"),
    ("A3: CNN + Confidence Guidance", "cnn_confidence"),
    ("A4: CNN + Conf + Boundary Refinement", "cnn_boundary"),
    ("A5: EdgeFuse-DCR Full Hybrid (Ours)", "hybrid")
]

for dkey, dinfo in manifest["datasets"].items():
    max_d = dinfo.get("max_eval_depth", 10.0)
    for sample in dinfo["samples"]:
        sid = sample["id"]
        print(f"\nProcessing {sid} ({sample['title']})...")
        
        rgb_path = os.path.join(DEMO_DIR, sample["rgb_file"])
        d_path = os.path.join(DEMO_DIR, sample["input_file"])
        gt_path = os.path.join(DEMO_DIR, sample["gt_file"])
        
        rgb = np.array(Image.open(rgb_path))
        depth_lr = np.load(d_path)
        gt = np.load(gt_path)
        
        # 1. Dual Thumbnails
        v_mask = (depth_lr > 0.05) & np.isfinite(depth_lr)
        sample["rgb_thumb_b64"] = make_thumb_b64(rgb, is_depth=False)
        sample["depth_thumb_b64"] = make_thumb_b64(depth_lr, is_depth=True, valid_mask=v_mask, max_depth=max_d)
        
        # 2. Benchmarks for all algorithms
        bench_rows = []
        best_rmse = float("inf")
        best_psnr = -1.0
        
        for name, m_key in algorithms:
            res = reconstruct_depth(rgb, depth_lr, factor=4, method=m_key, cnn_model=MODEL, device=DEVICE, apply_suppression=True)
            rec = res["reconstructed_depth"]
            lat = res["latency_ms"]
            
            # Evaluate against GT
            eval_mask = (gt > 0.05) & np.isfinite(gt) & (rec > 0.05) & np.isfinite(rec)
            met = compute_depth_metrics(rec, gt, valid_mask=eval_mask, max_depth=max_d)
            
            bench_rows.append({
                "method": name,
                "key": m_key,
                "rmse": f"{met['rmse']:.4f}",
                "mae": f"{met['mae']:.4f}",
                "psnr": f"{met['psnr']:.2f}",
                "ssim": f"{met['ssim']:.4f}",
                "f1": f"{met['boundary_f1']:.4f}",
                "grad": f"{met['grad_mae']:.4f}",
                "latency": f"{lat:.1f} ms",
                "highlight": (m_key == "hybrid")
            })
            print(f"  {m_key:15s} -> RMSE: {met['rmse']:.4f}m, PSNR: {met['psnr']:.2f}dB, SSIM: {met['ssim']:.4f}, Latency: {lat:.1f}ms")
            
        sample["benchmark_rows"] = bench_rows

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nManifest successfully updated with authentic dual thumbnails and scene benchmarks!")

