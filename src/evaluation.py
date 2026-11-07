"""
evaluation.py - Evaluation Engine & Ablation Study for EdgeFuse-DCR.

Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - Boundary Precision, Recall, F1 (Depth Discontinuity Alignment)
  - Depth-Gradient Error (Sobel gradient MAE)
  - Inference Latency (ms)
  - Memory Peak (MB)

Ablation Levels:
  A0: Nearest, Bilinear, Bicubic
  A1: Joint Bilateral, Guided Filter
  A2: CNN Model Alone
  A3: CNN + Confidence-Aware Correction
  A4: CNN + Confidence + Boundary-Aware Refinement
  A5: Full EdgeFuse-DCR (CNN + Confidence + Boundary + Texture Suppression)
"""

import time
import tracemalloc
import numpy as np
import cv2


# ===========================================================================
# 1. Metric Calculations
# ===========================================================================

def compute_depth_metrics(pred_depth, gt_depth, valid_mask=None, max_depth=10.0):
    """
    Computes standard depth reconstruction metrics on valid ground-truth pixels.
    """
    if valid_mask is None:
        mask = (gt_depth > 0.05) & (gt_depth <= max_depth) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    else:
        mask = valid_mask & (gt_depth > 0.05) & (gt_depth <= max_depth) & np.isfinite(gt_depth) & np.isfinite(pred_depth)

    if not np.any(mask):
        return {
            "rmse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0,
            "boundary_p": 0.0, "boundary_r": 0.0, "boundary_f1": 0.0,
            "grad_mae": 0.0, "valid_pixels": 0
        }

    pred_v = pred_depth[mask]
    gt_v = gt_depth[mask]

    # Clean non-finite values for spatial operations (Gaussian blur, Sobel)
    pred_clean = np.nan_to_num(pred_depth, nan=0.0, posinf=max_depth, neginf=0.0)
    gt_clean = np.nan_to_num(gt_depth, nan=0.0, posinf=max_depth, neginf=0.0)

    # 1. MAE & RMSE
    diff = pred_v - gt_v
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # 2. PSNR
    mse = float(np.mean(diff ** 2))
    psnr = float(20.0 * np.log10(max_depth) - 10.0 * np.log10(max(1e-8, mse)))

    # 3. SSIM (on cropped normalized valid box)
    # Fast SSIM calculation over valid region
    norm_pred = np.clip(pred_clean / max_depth, 0.0, 1.0)
    norm_gt = np.clip(gt_clean / max_depth, 0.0, 1.0)

    u_pred = cv2.GaussianBlur(norm_pred, (11, 11), 1.5)
    u_gt = cv2.GaussianBlur(norm_gt, (11, 11), 1.5)

    u_pred_sq = u_pred ** 2
    u_gt_sq = u_gt ** 2
    u_pred_gt = u_pred * u_gt

    sig_pred_sq = cv2.GaussianBlur(norm_pred ** 2, (11, 11), 1.5) - u_pred_sq
    sig_gt_sq = cv2.GaussianBlur(norm_gt ** 2, (11, 11), 1.5) - u_gt_sq
    sig_pred_gt = cv2.GaussianBlur(norm_pred * norm_gt, (11, 11), 1.5) - u_pred_gt

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    denom = (u_pred_sq + u_gt_sq + c1) * (sig_pred_sq + sig_gt_sq + c2)
    ssim_map = np.where(denom > 1e-9, ((2 * u_pred_gt + c1) * (2 * sig_pred_gt + c2)) / denom, 1.0)
    ssim_valid = ssim_map[mask]
    ssim = float(np.mean(ssim_valid)) if len(ssim_valid) > 0 else 0.0

    # 4. Depth-Gradient MAE (Sobel)
    sobel_px = cv2.Sobel(pred_clean, cv2.CV_32F, 1, 0, ksize=3)
    sobel_py = cv2.Sobel(pred_clean, cv2.CV_32F, 0, 1, ksize=3)
    grad_pred = np.sqrt(sobel_px ** 2 + sobel_py ** 2)

    sobel_gx = cv2.Sobel(gt_clean, cv2.CV_32F, 1, 0, ksize=3)
    sobel_gy = cv2.Sobel(gt_clean, cv2.CV_32F, 0, 1, ksize=3)
    grad_gt = np.sqrt(sobel_gx ** 2 + sobel_gy ** 2)

    grad_diff = np.abs(grad_pred[mask] - grad_gt[mask])
    grad_mae = float(np.mean(grad_diff)) if len(grad_diff) > 0 else 0.0

    # 5. Boundary Precision, Recall, and F1
    # Threshold relative depth gradient to find boundaries
    rel_gt_grad = np.where(mask, grad_gt / (gt_clean + 1e-3), 0.0)
    rel_pred_grad = np.where(mask, grad_pred / (pred_clean + 1e-3), 0.0)

    gt_edges = (rel_gt_grad > 0.08) & mask
    pred_edges = (rel_pred_grad > 0.08) & mask

    # Morphological dilation for 1-pixel tolerance boundary evaluation
    kernel = np.ones((3, 3), np.uint8)
    gt_edges_dilated = cv2.dilate(gt_edges.astype(np.uint8), kernel) > 0
    pred_edges_dilated = cv2.dilate(pred_edges.astype(np.uint8), kernel) > 0

    tp_precision = np.sum(pred_edges & gt_edges_dilated)
    total_pred = np.sum(pred_edges)
    precision = float(tp_precision / (total_pred + 1e-6))

    tp_recall = np.sum(gt_edges & pred_edges_dilated)
    total_gt = np.sum(gt_edges)
    recall = float(tp_recall / (total_gt + 1e-6))

    f1 = float(2.0 * precision * recall / (precision + recall + 1e-6))

    def _safe(v, d=0.0):
        return d if (v is None or not np.isfinite(v)) else float(v)

    return {
        "rmse": _safe(rmse),
        "mae": _safe(mae),
        "psnr": _safe(psnr),
        "ssim": _safe(ssim),
        "boundary_p": _safe(precision),
        "boundary_r": _safe(recall),
        "boundary_f1": _safe(f1),
        "grad_mae": _safe(grad_mae),
        "valid_pixels": int(np.sum(mask))
    }


# ===========================================================================
# 2. Automated Benchmark & Ablation Study Runner
# ===========================================================================

def run_comparative_benchmark(rgb_image, gt_depth, factor=4, model=None, device="cpu", max_depth=10.0):
    """
    Executes all comparison levels (A0, A1, A2, A3, A4, A5) on identical input.
    """
    from .depth_engine import reconstruct_depth
    from .data import generate_controlled_input

    # Generate controlled low-resolution depth
    lr_depth, _ = generate_controlled_input(gt_depth, factor=factor, mode="downsample")

    methods_to_test = [
        ("A0-Nearest", "nearest"),
        ("A0-Bilinear", "bilinear"),
        ("A0-Bicubic", "bicubic"),
        ("A1-JointBilateral", "joint_bilateral"),
        ("A1-GuidedFilter", "guided"),
        ("A2-CNN-Alone", "cnn"),
        ("A3-CNN+Confidence", "cnn_confidence"),
        ("A4-CNN+Conf+Boundary", "cnn_boundary"),
        ("A5-EdgeFuse-DCR", "hybrid"),
    ]

    results = []

    for label, method_name in methods_to_test:
        tracemalloc.start()
        t0 = time.perf_counter()

        res = reconstruct_depth(rgb_image, lr_depth, factor=factor, method=method_name, cnn_model=model, device=device)

        latency = (time.perf_counter() - t0) * 1000.0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)

        metrics = compute_depth_metrics(res["reconstructed_depth"], gt_depth, max_depth=max_depth)
        metrics["label"] = label
        metrics["method"] = method_name
        metrics["latency_ms"] = latency
        metrics["peak_mb"] = peak_mb

        results.append(metrics)

    return results


def format_evaluation_table_markdown(benchmark_results, team_usns="1RUA24CSE0243, 1RUA24CSE0263, 1RUA24CSE0270, 1RUA24CSE0308"):
    """
    Formats the benchmark results into the official Markdown table format.
    First line contains team USNs.
    """
    lines = [
        team_usns,
        "# EdgeFuse-DCR: Comprehensive Comparative Evaluation",
        "",
        "| Pipeline Stage / Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Boundary F1 | Grad MAE | Latency (ms) | Peak RAM (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in benchmark_results:
        line = (
            f"| **{r['label']}** "
            f"| {r['rmse']:.4f} "
            f"| {r['mae']:.4f} "
            f"| {r['psnr']:.2f} "
            f"| {r['ssim']:.4f} "
            f"| {r['boundary_f1']:.4f} "
            f"| {r['grad_mae']:.4f} "
            f"| {r['latency_ms']:.1f} "
            f"| {r['peak_mb']:.2f} |"
        )
        lines.append(line)

    return "\n".join(lines)

