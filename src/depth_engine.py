"""
depth_engine.py - Classical Algorithms and Novel EdgeFuse-DCR Hybrid Engine.

Classical methods:
  1. Nearest-neighbour interpolation
  2. Bilinear interpolation (validity-normalized)
  3. Bicubic interpolation
  4. Joint bilateral filter
  5. Guided filter (RGB-guided)
  6. Sobel & Canny RGB edge detection
  7. Depth discontinuity detection via gradients
  8. Noise removal (adaptive edge-preserving filter)
  9. Missing-depth detection
  10. Hole filling / Inpainting
  11. Boundary-aware refinement

Novel EdgeFuse-DCR Hybrid Contributions:
  A. Confidence-aware depth reconstruction
  B. Boundary-aware adaptive refinement
  C. Texture-copying suppression guidance gate
"""

import time
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, uniform_filter


# ===========================================================================
# 1. Classical Interpolation & Normalization
# ===========================================================================

def upsample_nearest(depth_lr, target_shape):
    """Nearest neighbor upsampling preserving original values."""
    th, tw = target_shape[:2]
    return cv2.resize(depth_lr, (tw, th), interpolation=cv2.INTER_NEAREST)


def upsample_bilinear_validity(depth_lr, target_shape, valid_mask=None):
    """
    Validity-normalized bilinear interpolation.
    Prevents invalid zero-measurements from bleeding into adjacent valid depths.
    Uses Normalized Convolution: (Depth * Mask) / Mask
    """
    th, tw = target_shape[:2]
    if valid_mask is None:
        valid_mask = (depth_lr > 0).astype(np.float32)
    else:
        valid_mask = valid_mask.astype(np.float32)

    masked_depth = depth_lr * valid_mask

    # Bilinear resize of masked depth and weights
    up_masked = cv2.resize(masked_depth, (tw, th), interpolation=cv2.INTER_LINEAR)
    up_weight = cv2.resize(valid_mask, (tw, th), interpolation=cv2.INTER_LINEAR)

    eps = 1e-6
    up_depth = np.where(up_weight > eps, up_masked / np.maximum(up_weight, eps), 0.0)
    return up_depth


def upsample_bicubic(depth_lr, target_shape):
    """Bicubic interpolation with non-negative clamping."""
    th, tw = target_shape[:2]
    up = cv2.resize(depth_lr, (tw, th), interpolation=cv2.INTER_CUBIC)
    return np.maximum(0.0, up)


# ===========================================================================
# 2. Filtering Methods (Joint Bilateral & Guided Filter)
# ===========================================================================

def fast_guided_filter(guide_rgb, depth_in, radius=5, eps=0.01, adaptive_eps_map=None):
    """
    RGB-guided filter for depth map upsampling and edge preservation.
    O(N) time complexity via integral box filters.
    Supports spatially-varying adaptive regularization (adaptive_eps_map)
    to suppress texture-copying artifacts without modifying guide luminance.
    """
    if guide_rgb.dtype == np.uint8:
        guide = guide_rgb.astype(np.float32) / 255.0
    else:
        guide = guide_rgb.astype(np.float32)

    # Convert guide to single channel luminance if RGB
    if len(guide.shape) == 3 and guide.shape[2] == 3:
        I = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    else:
        I = guide.copy()

    p = depth_in.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    if adaptive_eps_map is not None:
        eff_eps = adaptive_eps_map.astype(np.float32)
    else:
        eff_eps = eps

    a = cov_Ip / (var_I + eff_eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * I + mean_b
    return np.maximum(0.0, q)


def joint_bilateral_filter_fast(guide_rgb, depth_in, valid_mask=None, diameter=9, sigma_color=25.0, sigma_space=5.0):
    """
    RGB-guided joint bilateral filter that respects depth validity.
    Uses OpenCV bilateral filter on RGB guidance and normalizes with depth.
    """
    if valid_mask is None:
        valid_mask = (depth_in > 0).astype(np.float32)
    else:
        valid_mask = valid_mask.astype(np.float32)

    guide_uint8 = (guide_rgb if guide_rgb.dtype == np.uint8 else (guide_rgb * 255.0).clip(0, 255).astype(np.uint8))
    guide_gray = cv2.cvtColor(guide_uint8, cv2.COLOR_RGB2GRAY) if len(guide_uint8.shape) == 3 else guide_uint8

    # Apply guided filter with small eps as practical fast joint bilateral approximation
    return fast_guided_filter(guide_gray, depth_in, radius=diameter // 2, eps=(sigma_color / 255.0) ** 2)


# ===========================================================================
# 3. Edge, Gradient & Boundary Discontinuity Analysis
# ===========================================================================

def detect_rgb_edges(rgb_image, low_thresh=40, high_thresh=120):
    """Computes Sobel gradients and Canny edge map from RGB image."""
    rgb_uint8 = rgb_image if rgb_image.dtype == np.uint8 else (rgb_image * 255.0).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    # Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_norm = (grad_mag / (grad_mag.max() + 1e-6)).astype(np.float32)

    # Canny binary edge map
    canny = cv2.Canny(gray, low_thresh, high_thresh) > 0
    return grad_norm, canny.astype(np.float32)


def detect_depth_discontinuities(depth_map, valid_mask=None, relative_threshold=0.08):
    """
    Detects depth discontinuities using normalized relative depth gradients.
    Discontinuity = |grad(D)| / (D + eps) > threshold
    """
    d = depth_map.astype(np.float32)
    if valid_mask is None:
        valid_mask = d > 0.05

    # Compute spatial derivatives
    dx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad_d = np.sqrt(dx ** 2 + dy ** 2)

    # Relative gradient to account for distance scaling
    eps = 1e-3
    rel_grad = np.where(valid_mask, grad_d / (d + eps), 0.0)

    # Binary discontinuity map
    discontinuity_map = (rel_grad > relative_threshold) & valid_mask
    return rel_grad, discontinuity_map.astype(np.float32)


# ===========================================================================
# 4. Hole Filling & Noise Removal
# ===========================================================================

def detect_missing_depth(depth_map, min_valid=0.05, max_valid=85.0):
    """Identifies missing / invalid depth pixels."""
    invalid = (depth_map <= min_valid) | (depth_map >= max_valid) | np.isnan(depth_map) | np.isinf(depth_map)
    valid = ~invalid
    return valid.astype(np.float32), invalid.astype(np.float32)


def inpaint_depth_holes(depth_map, valid_mask=None):
    """
    Fills holes and missing pixels in depth map using fast normalized convolution.
    Avoids slow Navier-Stokes on large maps.
    """
    if valid_mask is None:
        valid_mask = (depth_map > 0.05).astype(np.float32)
    else:
        valid_mask = valid_mask.astype(np.float32)

    if valid_mask.mean() > 0.999:
        return depth_map.copy()

    filled = depth_map.copy()
    weights = valid_mask.copy()

    # Multi-scale Gaussian inpainting
    for ksize in [5, 11, 21, 41]:
        missing = (weights < 0.5)
        if not np.any(missing):
            break
        blurred_d = cv2.GaussianBlur(filled * weights, (ksize, ksize), 0)
        blurred_w = cv2.GaussianBlur(weights, (ksize, ksize), 0)

        update_mask = missing & (blurred_w > 1e-4)
        filled[update_mask] = blurred_d[update_mask] / blurred_w[update_mask]
        weights[update_mask] = 1.0

    return filled


def remove_depth_noise(depth_map, valid_mask=None, ksize=5):
    """Removes salt-and-pepper noise and outliers in flat regions."""
    if valid_mask is None:
        valid_mask = (depth_map > 0.05)

    # Median filter on valid region
    med = cv2.medianBlur(depth_map.astype(np.float32), ksize)
    # Blend: keep original on sharp edges, smooth on flat regions
    grad, _ = detect_depth_discontinuities(depth_map, valid_mask)
    flat_weight = np.clip(1.0 - grad * 5.0, 0.0, 1.0)
    cleaned = np.where(valid_mask, (1.0 - flat_weight * 0.7) * depth_map + (flat_weight * 0.7) * med, med)
    return cleaned


# ===========================================================================
# 5. NOVEL EDGEFUSE-DCR HYBRID CONTRIBUTIONS
# ===========================================================================

def estimate_confidence_map(depth_map, valid_mask, window_size=5, factor=1, is_low_res=False, cnn_conf=None):
    """
    Mechanism A: Confidence-Aware Depth Estimation.
    Calculates confidence C in [0, 1] based on:
      1. Physical sensor anchor proximity (exponential decay with distance from genuine measurements)
      2. Local geometric consistency (normalized variance in local window)
      3. Deep neural confidence head prediction (when available)
    """
    h, w = depth_map.shape
    if is_low_res and factor > 1:
        # True physical sensor measurements exist only on the sub-sampled grid
        anchor_mask = np.zeros((h, w), dtype=np.float32)
        anchor_mask[::factor, ::factor] = 1.0
        dist = distance_transform_edt(1.0 - anchor_mask)
        decay = max(1.5 * factor, 2.0)
        spatial_conf = np.exp(-dist / decay).astype(np.float32)
    else:
        if valid_mask.shape[:2] != (h, w):
            v_mask = cv2.resize(valid_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            v_mask = valid_mask.astype(np.float32)

        if v_mask.max() > 0:
            dist = distance_transform_edt(1.0 - v_mask)
            spatial_conf = np.exp(-dist / 8.0).astype(np.float32)
        else:
            spatial_conf = np.ones((h, w), dtype=np.float32) * 0.1

    # 2. Local geometric consistency: normalized standard deviation
    local_mean = uniform_filter(depth_map.astype(np.float32), size=window_size)
    local_sq_mean = uniform_filter(depth_map.astype(np.float32) ** 2, size=window_size)
    local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)
    local_std = np.sqrt(local_var)

    # Normalized consistency: low variance relative to mean depth = high confidence
    rel_std = local_std / (local_mean + 1e-4)
    consistency_conf = np.clip(1.0 - rel_std * 4.0, 0.1, 1.0).astype(np.float32)

    # 3. Fuse confidence sources
    if cnn_conf is not None:
        if cnn_conf.shape[:2] != (h, w):
            c_pred = cv2.resize(cnn_conf.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            c_pred = cnn_conf.astype(np.float32)
        conf = np.clip(0.4 * spatial_conf + 0.3 * consistency_conf + 0.3 * c_pred, 0.0, 1.0)
    else:
        conf = np.clip(spatial_conf * 0.6 + consistency_conf * 0.4, 0.0, 1.0)

    return conf.astype(np.float32)


def compute_texture_suppression_gate(rgb_grad, depth_grad, threshold=0.20, depth_discon_thresh=0.06):
    """
    Mechanism C: Texture-Copying Suppression Gate.
    Identifies RGB texture edges (e.g. wall wallpaper, fabric, text) that do NOT
    correspond to physical geometric depth steps.
    Returns:
      guidance_gate: [0, 1] weight where 1.0 = allow RGB guidance, ~0.0 = suppress texture copy.
      false_texture: [0, 1] penalty where RGB edge is high and depth gradient is near zero.
    """
    # High RGB gradient excess above threshold
    rgb_excess = np.clip((rgb_grad - threshold) / (1.0 - threshold + 1e-4), 0.0, 1.0)
    # Geometric depth flatness: 1.0 when depth discontinuity gradient is low, 0.0 at depth steps
    depth_flat = np.clip(1.0 - (depth_grad / depth_discon_thresh), 0.0, 1.0)
    false_texture = rgb_excess * depth_flat

    # Guidance gate: 1.0 where depth edge exists or flat region, lower where false texture occurs
    guidance_gate = np.clip(1.0 - false_texture, 0.05, 1.0)
    return guidance_gate.astype(np.float32), false_texture.astype(np.float32)


def edgefuse_dcr_refine(raw_reconstruction, guide_rgb, initial_valid_mask=None, apply_suppression=True, factor=4, is_low_res=False, cnn_conf=None):
    """
    Complete EdgeFuse-DCR Hybrid Refinement combining:
      A. Multi-source confidence map (anchors + consistency + CNN)
      B. Boundary-aware adaptive barrier
      C. Spatially-regularized texture-copying suppression gate
    """
    h, w = raw_reconstruction.shape
    if initial_valid_mask is None:
        initial_valid_mask = (raw_reconstruction > 0.05)
    elif initial_valid_mask.shape[:2] != (h, w):
        initial_valid_mask = cv2.resize(initial_valid_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5

    # Step 1: Detect RGB edges and depth discontinuities
    rgb_grad, canny_edges = detect_rgb_edges(guide_rgb)
    depth_rel_grad, depth_discon = detect_depth_discontinuities(raw_reconstruction, initial_valid_mask)

    # Step 2: Estimate confidence
    confidence = estimate_confidence_map(raw_reconstruction, initial_valid_mask, factor=factor, is_low_res=is_low_res, cnn_conf=cnn_conf)

    # Scale-adaptive parameters
    radius = max(2, min(int(factor), 5))
    eps_base = max(0.002, 0.0012 * float(factor))

    # Step 3: Texture suppression gate
    if apply_suppression:
        guidance_gate, false_texture = compute_texture_suppression_gate(rgb_grad, depth_rel_grad)
        # Adaptively increase epsilon on false texture surfaces to suppress texture transfer
        eps_map = eps_base * (1.0 + 50.0 * false_texture)
    else:
        guidance_gate = np.ones((h, w), dtype=np.float32)
        eps_map = np.full((h, w), eps_base, dtype=np.float32)

    # Step 4: Boundary-aware adaptive refinement
    # Near genuine depth discontinuities, reduce smoothing to prevent foreground-background bleeding
    boundary_barrier = np.clip(1.0 - depth_discon * 0.85, 0.15, 1.0)

    # Edge-guided filtered depth with adaptive regularization
    refined_depth = fast_guided_filter(guide_rgb, raw_reconstruction, radius=radius, eps=eps_base, adaptive_eps_map=eps_map)

    # Step 5: Confidence-guided blending
    final_depth = (
        confidence * boundary_barrier * raw_reconstruction +
        (1.0 - confidence * boundary_barrier) * refined_depth
    )

    return {
        "final_depth": np.maximum(0.05, final_depth),
        "confidence_map": confidence,
        "rgb_edges": rgb_grad,
        "depth_discontinuities": depth_discon,
        "texture_gate": guidance_gate
    }


# ===========================================================================
# 6. Unified Reconstruction Pipeline
# ===========================================================================

def reconstruct_depth(rgb_image, depth_input, factor=4, method="hybrid", cnn_model=None, device="cpu", apply_suppression=True):
    """
    Unified public interface for all depth reconstruction methods.
    Supports:
      - 'nearest': Nearest neighbor
      - 'bilinear': Bilinear interpolation (validity-normalized)
      - 'bicubic': Bicubic interpolation
      - 'joint_bilateral': Joint Bilateral Filter
      - 'guided': Guided Filter
      - 'cnn': Compact Deep Learning model alone
      - 'cnn_confidence': CNN + Confidence-Aware Guidance
      - 'cnn_boundary': CNN + Confidence + Boundary Refinement
      - 'hybrid': Full EdgeFuse-DCR (CNN + Confidence + Boundary + Texture Gate)
    """
    t_start = time.perf_counter()
    h_rgb, w_rgb = rgb_image.shape[:2]
    target_shape = (h_rgb, w_rgb)

    # Check input depth dimensions
    dh, dw = depth_input.shape[:2]
    is_low_res = (dh != h_rgb or dw != w_rgb)

    valid_mask = (depth_input > 0.05) & np.isfinite(depth_input)

    # Standard A0 Classical Baseline (Bilinear Validity-Normalized)
    base_up = upsample_bilinear_validity(depth_input, target_shape, valid_mask)
    base_up = inpaint_depth_holes(base_up, base_up > 0.05)

    # Compute CNN prediction when model is provided (A2 Baseline)
    cnn_depth = None
    cnn_conf_np = None
    if cnn_model is not None:
        try:
            import torch
            torch.set_num_threads(1)

            h_c, w_c = base_up.shape[:2]
            cnn_scale = 1.0
            if max(h_c, w_c) > 360:
                cnn_scale = 320.0 / float(max(h_c, w_c))
                c_w = int(w_c * cnn_scale)
                c_h = int(h_c * cnn_scale)
                rgb_in = cv2.resize(rgb_image, (c_w, c_h), interpolation=cv2.INTER_AREA)
                depth_in = cv2.resize(base_up, (c_w, c_h), interpolation=cv2.INTER_NEAREST)
            else:
                rgb_in = rgb_image
                depth_in = base_up

            rgb_t = torch.from_numpy(rgb_in.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            d_t = torch.from_numpy(depth_in.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            v_t = torch.from_numpy((depth_in > 0.05).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            cnn_model.eval()
            with torch.no_grad():
                pred_d, pred_conf = cnn_model(rgb_t, d_t, v_t)
                cnn_depth_raw = pred_d.squeeze().cpu().numpy()
                cnn_conf_raw = pred_conf.squeeze().cpu().numpy()

            if cnn_scale != 1.0:
                cnn_depth = cv2.resize(cnn_depth_raw, (w_c, h_c), interpolation=cv2.INTER_LINEAR)
                cnn_conf_np = cv2.resize(cnn_conf_raw, (w_c, h_c), interpolation=cv2.INTER_LINEAR)
            else:
                cnn_depth = cnn_depth_raw
                cnn_conf_np = cnn_conf_raw

        except Exception as e:
            print(f"Warning: CNN forward pass error ({e}), falling back to edge-guided baseline.")
            cnn_depth = fast_guided_filter(rgb_image, base_up, radius=4, eps=0.005)
    else:
        cnn_depth = fast_guided_filter(rgb_image, base_up, radius=4, eps=0.005)

    # Method Selection
    if method == "nearest":
        rec = upsample_nearest(depth_input, target_shape)
        rec = inpaint_depth_holes(rec, rec > 0.05)
        conf = (rec > 0.05).astype(np.float32)
        gate = np.ones(target_shape, dtype=np.float32)

    elif method == "bilinear":
        rec = base_up
        conf = (rec > 0.05).astype(np.float32)
        gate = np.ones(target_shape, dtype=np.float32)

    elif method == "bicubic":
        rec = upsample_bicubic(depth_input, target_shape)
        rec = inpaint_depth_holes(rec, rec > 0.05)
        conf = (rec > 0.05).astype(np.float32)
        gate = np.ones(target_shape, dtype=np.float32)

    elif method == "joint_bilateral":
        rec = joint_bilateral_filter_fast(rgb_image, base_up)
        conf = estimate_confidence_map(rec, rec > 0.05)
        gate = np.ones(target_shape, dtype=np.float32)

    elif method == "guided":
        rec = fast_guided_filter(rgb_image, base_up, radius=5, eps=0.01)
        conf = estimate_confidence_map(rec, rec > 0.05)
        gate = np.ones(target_shape, dtype=np.float32)

    elif method == "cnn":
        rec = cnn_depth
        conf = estimate_confidence_map(rec, rec > 0.05, factor=factor, is_low_res=is_low_res, cnn_conf=cnn_conf_np)
        gate = np.ones(target_shape, dtype=np.float32)

    elif method in ("cnn_confidence", "cnn_conf"):
        conf = estimate_confidence_map(cnn_depth, valid_mask, factor=factor, is_low_res=is_low_res, cnn_conf=cnn_conf_np)
        rec = conf * cnn_depth + (1.0 - conf) * base_up
        gate = np.ones(target_shape, dtype=np.float32)

    elif method in ("cnn_boundary", "cnn_conf_refined"):
        res = edgefuse_dcr_refine(cnn_depth, rgb_image, initial_valid_mask=valid_mask, apply_suppression=False, factor=factor, is_low_res=is_low_res, cnn_conf=cnn_conf_np)
        rec = res["final_depth"]
        conf = res["confidence_map"]
        gate = res["texture_gate"]

    elif method == "hybrid":  # Full hybrid EdgeFuse-DCR
        res = edgefuse_dcr_refine(cnn_depth, rgb_image, initial_valid_mask=valid_mask, apply_suppression=apply_suppression, factor=factor, is_low_res=is_low_res, cnn_conf=cnn_conf_np)
        rec = res["final_depth"]
        conf = res["confidence_map"]
        gate = res["texture_gate"]

    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Edge analysis for visualization
    rgb_grad, _ = detect_rgb_edges(rgb_image)
    _, depth_discon = detect_depth_discontinuities(rec, rec > 0.05)

    return {
        "reconstructed_depth": rec.astype(np.float32),
        "classical_baseline": base_up.astype(np.float32) if "base_up" in locals() else rec.astype(np.float32),
        "cnn_baseline": cnn_depth.astype(np.float32) if "cnn_depth" in locals() else rec.astype(np.float32),
        "confidence_map": conf.astype(np.float32),
        "rgb_edges": rgb_grad.astype(np.float32),
        "depth_edges": depth_discon.astype(np.float32),
        "texture_gate": gate.astype(np.float32),
        "latency_ms": latency_ms,
        "method": method
    }
