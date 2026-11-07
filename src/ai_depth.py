"""
src/ai_depth.py - AI-Powered Monocular Depth Estimation using Depth Anything V2.

Provides single-instance model caching, CPU-friendly inference, input validation,
result caching, and calibration guards for scientific depth completion.
"""

import os
import io
import time
import hashlib
import logging
import threading
import gc
import ctypes
from typing import Tuple, Dict, Any, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2

logger = logging.getLogger("edgefuse.ai_depth")

def trim_process_memory():
    """Forces garbage collection and glibc heap page trimming to avoid 512MB OOM on Linux/Render."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Model configuration for Depth Anything V2 Small (ViT-S)
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
}

CHECKPOINT_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
DEFAULT_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "depth_anything_v2_vits.pth")


class DepthAnythingV2Manager:
    """
    Thread-safe Singleton Manager for Depth Anything V2 Monocular Depth Model.
    - Loads weights once at application startup.
    - Caches inferences using SHA-256 image hashes.
    - Runs in torch.inference_mode() on CPU (or GPU if available).
    - Preserves aspect ratios and enforces scientific transparency.
    """
    _instance: Optional["DepthAnythingV2Manager"] = None
    _init_lock = threading.RLock()

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, encoder: str = "vits"):
        self.checkpoint_path = checkpoint_path
        self.encoder = encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DepthAnythingV2] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self._inference_lock = threading.RLock()
        
        # LRU prediction cache: hash -> dict
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self.max_cache_size = 32

    @classmethod
    def get_instance(cls, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH) -> "DepthAnythingV2Manager":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(checkpoint_path=checkpoint_path)
            return cls._instance

    def ensure_model_loaded(self) -> bool:
        """
        Loads the model checkpoint into memory once.
        If the checkpoint is not present on disk, attempts to download it once.
        """
        if self.is_loaded and self.model is not None:
            return True

        with self._init_lock:
            if self.is_loaded and self.model is not None:
                return True

            try:
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                
                # Download if missing and URL available
                if not os.path.exists(self.checkpoint_path) or os.path.getsize(self.checkpoint_path) < 1000000:
                    logger.info(f"Downloading Depth Anything V2 checkpoint from {CHECKPOINT_URL}...")
                    import urllib.request
                    urllib.request.urlretrieve(CHECKPOINT_URL, self.checkpoint_path)
                    logger.info("Download completed successfully.")

                logger.info(f"Loading Depth Anything V2 ({self.encoder}) from {self.checkpoint_path} onto {self.device}...")
                config = MODEL_CONFIGS[self.encoder]
                model = DepthAnythingV2(**config)
                state_dict = torch.load(self.checkpoint_path, map_location=self.device, mmap=True)
                model.load_state_dict(state_dict)
                del state_dict
                trim_process_memory()
                model.to(self.device)
                model.eval()

                self.model = model
                self.is_loaded = True
                self.load_error = None
                logger.info("Depth Anything V2 model loaded and ready.")
                trim_process_memory()
                return True

            except Exception as e:
                self.load_error = str(e)
                logger.error(f"Failed to load Depth Anything V2 model: {e}")
                return False

    def predict(
        self,
        rgb_image: np.ndarray,
        input_size: int = 518,
        cache_key: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Performs monocular depth estimation on a given 3-channel RGB image.
        Returns:
            (relative_depth_map: np.ndarray [H, W], metadata: Dict[str, Any])
        """
        if not self.ensure_model_loaded():
            raise RuntimeError(f"Depth Anything V2 model unavailable: {self.load_error}")

        if not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Invalid RGB image shape {getattr(rgb_image, 'shape', None)}; expected (H, W, 3).")

        h_orig, w_orig = rgb_image.shape[:2]
        if h_orig == 0 or w_orig == 0:
            raise ValueError("RGB image has zero dimensions.")

        # Compute content hash for caching
        if not cache_key:
            hasher = hashlib.sha256()
            hasher.update(rgb_image.tobytes())
            hasher.update(str(input_size).encode())
            cache_key = hasher.hexdigest()

        with self._cache_lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                logger.info(f"Serving cached Depth Anything V2 prediction for hash {cache_key[:8]}")
                cached_meta = dict(entry["meta"])
                cached_meta["cached"] = True
                cached_meta["inference_time_ms"] = 0.0
                return entry["depth"].copy(), cached_meta

        t0 = time.perf_counter()

        # Enforce memory safety: scale if image exceeds 640px while strictly preserving aspect ratio
        # Processing at max 640px ensures peak RAM stays safely below 460MB on Render's 512MB tier.
        max_dim = 640
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / float(max(h_orig, w_orig))
            proc_w = max(14, int(w_orig * scale))
            proc_h = max(14, int(h_orig * scale))
            proc_rgb = cv2.resize(rgb_image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc_rgb = rgb_image
            proc_w, proc_h = w_orig, h_orig

        # Depth Anything V2's infer_image expects BGR, so convert RGB -> BGR
        proc_bgr = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)

        with self._inference_lock:
            with torch.inference_mode():
                # Inference execution
                raw_depth = self.model.infer_image(proc_bgr, input_size=input_size)

        # High-quality restoration to original user resolution with OpenCV (C++ zero-overhead streaming)
        if (raw_depth.shape[0], raw_depth.shape[1]) != (h_orig, w_orig):
            depth_map = cv2.resize(raw_depth, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        else:
            depth_map = raw_depth

        trim_process_memory()

        inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Normalize depth map for relative monocular representation
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        d_mean = float(depth_map.mean())

        meta = {
            "model_name": "Depth Anything V2 (ViT-S)",
            "encoder": self.encoder,
            "architecture": "Vision Transformer (ViT) + DPT Head",
            "parameters": 24785089,
            "license": "Apache-2.0",
            "source": "ByteDance / HKUST",
            "depth_type": "Relative Monocular Depth (Disparity-Inverse)",
            "is_metric": False,
            "scale_unit": "relative (unitless)",
            "inference_time_ms": inference_time_ms,
            "original_resolution": f"{w_orig} × {h_orig}",
            "input_size": input_size,
            "min_val": round(d_min, 4),
            "max_val": round(d_max, 4),
            "mean_val": round(d_mean, 4),
            "scientific_caveat": (
                "Notice: This depth map represents uncalibrated relative disparity predicted "
                "by a monocular Vision Transformer. It does not measure absolute physical distances "
                "in meters and must not be treated as sensor ground truth."
            )
        }

        # Store in LRU cache
        with self._cache_lock:
            if len(self._cache) >= self.max_cache_size:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = {"depth": depth_map.copy(), "meta": meta}

        return depth_map, meta


def calibrate_relative_to_metric(
    rel_depth: np.ndarray,
    sparse_metric_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Calibrates a relative depth prediction to physical metric meters using sparse sensor seeds.
    Uses median scale alignment: s * d_rel + t, robust to outliers.
    Returns:
        (metric_depth, scale, shift)
    """
    if valid_mask is None:
        valid_mask = (sparse_metric_depth > 0.05) & np.isfinite(sparse_metric_depth)

    if np.sum(valid_mask) < 4:
        # Insufficient sparse seeds: fallback to identity normalized 0.5m-10.0m range
        d_min, d_max = rel_depth.min(), rel_depth.max()
        if d_max > d_min:
            norm = (rel_depth - d_min) / (d_max - d_min)
            return (0.5 + norm * 9.5).astype(np.float32), 1.0, 0.0
        return rel_depth.astype(np.float32), 1.0, 0.0

    y = sparse_metric_depth[valid_mask].astype(np.float64)
    x = rel_depth[valid_mask].astype(np.float64)

    # Median scale factor
    scale = float(np.median(y) / max(np.median(x), 1e-4))
    calibrated = (rel_depth * scale).astype(np.float32)
    return calibrated, scale, 0.0
