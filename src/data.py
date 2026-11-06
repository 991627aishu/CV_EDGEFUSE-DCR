"""
data.py - Dataset loaders, preprocessing, and degradation engine for EdgeFuse-DCR.
Supports:
  - NYU Depth V2 (HDF5 labeled dataset)
  - KITTI Depth Completion (RGB + sparse LiDAR velodyne + annotated GT)
  - Middlebury 2021 (PFM disparities + RGB + calibration)
"""

import os
import re
import h5py
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# NYU Depth V2 Loader
# ---------------------------------------------------------------------------
class NYUDepthV2Dataset(Dataset):
    """
    Loads RGB images and dense infilled depth maps from nyu_depth_v2_labeled.mat.
    Generates controlled low-resolution / sparse degraded inputs for training.
    """
    def __init__(self, mat_path, split="train", split_ratio=0.85, downsample_factor=4, crop_size=(240, 320), max_samples=None):
        super().__init__()
        self.mat_path = mat_path
        self.downsample_factor = downsample_factor
        self.crop_size = crop_size
        self.split = split
        self.h5_file = None

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"NYU Depth V2 file not found at: {mat_path}")

        with h5py.File(mat_path, "r") as f:
            total_samples = f["images"].shape[0]

        train_count = int(total_samples * split_ratio)
        if split == "train":
            self.indices = list(range(0, train_count))
        else:
            self.indices = list(range(train_count, total_samples))

        if max_samples and max_samples < len(self.indices):
            self.indices = self.indices[:max_samples]

    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.mat_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open_file()
        sample_idx = self.indices[idx]

        # MATLAB HDF5 arrays are stored (C, W, H) -> transpose to (H, W, C)
        raw_img = self.h5_file["images"][sample_idx]  # (3, 640, 480)
        img = np.transpose(raw_img, (2, 1, 0)).astype(np.float32) / 255.0  # (480, 640, 3)

        raw_depth = self.h5_file["depths"][sample_idx]  # (640, 480)
        depth = np.transpose(raw_depth, (1, 0)).astype(np.float32)  # (480, 640)

        # Random or center crop
        h, w = depth.shape
        ch, cw = self.crop_size
        if self.split == "train":
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
        else:
            top = (h - ch) // 2
            left = (w - cw) // 2

        img_crop = img[top:top+ch, left:left+cw]
        gt_depth = depth[top:top+ch, left:left+cw]

        # Generate low-resolution depth
        lr_h, lr_w = ch // self.downsample_factor, cw // self.downsample_factor
        lr_depth = cv2.resize(gt_depth, (lr_w, lr_h), interpolation=cv2.INTER_NEAREST)

        # Upsample back to target resolution for network input
        lr_depth_up = cv2.resize(lr_depth, (cw, ch), interpolation=cv2.INTER_NEAREST)

        # Validity mask (all valid in dense GT > 0)
        valid_mask = (lr_depth_up > 0.05).astype(np.float32)

        # Convert to torch tensors (B, C, H, W)
        rgb_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float()  # (3, H, W)
        lr_depth_tensor = torch.from_numpy(lr_depth_up).unsqueeze(0).float()  # (1, H, W)
        valid_mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).float()  # (1, H, W)
        gt_depth_tensor = torch.from_numpy(gt_depth).unsqueeze(0).float()  # (1, H, W)

        return {
            "rgb": rgb_tensor,
            "lr_depth": lr_depth_tensor,
            "valid_mask": valid_mask_tensor,
            "gt_depth": gt_depth_tensor
        }

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None


def load_nyu_sample(mat_path, index=0):
    """Loads a single uncropped NYU sample for inference/evaluation."""
    with h5py.File(mat_path, "r") as f:
        raw_img = f["images"][index]
        rgb = np.transpose(raw_img, (2, 1, 0))  # (480, 640, 3) uint8
        raw_gt = f["depths"][index]
        gt_depth = np.transpose(raw_gt, (1, 0)).astype(np.float32)  # (480, 640) float32 in meters
        raw_meas = f["rawDepths"][index]
        raw_depth = np.transpose(raw_meas, (1, 0)).astype(np.float32)  # (480, 640) float32 with holes
    return rgb, gt_depth, raw_depth


# ---------------------------------------------------------------------------
# KITTI Depth Completion Loader
# ---------------------------------------------------------------------------
def load_kitti_val_samples(kitti_val_dir, max_samples=50):
    """
    Loads synchronized validation samples from KITTI val_selection_cropped:
      - RGB image (uint8)
      - Sparse LiDAR depth (float32 meters, 0=invalid)
      - Ground truth depth (float32 meters, 0=invalid)
      - Intrinsics
    """
    img_dir = os.path.join(kitti_val_dir, "image")
    velo_dir = os.path.join(kitti_val_dir, "velodyne_raw")
    gt_dir = os.path.join(kitti_val_dir, "groundtruth_depth")
    intr_dir = os.path.join(kitti_val_dir, "intrinsics")

    filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    if max_samples:
        filenames = filenames[:max_samples]

    samples = []
    for f in filenames:
        img_p = os.path.join(img_dir, f)
        base = f.replace("sync_image_", "sync_velodyne_raw_")
        velo_p = os.path.join(velo_dir, base)
        if not os.path.exists(velo_p):
            velo_p = os.path.join(velo_dir, f)

        base_gt = f.replace("sync_image_", "sync_groundtruth_depth_")
        gt_p = os.path.join(gt_dir, base_gt)
        if not os.path.exists(gt_p):
            gt_p = os.path.join(gt_dir, f)

        intr_f = os.path.splitext(f)[0] + ".txt"
        intr_p = os.path.join(intr_dir, intr_f)

        if os.path.exists(img_p) and os.path.exists(velo_p) and os.path.exists(gt_p):
            samples.append({
                "filename": f,
                "image_path": img_p,
                "velo_path": velo_p,
                "gt_path": gt_p,
                "intrinsics_path": intr_p if os.path.exists(intr_p) else None
            })
    return samples


def read_kitti_sample(sample_meta):
    """Reads a KITTI sample: depth is uint16 png / 256.0 meters."""
    rgb = np.array(Image.open(sample_meta["image_path"]))
    velo_raw = np.array(Image.open(sample_meta["velo_path"]), dtype=np.float32) / 256.0
    gt_raw = np.array(Image.open(sample_meta["gt_path"]), dtype=np.float32) / 256.0

    velo_valid = velo_raw > 0.1
    gt_valid = gt_raw > 0.1

    return {
        "rgb": rgb,
        "sparse_depth": velo_raw,
        "sparse_valid": velo_valid,
        "gt_depth": gt_raw,
        "gt_valid": gt_valid,
        "filename": sample_meta["filename"]
    }


# ---------------------------------------------------------------------------
# Middlebury 2021 Loader
# ---------------------------------------------------------------------------
def read_pfm(file_path):
    """Reads a .pfm (Portable Float Map) file."""
    with open(file_path, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        channels = 3 if header == "PF" else 1 if header == "Pf" else None
        if not channels:
            raise ValueError(f"Not a valid PFM file: {file_path}")
        dim_line = f.readline().decode("utf-8").strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("utf-8").strip()
        w, h = map(int, dim_line.split())
        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        data = np.reshape(data, (h, w) if channels == 1 else (h, w, 3))
        data = np.flipud(data)
        return data, scale


def load_middlebury_scene(scene_dir):
    """
    Loads Middlebury scene:
    Converts disparity map to metric depth Z = (fx * baseline) / (disp + doffs).
    """
    calib_p = os.path.join(scene_dir, "calib.txt")
    disp0_p = os.path.join(scene_dir, "disp0.pfm")
    im0_p = os.path.join(scene_dir, "im0.png")

    if not (os.path.exists(calib_p) and os.path.exists(disp0_p) and os.path.exists(im0_p)):
        raise FileNotFoundError(f"Missing Middlebury files in {scene_dir}")

    # Parse calib.txt
    calib_dict = {}
    with open(calib_p, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                calib_dict[k.strip()] = v.strip()

    cam0_raw = calib_dict["cam0"].replace("[", "").replace("]", "").replace(";", " ")
    cam0 = np.fromstring(cam0_raw, sep=" ").reshape(3, 3)
    fx = cam0[0, 0]
    baseline = float(calib_dict["baseline"])  # in mm
    doffs = float(calib_dict.get("doffs", 0))

    rgb = np.array(Image.open(im0_p))
    disp, _ = read_pfm(disp0_p)

    # Disparity to metric depth (meters)
    # Z (meters) = (fx * (baseline / 1000.0)) / (disp + doffs)
    valid_disp = np.isfinite(disp) & (disp > 0)
    depth_m = np.zeros_like(disp, dtype=np.float32)
    depth_m[valid_disp] = (fx * (baseline / 1000.0)) / (disp[valid_disp] + doffs)

    return {
        "rgb": rgb,
        "disp": disp,
        "depth": depth_m,
        "valid_mask": valid_disp & (depth_m > 0.1) & (depth_m < 50.0),
        "fx": fx,
        "baseline_mm": baseline
    }


# ---------------------------------------------------------------------------
# Controlled Degradation Engine
# ---------------------------------------------------------------------------
def generate_controlled_input(dense_depth, factor=4, mode="downsample", sparsity=0.05, noise_std=0.0):
    """
    Generates controlled low-resolution, sparse, or noisy depth map from ground truth.
    Supports:
      - 'downsample': scale down by factor and return low-res depth
      - 'sparse': randomly sample points (e.g. 5% simulating LiDAR)
      - 'noise': add zero-mean Gaussian noise scaled by depth
    """
    h, w = dense_depth.shape
    valid_mask = (dense_depth > 0.05) & np.isfinite(dense_depth)
    clean_depth = np.where(valid_mask, dense_depth, 0.0).astype(np.float32)

    if mode == "downsample":
        lr_h = max(h // factor, 1)
        lr_w = max(w // factor, 1)
        # Low resolution map
        lr_depth = cv2.resize(clean_depth, (lr_w, lr_h), interpolation=cv2.INTER_NEAREST)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, lr_depth.shape).astype(np.float32)
            lr_depth = np.where(lr_depth > 0, np.maximum(0.1, lr_depth + noise), 0.0)
        return lr_depth, (lr_depth > 0)

    elif mode == "sparse":
        # Simulate sparse LiDAR scanlines / points
        sparse_depth = np.zeros_like(clean_depth)
        mask = (np.random.rand(h, w) < sparsity) & valid_mask
        sparse_depth[mask] = clean_depth[mask]
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, clean_depth.shape).astype(np.float32)
            sparse_depth[mask] = np.maximum(0.1, sparse_depth[mask] + noise[mask])
        return sparse_depth, mask

    else:
        # Default full resolution with holes / noise
        noisy_depth = clean_depth.copy()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, clean_depth.shape).astype(np.float32)
            noisy_depth[valid_mask] = np.maximum(0.1, noisy_depth[valid_mask] + noise[valid_mask])
        return noisy_depth, valid_mask
