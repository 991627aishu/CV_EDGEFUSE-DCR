"""
model.py - Compact RGB-Guided Depth Reconstruction Network (EdgeFuseNet).

Architecture:
  - Lightweight Multi-Scale Residual Encoder-Decoder with Skip Connections.
  - Inputs: RGB (3 channels) + Depth (1 channel) + Validity Mask (1 channel) = 5 channels.
  - Multi-head output:
      1. Reconstructed Dense Depth Map (Residual formulation)
      2. Learned Confidence Map (Sigmoid activation)
  - Loss: Masked L1 Loss + Edge-Preserving Gradient Loss (Sobel depth loss).
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


# ---------------------------------------------------------------------------
# EdgeFuseNet Architecture
# ---------------------------------------------------------------------------
class EdgeFuseNet(nn.Module):
    """
    Compact RGB-guided Depth Reconstruction Network.
    Params: 1,308,610 trainable parameters (1,311,508 total including BatchNorm buffers).
    Fast CPU/GPU inference (~15ms on GPU, ~40ms on CPU).
    """
    def __init__(self, in_channels=5, base_channels=32):
        super().__init__()
        # Input stem: 5 channels (RGB + Depth + Mask)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels)
        )

        # Down 1
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels * 2)
        )

        # Down 2
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels * 4)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )

        # Up 1
        self.upconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.upblock1 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels * 2)
        )

        # Up 2
        self.upconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.upblock2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels)
        )

        # Depth output head (Residual prediction)
        self.depth_head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        # Confidence output head
        self.conf_head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, rgb, depth, mask):
        """
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W)
        mask: (B, 1, H, W)
        """
        x_in = torch.cat([rgb, depth, mask], dim=1)

        # Encoder
        x0 = self.stem(x_in)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        # Bottleneck
        b = self.bottleneck(x2)

        # Decoder with skip connections
        u1 = self.upconv1(b)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.upblock1(u1)

        u2 = self.upconv2(u1)
        u2 = torch.cat([u2, x0], dim=1)
        u2 = self.upblock2(u2)

        # Residual depth prediction
        delta_depth = self.depth_head(u2)
        pred_depth = F.relu(depth + delta_depth)

        pred_conf = self.conf_head(u2)
        return pred_depth, pred_conf


# ---------------------------------------------------------------------------
# Training Loss Functions
# ---------------------------------------------------------------------------
class EdgeAwareDepthLoss(nn.Module):
    """
    Combines:
      1. Masked L1 reconstruction loss
      2. Edge-aware gradient loss in X and Y
    """
    def __init__(self, alpha_l1=1.0, alpha_grad=0.25):
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_grad = alpha_grad

        # Sobel kernels for gradient computation
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred_depth, gt_depth, valid_mask):
        mask = (gt_depth > 0.05) & (valid_mask > 0.5)
        num_valid = mask.sum().clamp(min=1.0)

        # 1. Masked L1 loss
        l1_diff = torch.abs(pred_depth - gt_depth) * mask
        l1_loss = l1_diff.sum() / num_valid

        # 2. Gradient / Edge loss
        pred_dx = F.conv2d(pred_depth, self.kx, padding=1)
        pred_dy = F.conv2d(pred_depth, self.ky, padding=1)
        gt_dx = F.conv2d(gt_depth, self.kx, padding=1)
        gt_dy = F.conv2d(gt_depth, self.ky, padding=1)

        grad_diff = (torch.abs(pred_dx - gt_dx) + torch.abs(pred_dy - gt_dy)) * mask
        grad_loss = grad_diff.sum() / num_valid

        total_loss = self.alpha_l1 * l1_loss + self.alpha_grad * grad_loss
        return total_loss, l1_loss, grad_loss


# ---------------------------------------------------------------------------
# Checkpoint Saving & Loading
# ---------------------------------------------------------------------------
def save_checkpoint(model, path, epoch=0, val_loss=0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "arch": "EdgeFuseNet",
        "in_channels": 5,
        "base_channels": 32
    }, path)


def load_model(checkpoint_path, device="cpu"):
    model = EdgeFuseNet(in_channels=5, base_channels=32)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

