"""
train.py - Supervised Training for EdgeFuseNet on NYU Depth V2.
Trains a compact residual RGB-guided depth network with edge-aware loss.
"""

import os
import sys
import time
import json
import torch
from torch.utils.data import DataLoader

from src.data import NYUDepthV2Dataset
from src.model import EdgeFuseNet, EdgeAwareDepthLoss, save_checkpoint


def train(
    mat_path="Datasets/Extracted_Dataset_Files/NYU Depth V2/nyu_depth_v2_labeled.mat",
    output_checkpoint="EDGEFUSE_DCR/models/edgefuse_dcr_compact.pth",
    epochs=5,
    batch_size=4,
    lr=1e-3,
    train_samples=50,
    val_samples=10,
    downsample_factor=4,
    device="cpu"
):
    print("=" * 60)
    print("EdgeFuse-DCR: Compact RGB-Guided Network Training")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"NYU Dataset: {mat_path}")
    print(f"Downsample factor: {downsample_factor}x")
    print("=" * 60)

    # 1. Dataset & DataLoader
    train_dataset = NYUDepthV2Dataset(
        mat_path=mat_path,
        split="train",
        split_ratio=0.85,
        downsample_factor=downsample_factor,
        crop_size=(240, 320),
        max_samples=train_samples
    )
    val_dataset = NYUDepthV2Dataset(
        mat_path=mat_path,
        split="val",
        split_ratio=0.85,
        downsample_factor=downsample_factor,
        crop_size=(240, 320),
        max_samples=val_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 2. Model & Optimizer
    model = EdgeFuseNet(in_channels=5, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = EdgeAwareDepthLoss(alpha_l1=1.0, alpha_grad=0.25).to(device)

    best_val_rmse = float("inf")
    history = []

    # 3. Training Loop
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_grad = 0.0
        n_batches = 0

        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            lr_depth = batch["lr_depth"].to(device)
            mask = batch["valid_mask"].to(device)
            gt_depth = batch["gt_depth"].to(device)

            optimizer.zero_grad()
            pred_d, _ = model(rgb, lr_depth, mask)

            loss, l1, grad = criterion(pred_d, gt_depth, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_l1 += l1.item()
            train_grad += grad.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)
        avg_l1 = train_l1 / max(n_batches, 1)
        avg_grad = train_grad / max(n_batches, 1)

        # Validation
        model.eval()
        val_rmse_acc = 0.0
        val_mae_acc = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                lr_depth = batch["lr_depth"].to(device)
                mask = batch["valid_mask"].to(device)
                gt_depth = batch["gt_depth"].to(device)

                pred_d, _ = model(rgb, lr_depth, mask)

                valid_pixels = (gt_depth > 0.05) & (mask > 0.5)
                diff = (pred_d - gt_depth) * valid_pixels

                val_mae_acc += diff.abs().sum().item() / valid_pixels.sum().clamp(min=1).item()
                val_rmse_acc += torch.sqrt((diff ** 2).sum() / valid_pixels.sum().clamp(min=1)).item()
                val_count += 1

        avg_val_rmse = val_rmse_acc / max(val_count, 1)
        avg_val_mae = val_mae_acc / max(val_count, 1)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Loss: {avg_train_loss:.4f} (L1: {avg_l1:.4f}, Grad: {avg_grad:.4f}) | "
            f"Val RMSE: {avg_val_rmse:.4f}m, MAE: {avg_val_mae:.4f}m | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_rmse": avg_val_rmse,
            "val_mae": avg_val_mae
        })

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            save_checkpoint(model, output_checkpoint, epoch=epoch, val_loss=avg_val_rmse)
            print(f"  -> Best model saved to: {output_checkpoint} (Val RMSE: {best_val_rmse:.4f}m)")

    train_dataset.close()
    val_dataset.close()

    total_time = time.time() - t_start
    print("=" * 60)
    print(f"Training completed in {total_time:.1f}s. Final Best Val RMSE: {best_val_rmse:.4f}m")
    print(f"Checkpoint location: {output_checkpoint}")
    print("=" * 60)

    # Save training log JSON
    log_p = os.path.splitext(output_checkpoint)[0] + "_history.json"
    with open(log_p, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(epochs=5, batch_size=4, train_samples=40, val_samples=10, device=device)

