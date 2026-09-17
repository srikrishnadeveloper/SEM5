"""Training pipeline for Stage 2 High-Resolution Patch Refiner.

Trains on extracted physical filament crops using AMP, CosineAnnealingLR,
and Composite Loss (BCE + SoftDice + Focal).
"""

from pathlib import Path
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cascade.model_refiner import build_crop_refiner, CompositeRefinerLoss
from cascade.dataset import FilamentCropDataset, load_dataset_records


def train_refiner(
    data_dir: Path,
    output_ckpt: Path,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    target_size: int = 384,
    arch: str = "unetplusplus",
    encoder: str = "tu-efficientnet_b2",
    device: Optional[torch.device] = None,
) -> Path:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🚀 CASCADE 2.0: STAGE 2 CROP REFINER TRAINING")
    print("=" * 70)
    print(f"  Device:         {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Architecture:   {arch} ({encoder})")
    print(f"  Target Size:    {target_size}x{target_size}")
    print(f"  Epochs:         {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Data Directory: {data_dir}")
    print("=" * 70)

    # 1. Load train and val records
    t0 = time.time()
    train_records, val_records = load_dataset_records(data_dir)
    print(f"📦 Extracted {len(train_records)} train crops and {len(val_records)} val crops in {time.time() - t0:.2f}s")

    train_ds = FilamentCropDataset(train_records, target_size=target_size, augment=True)
    val_ds = FilamentCropDataset(val_records, target_size=target_size, augment=False)

    num_workers = 2 if torch.cuda.is_available() and sys.platform != "win32" else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 2. Build model, loss, optimizer
    model = build_crop_refiner(arch=arch, encoder_name=encoder, in_channels=3).to(device)
    loss_fn = CompositeRefinerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_iou = -1.0
    best_loss = float("inf")
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        val_inter = 0.0
        val_union = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                    loss = loss_fn(logits, masks)

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.50).float()
                val_inter += (preds * masks).sum().item()
                val_union += (preds + masks).clamp(0, 1).sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_iou = (val_inter + 1e-6) / (val_union + 1e-6)
        ep_time = time.time() - ep_t0

        is_best = avg_val_iou > best_val_iou
        if is_best:
            best_val_iou = avg_val_iou
            best_loss = avg_val_loss
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "val_iou": best_val_iou,
                "arch": arch,
                "encoder": encoder,
            }, output_ckpt)

        mark = "★ BEST" if is_best else ""
        print(f"  Epoch [{ep:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | {ep_time:.1f}s {mark}")

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"✅ Training completed in {total_time / 60:.2f} minutes.")
    print(f"🏆 Best Checkpoint saved to: {output_ckpt} (Val IoU: {best_val_iou:.4f})")
    print("=" * 70)
    return output_ckpt


if __name__ == "__main__":
    from typing import Optional
    parser = argparse.ArgumentParser(description="Train Stage 2 Crop Refiner")
    parser.add_argument("--data-dir", type=str, default="data/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--out", type=str, default="models/best_crop_refiner.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch", type=str, default="unetplusplus")
    parser.add_argument("--encoder", type=str, default="tu-efficientnet_b2")
    args = parser.parse_args()

    train_refiner(
        data_dir=Path(args.data_dir),
        output_ckpt=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        arch=args.arch,
        encoder=args.encoder,
    )
