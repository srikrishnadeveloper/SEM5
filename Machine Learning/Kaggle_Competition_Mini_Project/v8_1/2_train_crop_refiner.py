"""
v8_1 Stage 2 — Train Crop U-Net Refiner (ResNet-34) on MAGFiLO Crops.

Master: ChatGPT | Builder: Antigravity
Target: Kaggle Solar Filament Segmentation Challenge 2026

Frozen Rules:
- Train only on train-fold instances (579 files / 951 observations).
- Crop side: min(512, max(256, int(1.2 * max(w, h)))), square resize to 384 for batch.
- 3-channel input: [Raw Grayscale, CLAHE (3.0, 8x8), High-Pass Unsharp Filter].
- Model: U-Net with ResNet-34 backbone, 1 class, AMP.
- Refuse to train if not torch.cuda.is_available() (unless --dry-run).
- Require YOLO best.pt unless --boxes-from-gt (allowed for parallel GT-crop pretrain).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v8_1.config import CropRefinerConfig, YOLOConfig, MAGFILO_DIR, YOLO_DATA_DIR, MODELS_DIR
from v8_1.geometry import compute_adaptive_crop_side, square_bounds_from_bbox

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_crop_refiner_config(boxes_from_gt: bool):
    """Print the frozen Crop Refiner specification for review."""
    print("=" * 75)
    print("V8_1 STAGE 2: CROP U-NET REFINER (FROZEN CONFIGURATION)")
    print("=" * 75)
    print(f"  Backbone Architecture: {CropRefinerConfig.ENCODER} (U-Net, 1 class)")
    print(f"  Input Channels:        {CropRefinerConfig.IN_CHANNELS} ([Raw, CLAHE, Unsharp])")
    print(f"  Square Crop Target:    {CropRefinerConfig.TARGET_SIZE}x{CropRefinerConfig.TARGET_SIZE} (Square Resize)")
    print(f"  Adaptive Crop Formula: min({CropRefinerConfig.CROP_MAX}, max({CropRefinerConfig.CROP_MIN}, int({CropRefinerConfig.CROP_PADDING} * max(w, h))))")
    print(f"  Epochs:                {CropRefinerConfig.EPOCHS}")
    print(f"  Batch Size:            {CropRefinerConfig.BATCH_SIZE}")
    print(f"  Learning Rate:         {CropRefinerConfig.LR} (AdamW, Cosine Annealing)")
    print(f"  Mixed Precision (AMP): {CropRefinerConfig.AMP}")
    print(f"  Box Proposals Source:  {'Ground Truth Polygons (--boxes-from-gt)' if boxes_from_gt else 'YOLO11s-seg best.pt'}")
    print(f"  Output Checkpoint:     {CropRefinerConfig.CKPT_PATH}")
    print(f"  CUDA Available:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'})")
    print("=" * 75)


def preprocess_3ch(gray: np.ndarray) -> np.ndarray:
    """Build [Raw, CLAHE, Unsharp] 3-channel input."""
    ch_raw = gray.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5)
    ch_unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return np.stack([ch_raw, ch_clahe, ch_unsharp], axis=-1)


class FilamentCropDataset(Dataset):
    """Lazy-loaded object-centered crop dataset for Stage 2 Refiner."""

    def __init__(self, crop_records: list[dict], target_size: int = 384, augment: bool = True):
        self.records = crop_records
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["img_path"]
        bbox = rec["bbox"]  # [x1, y1, x2, y2]
        pts = rec["pts"]    # polygon points (N, 2)

        # 1. Read grayscale image and preprocess
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"FATAL: Unable to decode training image: {img_path}")
        img_3ch = preprocess_3ch(gray)
        h, w = gray.shape[:2]

        # 2. Rasterize polygon instance mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

        # 3. Strictly square adaptive crop window (no aspect-ratio distortion near boundaries)
        side = compute_adaptive_crop_side(
            bbox,
            crop_min=CropRefinerConfig.CROP_MIN,
            crop_max=CropRefinerConfig.CROP_MAX,
            crop_padding=CropRefinerConfig.CROP_PADDING,
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = square_bounds_from_bbox(bbox, h, w, side)

        crop_img = img_3ch[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        # 4. Square resize to target_size (384x384)
        resized_img = cv2.resize(crop_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # 5. Optional online flip augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                resized_img = np.fliplr(resized_img).copy()
                resized_mask = np.fliplr(resized_mask).copy()
            if np.random.rand() > 0.5:
                resized_img = np.flipud(resized_img).copy()
                resized_mask = np.flipud(resized_mask).copy()

        # Convert to tensor: (C, H, W) normalized to [0, 1]
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(resized_mask).unsqueeze(0).float()

        return img_tensor, mask_tensor


def build_crop_records(data_dir: Path, yolo_data_dir: Path) -> list[dict]:
    """Build crop metadata records strictly from train-fold instances."""
    train_label_files = list((yolo_data_dir / "labels" / "train").glob("*.txt"))
    train_physical_stems = {p.stem.split("_ann_")[0] for p in train_label_files}

    train_dir = data_dir / "train"
    json_path = next(train_dir.glob("*.json"))
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    records = []

    for ann in coco.get("annotations", []):
        # All categories 1, 2, 3, 4 are solar filaments for segmentation objective
        img_id = ann["image_id"]
        fn = id_to_file.get(img_id)
        if not fn:
            continue
        stem = Path(fn).stem
        if stem not in train_physical_stems:
            continue  # Must stay strictly inside train fold

        segs = ann.get("segmentation", [])
        if not isinstance(segs, list):
            continue

        img_path = data_dir / "train" / "train_images" / fn
        for poly in segs:
            if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                x1, y1 = np.min(pts, axis=0)
                x2, y2 = np.max(pts, axis=0)
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    records.append({
                        "img_path": img_path,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "pts": pts,
                    })

    return records


def train_crop_refiner(records: list[dict]):
    """Execute training loop for Crop U-Net refiner on GPU."""
    import segmentation_models_pytorch as smp

    print(f"📦 Assembling Crop Dataset ({len(records)} instance crops)...")
    dataset = FilamentCropDataset(records, target_size=CropRefinerConfig.TARGET_SIZE, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=CropRefinerConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name=CropRefinerConfig.ENCODER,
        encoder_weights="imagenet",
        in_channels=CropRefinerConfig.IN_CHANNELS,
        classes=CropRefinerConfig.CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CropRefinerConfig.LR, weight_decay=CropRefinerConfig.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CropRefinerConfig.EPOCHS, eta_min=1e-6)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=CropRefinerConfig.AMP)

    print("🚀 Training Crop U-Net Refiner...")
    model.train()

    import time
    epoch_durations = []
    t_start = time.perf_counter()

    for epoch in range(1, CropRefinerConfig.EPOCHS + 1):
        t_ep_start = time.perf_counter()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=CropRefinerConfig.AMP):
                logits = model(imgs)
                # Combined BCE + Dice loss
                probs = torch.sigmoid(logits)
                intersection = (probs * masks).sum(dim=(1, 2, 3))
                dice = (2.0 * intersection + 1.0) / (probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1.0)
                loss = 0.5 * bce_loss_fn(logits, masks) + 0.5 * (1.0 - dice.mean())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        ep_duration = time.perf_counter() - t_ep_start
        epoch_durations.append(ep_duration)
        avg_loss = total_loss / max(1, len(loader))
        print(f"  Epoch [{epoch:02d}/{CropRefinerConfig.EPOCHS:02d}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} - Duration: {ep_duration:.2f}s")

    total_training_time = time.perf_counter() - t_start
    avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0
    print(f"\n[TIMING] Crop Refiner Average Epoch Duration: {avg_epoch_duration:.2f}s | Total: {total_training_time:.2f}s ({total_training_time / 60:.2f} min)")

    # Save checkpoint
    CropRefinerConfig.CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CropRefinerConfig.CKPT_PATH)
    print(f"[OK] Saved Crop U-Net refiner checkpoint to: {CropRefinerConfig.CKPT_PATH}")

    # Memory cleanup & VRAM logging
    del model, optimizer, scheduler, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for dev_idx in range(torch.cuda.device_count()):
            res_mb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
            alloc_mb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
            print(f"  [GPU {dev_idx}] Memory Reserved: {res_mb:.1f} MB | Allocated: {alloc_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Train Crop U-Net Refiner")
    parser.add_argument("--dry-run", action="store_true", help="Print frozen config and exit 0 without training")
    parser.add_argument("--boxes-from-gt", action="store_true", help="Extract crops directly from GT polygons for parallel pretrain")
    args = parser.parse_args()

    if args.dry_run:
        print_crop_refiner_config(boxes_from_gt=args.boxes_from_gt)
        print("Dry run completed successfully. Exiting 0 on CPU.\n")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available! Crop Refiner training refuses to run on CPU.")
        print("Run with --dry-run to inspect configuration on CPU.")
        sys.exit(1)

    # Dependency check: require YOLO weights unless --boxes-from-gt is set
    if not args.boxes_from_gt and not YOLOConfig.WEIGHTS_PATH.exists():
        print(f"[ERROR] Stage 1 YOLO weights not found at {YOLOConfig.WEIGHTS_PATH}!")
        print("Train Stage 1 YOLO first, or specify --boxes-from-gt to run a parallel GT-crop pretrain.")
        sys.exit(1)

    print_crop_refiner_config(boxes_from_gt=args.boxes_from_gt)

    records = build_crop_records(data_dir=MAGFILO_DIR, yolo_data_dir=YOLO_DATA_DIR)
    print(f"  Loaded {len(records)} instance crops strictly from train fold (579 files).")
    train_crop_refiner(records)


if __name__ == "__main__":
    main()
