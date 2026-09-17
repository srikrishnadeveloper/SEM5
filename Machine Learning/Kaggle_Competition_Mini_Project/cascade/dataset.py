"""Dataset and feature extraction utilities for Cascade 2.0.

Hardened with:
1. Multi-annotator zero-leakage group partitioning by physical image filename.
2. Astronomical augmentations: Flips, 90-deg rotations, brightness/contrast jitter.
3. Realistic prior perturbation: Simulates YOLO prototype dilation, erosion, and unsharp fallback.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cascade.geometry import compute_adaptive_crop_side, square_bounds_from_bbox


def build_3channel_input(
    gray: np.ndarray,
    coarse_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build 3-channel astronomical feature representation:
    Channel 0: Normalized raw grayscale image.
    Channel 1: CLAHE enhanced channel (local chromospheric contrast).
    Channel 2: Coarse mask prior (if available) or High-Pass unsharp filter.
    """
    ch_raw = gray.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(gray)

    if coarse_prior is not None and coarse_prior.sum() > 0:
        ch_prior = (coarse_prior > 0).astype(np.uint8) * 255
    else:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5)
        ch_prior = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return np.stack([ch_raw, ch_clahe, ch_prior], axis=-1)


class FilamentCropDataset(Dataset):
    """Dataset yielding high-resolution filament crops and instance masks."""

    def __init__(
        self,
        records: List[Dict],
        target_size: int = 384,
        augment: bool = True,
        simulate_prior: bool = True,
    ):
        self.records = records
        self.target_size = target_size
        self.augment = augment
        self.simulate_prior = simulate_prior

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_path = rec["img_path"]
        bbox = rec["bbox"]  # [x1, y1, x2, y2]
        pts = rec["pts"]    # (N, 2) float polygon

        # 1. Read grayscale full image
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Failed to read image at: {img_path}")
        H, W = gray.shape[:2]

        # 2. Rasterize full instance mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

        # 3. Simulate realistic YOLO prior variations for Channel 2
        if self.simulate_prior:
            r = np.random.rand()
            if r < 0.60:
                # Prototype blur/dilation (typical YOLO prototype behavior)
                k = np.random.randint(3, 9)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                coarse_prior = cv2.dilate(mask, kernel, iterations=1)
            elif r < 0.80:
                # Weak detection / tip erosion
                k = np.random.randint(3, 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                coarse_prior = cv2.erode(mask, kernel, iterations=1)
                if coarse_prior.sum() == 0:
                    coarse_prior = mask
            elif r < 0.90:
                # Box fallback
                coarse_prior = np.zeros_like(mask)
                coarse_prior[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            else:
                # Fallback to unsharp feature channel (no prior)
                coarse_prior = None
        else:
            coarse_prior = mask

        img_3ch = build_3channel_input(gray, coarse_prior)

        # 4. Compute adaptive non-truncating square crop bounds (up to 2048px)
        side = compute_adaptive_crop_side(bbox, crop_min=256, crop_max=2048, crop_padding=1.25)
        cx1, cy1, cx2, cy2 = square_bounds_from_bbox(bbox, H, W, side)

        crop_img = img_3ch[cy1:cy2, cx1:cx2]
        crop_mask = mask[cy1:cy2, cx1:cx2]

        # 5. Resize to target patch size
        resized_img = cv2.resize(crop_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # 6. Online geometric & photometric augmentations
        if self.augment:
            # Flips
            if np.random.rand() > 0.5:
                resized_img = np.fliplr(resized_img).copy()
                resized_mask = np.fliplr(resized_mask).copy()
            if np.random.rand() > 0.5:
                resized_img = np.flipud(resized_img).copy()
                resized_mask = np.flipud(resized_mask).copy()
            # Rotations
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                resized_img = np.rot90(resized_img, rot_k).copy()
                resized_mask = np.rot90(resized_mask, rot_k).copy()
            # Subtle brightness / contrast jitter
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-15, 15)
                ch0 = np.clip(resized_img[:, :, 0] * alpha + beta, 0, 255).astype(np.uint8)
                ch1 = np.clip(resized_img[:, :, 1] * alpha + beta, 0, 255).astype(np.uint8)
                resized_img[:, :, 0] = ch0
                resized_img[:, :, 1] = ch1

        # 7. Convert to tensors normalized to [0, 1]
        tensor_img = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
        tensor_mask = torch.from_numpy(resized_mask).unsqueeze(0).float()

        return tensor_img, tensor_mask


def load_dataset_records(
    data_dir: Path,
    val_ratio: float = 0.18,
    seed: int = 2026,
) -> Tuple[List[Dict], List[Dict]]:
    """Parse COCO annotations and split into train and val records grouped by physical disk."""
    train_dir = data_dir / "train"
    json_path = next(train_dir.glob("*.json"))

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}

    # Group physical files by unique stem to avoid multi-annotator leakage
    all_files = sorted(list({img["file_name"] for img in coco.get("images", [])}))
    np.random.seed(seed)
    shuffled = np.random.permutation(all_files)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_files = set(shuffled[:n_val])

    train_records = []
    val_records = []

    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        fn = id_to_file.get(img_id)
        if not fn:
            continue

        segs = ann.get("segmentation", [])
        if not isinstance(segs, list):
            continue

        img_path = train_dir / "train_images" / fn
        is_val = fn in val_files

        for poly in segs:
            if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                x1, y1 = np.min(pts, axis=0)
                x2, y2 = np.max(pts, axis=0)
                if (x2 - x1) > 4 and (y2 - y1) > 4:
                    rec = {
                        "img_path": img_path,
                        "file_name": fn,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "pts": pts,
                    }
                    if is_val:
                        val_records.append(rec)
                    else:
                        train_records.append(rec)

    return train_records, val_records
