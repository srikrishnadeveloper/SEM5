"""
Solar Filament Segmentation Challenge 2026 — Version 2 Pipeline (Bullet-Proof Production Edition)

Key Highlights:
- Hardcoded Optimal Architecture: UnetPlusPlus + EfficientNet-B4 + SCSE Attention (3-Channel RGB)
- Resolution: 512x512 Fast Training -> Bilinear Probability Upsampling to 2048x2048 Native Resolution
- Loss: Balanced Hybrid Loss (0.30*BCE + 0.40*Dice + 0.30*Focal) — Zero Gradient Conflicts
- Inference: 4-Flip Test-Time Augmentation (Original, Horizontal, Vertical, HV)
- Post-Processing: Morphological Cleanup + Connected Components + Area Filtering + COCO RLE Encoding
- Extensive Lag-Detection Logging, VRAM Tracking & Network-Safe Weight Loading
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import json
import time
import glob
import gc
import math
import random
import socket
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Set network socket timeout to avoid hanging on any external network calls
socket.setdefaulttimeout(15.0)

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pycocotools.mask as mask_utils
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────────────────────
# 1. Logging & Diagnostic Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_timestamp() -> str:
    """Return formatted timestamp string [HH:MM:SS]."""
    return time.strftime("[%H:%M:%S]")

def log_info(msg: str):
    """Print timestamped informational message."""
    print(f"{get_timestamp()} [INFO] {msg}", flush=True)

def log_warn(msg: str):
    """Print timestamped warning message."""
    print(f"{get_timestamp()} [WARN] {msg}", flush=True)

def log_error(msg: str):
    """Print timestamped error message."""
    print(f"{get_timestamp()} [ERROR] {msg}", flush=True)

def get_vram_info(device: str) -> str:
    """Return formatted GPU VRAM usage if CUDA is active."""
    if "cuda" in device and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"VRAM: {allocated:.2f}GB alloc / {reserved:.2f}GB res / {total:.2f}GB tot"
    return "Device: CPU"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hardcoded Best Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Experiment metadata
    exp_name: str = "v2_unetplusplus_effb4_production"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths (auto-resolved across Kaggle, Colab, and local environments)
    base_dir: str = ""
    train_images: str = ""
    train_json: str = ""
    test_images: str = ""
    output_dir: str = "submissions"
    models_dir: str = "models"
    
    # HARDCODED BEST ARCHITECTURE
    arch: str = "UnetPlusPlus"
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"
    decoder_attention: str = "scse"
    in_channels: int = 3
    classes: int = 1
    
    # HARDCODED BEST HYPERPARAMETERS
    img_size: int = 512                  # Optimal input size for receptive field vs speed
    full_res: int = 2048                 # Native competition evaluation resolution
    batch_size: int = 8                  # Fits P100 (16GB) & T4 (16GB) comfortably at 512px
    accum_steps: int = 1                 # Gradient accumulation steps
    epochs: int = 25                     # Epoch count for full convergence
    lr: float = 2e-4                     # Initial AdamW learning rate
    min_lr: float = 1e-6                 # Cosine annealing floor
    weight_decay: float = 1e-4
    
    # Cross validation
    n_folds: int = 5
    val_fold: int = 0                    # Default validation fold
    train_all_folds: bool = True         # True: Train all 5 folds for winning ensemble
    
    # Hardware & Precision
    use_amp: bool = True                 # Mixed precision FP16
    num_workers: int = 2 if os.name != "nt" and torch.cuda.is_available() else 0
    lag_threshold_sec: float = 3.0       # Warning threshold for data/compute lag
    
    # Loss weights (proven balanced configuration)
    w_bce: float = 0.30
    w_dice: float = 0.40
    w_focal: float = 0.30
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Inference & Post-processing
    use_tta: bool = True                 # 4-flip TTA
    threshold: float = 0.45              # Default probability binarization threshold
    min_area: int = 200                  # Minimum filament pixel area at 2048x2048
    morph_k: int = 3                     # Morphological closing kernel diameter
    
    def resolve_paths(self):
        """Auto-detect dataset directories safely without network hangs."""
        candidate_bases = [
            Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/kaggle/input/filament-segmentation-2026"),
            Path("/content/MAGFiLO_1.0_Kaggle_2026"),
            Path("/content/filament_data/MAGFiLO_1.0_Kaggle_2026"),
            Path("/content/drive/MyDrive/MAGFiLO_1.0_Kaggle_2026"),
            Path("/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/content/drive/MyDrive/kaggle_SF/MAGFiLO_1.0_Kaggle_2026"),
            Path("data/MAGFiLO_1.0_Kaggle_2026"),
            Path("../data/MAGFiLO_1.0_Kaggle_2026"),
            Path("./data"),
        ]
        detected_base = None
        for cand in candidate_bases:
            if cand.exists():
                detected_base = cand
                break
                
        if self.base_dir and Path(self.base_dir).exists():
            detected_base = Path(self.base_dir)
            
        if detected_base is not None:
            self.base_dir = str(detected_base)
            tr_cand = list(detected_base.glob("**/train_images"))
            self.train_images = str(tr_cand[0]) if tr_cand else str(detected_base / "train" / "train_images")
            
            ts_cand = list(detected_base.glob("**/test_images"))
            self.test_images = str(ts_cand[0]) if ts_cand else str(detected_base / "test" / "test_images")
            
            json_cand = list(detected_base.glob("**/*.json"))
            self.train_json = str(json_cand[0]) if json_cand else str(detected_base / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
            
        # Set working directory for outputs
        if os.path.exists("/kaggle/working"):
            out_root = Path("/kaggle/working")
        elif os.path.exists("/content"):
            out_root = Path("/content")
        else:
            out_root = Path("./")
            
        self.output_dir = str(out_root / "submissions")
        self.models_dir = str(out_root / "models")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)


def seed_everything(seed: int = 42):
    """Seed Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Robust Data Parsing & Splitting
# ─────────────────────────────────────────────────────────────────────────────

def parse_coco_annotations(ann_path: str) -> Tuple[Dict[str, List[List[float]]], Dict[str, Tuple[int, int]]]:
    """Parse COCO annotations JSON with validation for corrupted or missing items."""
    if not os.path.exists(ann_path):
        log_warn(f"Annotation file not found: {ann_path}")
        return {}, {}
        
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log_error(f"Failed to read JSON at {ann_path}: {e}")
        return {}, {}
        
    id_to_filename = {}
    filename_to_shape = {}
    for img in data.get("images", []):
        img_id = img.get("id")
        file_name = img.get("file_name")
        if img_id is not None and file_name:
            id_to_filename[img_id] = file_name
            filename_to_shape[file_name] = (img.get("height", 2048), img.get("width", 2048))
            
    filename_to_polys = defaultdict(list)
    for ann in data.get("annotations", []):
        fn = id_to_filename.get(ann.get("image_id"))
        if not fn:
            continue
        segs = ann.get("segmentation", [])
        if isinstance(segs, list):
            for poly in segs:
                if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
                    filename_to_polys[fn].append(poly)
                elif isinstance(poly, np.ndarray) and poly.size >= 6 and poly.size % 2 == 0:
                    filename_to_polys[fn].append(poly.tolist())
                    
    log_info(f"Loaded annotations for {len(filename_to_polys)} images ({sum(len(v) for v in filename_to_polys.values())} total polygons)")
    return filename_to_polys, filename_to_shape


def rasterize_polygons(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Render polygon coordinates onto a binary uint8 mask of shape (height, width)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask
    for poly in polygons:
        try:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        except Exception:
            continue
    return mask


def build_splits(train_images_dir: str, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """GroupKFold splitting by Year to prevent temporal and multi-annotator leakage."""
    if not os.path.exists(train_images_dir):
        raise FileNotFoundError(f"Train directory not found: {train_images_dir}")
        
    files = sorted([f for f in os.listdir(train_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not files:
        raise ValueError(f"No valid image files found in {train_images_dir}")
        
    df = pd.DataFrame({"filename": files})
    
    def get_group(fn):
        digits = "".join(filter(str.isdigit, fn))
        return digits[:4] if len(digits) >= 4 else "unknown"
        
    df["group"] = df["filename"].apply(get_group)
    df["fold"] = -1

    unique_groups = df["group"].unique()
    if len(unique_groups) >= n_folds:
        gkf = GroupKFold(n_splits=n_folds)
        for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df["group"])):
            df.loc[val_idx, "fold"] = fold
        log_info(f"Built {n_folds}-fold GroupKFold dataset split over {len(df)} images across {len(unique_groups)} year groups.")
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, "fold"] = fold
        log_warn(f"Unique groups ({len(unique_groups)}) < n_folds ({n_folds}); fell back to standard KFold.")
        
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Augmentation & Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int) -> A.Compose:
    """Robust augmentations optimized for solar H-alpha imagery."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.90, 1.10), translate_percent=(-0.0625, 0.0625), rotate=(-30, 30), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int) -> A.Compose:
    """Standard normalization & resize for validation and inference."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class SolarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        filename_to_polys: Optional[Dict[str, List[List[float]]]] = None,
        transform: Optional[A.Compose] = None,
        is_test: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.filename_to_polys = filename_to_polys or {}
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        fn = self.df.iloc[idx]["filename"]
        path = os.path.join(self.img_dir, fn)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
            
        # Ensure 3-channel RGB image format
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        h, w = img.shape[:2]

        if self.is_test:
            if self.transform:
                aug = self.transform(image=img)
                img = aug["image"]
            return img, fn, (h, w)

        # Rasterize ground truth mask at native resolution
        polys = self.filename_to_polys.get(fn, [])
        mask = rasterize_polygons(polys, h, w)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        # Ensure mask is float32 with shape (1, H, W)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            mask = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()

        return img, mask, fn


# ─────────────────────────────────────────────────────────────────────────────
# 5. Losses & Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        return 1.0 - (num / den).mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        return (focal_weight * bce).mean()


class HybridLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=1.0)
        self.focal = BinaryFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_bce = self.bce(logits, targets)
        l_dice = self.dice(logits, targets)
        l_focal = self.focal(logits, targets)
        total = self.cfg.w_bce * l_bce + self.cfg.w_dice * l_dice + self.cfg.w_focal * l_focal
        metrics = {
            "loss": total.item(),
            "bce": l_bce.item(),
            "dice_l": l_dice.item(),
            "focal": l_focal.item()
        }
        return total, metrics


def build_model(cfg: Config) -> nn.Module:
    """Build segmentation model safely with fallback if offline or weights unavailable."""
    model_cls = getattr(smp, cfg.arch, smp.UnetPlusPlus)
    kwargs = {
        "encoder_name": cfg.encoder_name,
        "encoder_weights": cfg.encoder_weights,
        "in_channels": cfg.in_channels,
        "classes": cfg.classes,
        "activation": None
    }
    
    # Try with attention and pre-trained weights
    try:
        if cfg.decoder_attention:
            return model_cls(decoder_attention_type=cfg.decoder_attention, **kwargs)
        return model_cls(**kwargs)
    except Exception as e:
        log_warn(f"Standard model initialization failed ({e}). Falling back without attention or weights...")
        
    # Fallback attempt
    try:
        kwargs["encoder_weights"] = None
        return model_cls(**kwargs)
    except Exception as e2:
        log_error(f"Fallback model initialization failed: {e2}")
        raise e2


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training, Validation & Evaluation with Lag Logging
# ─────────────────────────────────────────────────────────────────────────────

def calculate_dice_score(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    total = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: HybridLoss,
    scaler: torch.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: Config
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    last_batch_time = time.time()
    for step, (images, masks, _) in enumerate(loader):
        t_data = time.time() - last_batch_time
        t_compute_start = time.time()

        images = images.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", enabled=cfg.use_amp):
            logits = model(images)
            loss, _ = loss_fn(logits, masks)
            loss = loss / cfg.accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * cfg.accum_steps

        if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        t_compute = time.time() - t_compute_start
        total_step_time = t_data + t_compute

        # Detect lag in step
        if total_step_time > cfg.lag_threshold_sec and step > 0:
            log_warn(f"Step {step}/{len(loader)} lag: {total_step_time:.2f}s (Data load: {t_data:.2f}s, Compute: {t_compute:.2f}s)")

        last_batch_time = time.time()

    if scheduler is not None:
        scheduler.step()

    return {"train_loss": total_loss / max(1, len(loader))}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: HybridLoss,
    cfg: Config
) -> Tuple[float, float, float]:
    model.eval()
    val_loss = 0.0
    dice_45_list = []
    dice_50_list = []

    for images, masks, _ in loader:
        images = images.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in cfg.device else "cpu", enabled=cfg.use_amp):
            logits = model(images)
            loss, _ = loss_fn(logits, masks)
            probs = torch.sigmoid(logits)

        val_loss += loss.item()
        dice_45_list.append(calculate_dice_score(probs, masks, threshold=0.45))
        dice_50_list.append(calculate_dice_score(probs, masks, threshold=0.50))

    avg_loss = val_loss / max(1, len(loader))
    avg_d45 = float(np.mean(dice_45_list)) if dice_45_list else 0.0
    avg_d50 = float(np.mean(dice_50_list)) if dice_50_list else 0.0
    return avg_loss, avg_d45, avg_d50


# ─────────────────────────────────────────────────────────────────────────────
# 7. Inference with 4-Flip TTA & Post-Processing
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """4-Flip Test-Time Augmentation (Original, Horizontal, Vertical, HV)."""
    p1 = torch.sigmoid(model(images))
    p2 = torch.flip(torch.sigmoid(model(torch.flip(images, dims=[3]))), dims=[3])
    p3 = torch.flip(torch.sigmoid(model(torch.flip(images, dims=[2]))), dims=[2])
    p4 = torch.flip(torch.sigmoid(model(torch.flip(images, dims=[2, 3]))), dims=[2, 3])
    return (p1 + p2 + p3 + p4) / 4.0


def mask_to_coco_rle(binary_mask: np.ndarray) -> str:
    """Encode 2D uint8 binary mask to compressed COCO RLE string."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    encoded = mask_utils.encode(fortran_mask)
    if isinstance(encoded["counts"], bytes):
        return encoded["counts"].decode("utf-8")
    return encoded["counts"]


def search_best_threshold(
    oof_probs: List[np.ndarray],
    oof_gts: List[np.ndarray],
    eval_res: int = 512
) -> Tuple[float, int, float]:
    """Fast grid-search on validation probability maps to maximize Dice score."""
    log_info(f"Optimizing threshold & min_area on {len(oof_probs)} OOF maps (eval @ {eval_res}px)...")
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    min_areas = [100, 200, 300, 500]

    best_dice = -1.0
    best_thresh = 0.45
    best_area = 200

    for t in thresholds:
        for a in min_areas:
            dices = []
            scaled_area = max(1, int(a * ((eval_res / 2048.0) ** 2)))
            for prob, gt in zip(oof_probs, oof_gts):
                bin_m = (prob > t).astype(np.uint8)
                if scaled_area > 1:
                    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_m, connectivity=8)
                    for lbl in range(1, n_labels):
                        if stats[lbl, cv2.CC_STAT_AREA] < scaled_area:
                            bin_m[labels == lbl] = 0

                intersection = (bin_m * gt).sum()
                total = bin_m.sum() + gt.sum()
                d = (2.0 * intersection + 1e-6) / (total + 1e-6)
                dices.append(d)

            mean_d = float(np.mean(dices))
            if mean_d > best_dice:
                best_dice = mean_d
                best_thresh = t
                best_area = a

    log_info(f"Optimal Parameters: Threshold = {best_thresh:.2f}, Min Area = {best_area} px -> OOF Dice: {best_dice:.4f}")
    return best_thresh, best_area, best_dice


# ─────────────────────────────────────────────────────────────────────────────
# 8. End-to-End Execution Engine
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(fold: int, df_splits: pd.DataFrame, cfg: Config) -> str:
    """Train single fold, track metrics, save best checkpoint and return model path."""
    print("\n" + "=" * 75)
    log_info(f"TRAINING FOLD {fold}/{cfg.n_folds - 1} | Architecture: {cfg.arch} ({cfg.encoder_name}) | {get_vram_info(cfg.device)}")
    print("=" * 75)

    train_df = df_splits[df_splits["fold"] != fold].reset_index(drop=True)
    val_df = df_splits[df_splits["fold"] == fold].reset_index(drop=True)
    log_info(f"Data allocation: Train={len(train_df)} images | Val={len(val_df)} images")

    fn_to_polys, _ = parse_coco_annotations(cfg.train_json)

    train_ds = SolarDataset(train_df, cfg.train_images, fn_to_polys, transform=get_train_transforms(cfg.img_size))
    val_ds = SolarDataset(val_df, cfg.train_images, fn_to_polys, transform=get_val_transforms(cfg.img_size))

    pin_mem = torch.cuda.is_available()
    use_persist = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=use_persist, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=use_persist
    )

    model = build_model(cfg).to(cfg.device)
    loss_fn = HybridLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    scaler = torch.amp.GradScaler(device="cuda" if "cuda" in cfg.device else "cpu", enabled=cfg.use_amp)

    best_val_dice = 0.0
    best_checkpoint_path = os.path.join(cfg.models_dir, f"best_model_fold_{fold}.pth")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, scheduler, cfg)
        val_loss, val_d45, val_d50 = validate_epoch(model, val_loader, loss_fn, cfg)
        elapsed = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        primary_dice = val_d45

        is_best = primary_dice > best_val_dice
        if is_best:
            best_val_dice = primary_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "config": cfg.__dict__
            }, best_checkpoint_path)

        star = " * BEST" if is_best else ""
        log_info(f"Epoch {epoch:02d}/{cfg.epochs:02d} [{elapsed:.1f}s] | "
                 f"LR: {curr_lr:.2e} | "
                 f"Train Loss: {train_stats['train_loss']:.4f} | "
                 f"Val Loss: {val_loss:.4f} | "
                 f"Val Dice@0.45: {val_d45:.4f} | "
                 f"Val Dice@0.50: {val_d50:.4f}{star} | {get_vram_info(cfg.device)}")

        # Prevent VRAM buildup
        if epoch % 5 == 0:
            gc.collect()
            if "cuda" in cfg.device:
                torch.cuda.empty_cache()

    log_info(f"Fold {fold} finished! Best Validation Dice: {best_val_dice:.4f} -> Checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


def run_test_submission(
    model_paths: List[str],
    cfg: Config,
    threshold: float = 0.45,
    min_area: int = 200
) -> str:
    """Generate final submission.csv from an ensemble of trained models."""
    print("\n" + "=" * 75)
    log_info(f"GENERATING TEST SUBMISSION | Models: {len(model_paths)} | Threshold: {threshold:.2f} | Min Area: {min_area}px")
    print("=" * 75)

    test_files = sorted([f for f in os.listdir(cfg.test_images) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    test_df = pd.DataFrame({"filename": test_files})
    log_info(f"Found {len(test_df)} test images in {cfg.test_images}")

    test_ds = SolarDataset(test_df, cfg.test_images, transform=get_val_transforms(cfg.img_size), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    models = []
    for p in model_paths:
        m = build_model(cfg)
        ckpt = torch.load(p, map_location=cfg.device)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        m.to(cfg.device)
        m.eval()
        models.append(m)

    submissions = []
    seen_ids = set()

    t_infer_start = time.time()
    for images, filenames, shapes in tqdm(test_loader, desc="[Inference] generating masks"):
        images = images.to(cfg.device, non_blocking=True)
        img_fn = filenames[0]
        image_stem = Path(img_fn).stem
        orig_h = shapes[0][0].item() if isinstance(shapes[0][0], torch.Tensor) else shapes[0][0]
        orig_w = shapes[1][0].item() if isinstance(shapes[1][0], torch.Tensor) else shapes[1][0]

        # Ensemble predictions across models with 4-flip TTA
        avg_prob = torch.zeros((1, 1, cfg.img_size, cfg.img_size), device=cfg.device)
        for m in models:
            if cfg.use_tta:
                avg_prob += predict_tta(m, images)
            else:
                with torch.no_grad():
                    avg_prob += torch.sigmoid(m(images))
        avg_prob /= float(len(models))

        # Upsample probability map to native 2048x2048 with bilinear interpolation
        prob_np = avg_prob.squeeze().cpu().numpy()
        if (prob_np.shape[0], prob_np.shape[1]) != (orig_h, orig_w):
            prob_full = cv2.resize(prob_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            prob_full = prob_np

        # Threshold to create clean binary mask
        binary_mask = (prob_full > threshold).astype(np.uint8)

        # Morphological closing to seal thin disjoint bridges
        if cfg.morph_k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_k, cfg.morph_k))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Instance extraction via connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        filament_count = 0

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            single_instance = (labels == lbl).astype(np.uint8)
            rle_str = mask_to_coco_rle(single_instance)

            filament_count += 1
            fid = f"{image_stem}_{filament_count}"
            if fid not in seen_ids:
                seen_ids.add(fid)
                submissions.append({"filament_id": fid, "segmentation_rle": rle_str})

        # Standard token for images with zero detected filaments
        if filament_count == 0:
            fid = f"{image_stem}_1"
            if fid not in seen_ids:
                seen_ids.add(fid)
                submissions.append({"filament_id": fid, "segmentation_rle": "PPP2"})

    total_infer_time = time.time() - t_infer_start
    log_info(f"Inference completed in {total_infer_time:.1f}s ({total_infer_time / max(1, len(test_df)):.2f}s per image)")

    sub_df = pd.DataFrame(submissions)[["filament_id", "segmentation_rle"]]
    sub_df = sub_df.drop_duplicates(subset="filament_id", keep="first").reset_index(drop=True)

    out_csv = os.path.join(cfg.output_dir, "submission.csv")
    sub_df.to_csv(out_csv, index=False)

    # In Kaggle notebooks, also ensure submission.csv exists at the root /kaggle/working/
    if os.path.exists("/kaggle/working") and os.path.abspath(out_csv) != os.path.abspath("/kaggle/working/submission.csv"):
        try:
            import shutil
            shutil.copy2(out_csv, "/kaggle/working/submission.csv")
            log_info("Copied submission to /kaggle/working/submission.csv for direct Kaggle submission.")
        except Exception:
            pass

    print("\n" + "=" * 75)
    log_info("SUBMISSION VERIFICATION CHECK")
    print("=" * 75)
    log_info(f"Output file          : {out_csv}")
    log_info(f"Total rows           : {len(sub_df)}")
    log_info(f"Unique filament_ids  : {sub_df['filament_id'].nunique()}")
    log_info(f"Empty (PPP2) rows    : {(sub_df['segmentation_rle'] == 'PPP2').sum()}")
    log_info(f"Non-empty filaments  : {(sub_df['segmentation_rle'] != 'PPP2').sum()}")
    print("\nFirst 5 rows:")
    print(sub_df.head(5).to_string(index=False))
    return out_csv


def run(cfg: Optional[Config] = None):
    """Main execution function with hardcoded optimal settings."""
    if cfg is None:
        cfg = Config()
    cfg.resolve_paths()
    seed_everything(cfg.seed)

    log_info(f"Starting V2 Pipeline: {cfg.exp_name}")
    log_info(f"Device: {cfg.device} | Resolution: {cfg.img_size}px -> {cfg.full_res}px | {get_vram_info(cfg.device)}")
    log_info(f"Architecture: {cfg.arch} ({cfg.encoder_name}) | Batch: {cfg.batch_size} | Epochs: {cfg.epochs}")

    # Build dataset splits
    df_splits = build_splits(cfg.train_images, n_folds=cfg.n_folds, seed=cfg.seed)

    # Train specified fold or all folds
    trained_models = []
    folds_to_train = list(range(cfg.n_folds)) if cfg.train_all_folds else [cfg.val_fold]

    for f in folds_to_train:
        model_path = train_fold(f, df_splits, cfg)
        trained_models.append(model_path)

    # Comprehensive OOF threshold optimization across all trained folds
    fn_to_polys, _ = parse_coco_annotations(cfg.train_json)
    oof_probs = []
    oof_gts = []

    log_info(f"Extracting OOF predictions across {len(folds_to_train)} trained fold(s)...")
    for idx, f in enumerate(folds_to_train):
        val_fold_df = df_splits[df_splits["fold"] == f].reset_index(drop=True)
        val_ds = SolarDataset(val_fold_df, cfg.train_images, fn_to_polys, transform=get_val_transforms(cfg.img_size))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

        eval_model = build_model(cfg).to(cfg.device)
        ckpt = torch.load(trained_models[idx], map_location=cfg.device)
        eval_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        eval_model.eval()

        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"[OOF] Fold {f} predictions", leave=False):
                images = images.to(cfg.device)
                prob = predict_tta(eval_model, images) if cfg.use_tta else torch.sigmoid(eval_model(images))
                oof_probs.append(prob.squeeze().cpu().numpy())
                oof_gts.append(masks.squeeze().cpu().numpy())

        del eval_model
        gc.collect()
        if "cuda" in cfg.device:
            torch.cuda.empty_cache()

    best_thresh, best_area, best_dice = search_best_threshold(oof_probs, oof_gts, eval_res=cfg.img_size)

    # Generate submission using optimal parameters and all trained models
    run_test_submission(trained_models, cfg, threshold=best_thresh, min_area=best_area)


if __name__ == "__main__":
    run()
