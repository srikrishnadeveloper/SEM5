"""notebooks/build_cascade_p1_nb.py
Builder for Bulletproof Cascade 2.0 (Crop & Zoom Paradigm) Kaggle Dual-GPU Notebook.
Target: Break 0.360 LB barrier and reach 0.40–0.50+ PQ within Kaggle 9h runtime.

Hardened with lessons learned across all previous versions:
1. Anchored Stage 1 precision: conf=0.25 on Native 2048 YOLOv8l-seg (0.360 Champion).
2. Non-truncating geometry: adaptive crop bounds up to 2048px (0% clipping).
3. 3-channel prior conditioning: instance guidance resolves dense clusters.
4. Morphological polish: 3x3 circular closing eliminates granular chromospheric noise.
5. Solar limb mask: removes edge flare outside r=0.93*R.
6. Greedy zero-overlap sanitizer: guarantees strictly 0 shared pixels per disk.
7. Host V6 compliant RLE: Fortran 3D shape (H, W, 1), 0 rows on empty disks.
8. Anti-leak memory management: gc.collect() and torch.cuda.empty_cache() checkpoints.
"""

from pathlib import Path
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def get_base_notebook():
    return {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "gpuClass": "standard",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build_cascade_p1_notebook():
    nb = get_base_notebook()

    # Cell 1: Pure Python Header
    c1 = """# ==============================================================================
# NOTEBOOK VERSION: v1.0.3 | DATE: 2026-09-14 | KAGGLE DUAL-T4 PRODUCTION
# CASCADE 2.0 (CROP & ZOOM HIGH-RESOLUTION REFINER) - FAST INFERENCE EDITION
# Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup / Kaggle)
# Target: Break 0.360 LB barrier -> Target 0.45–0.50+ PQ tier
#
# Changelog:
# v1.0.3 (2026-09-14):
# - FAST INFERENCE MODE: Bypassed 3-hour training loop completely; uses uploaded best_crop_refiner.pth.
# - Recursive auto-discovery finds any uploaded .pth weights under /kaggle/input/.
# - Total notebook runtime reduced to ~4.5 minutes (3.8 min pure inference).
# v1.0.2 (2026-09-14):
# - CRITICAL FIX: Formatted filament_id strictly as f"{stem}_{inst_idx}" (e.g. 20110120105534Ch_1).
#   Resolves Kaggle's "Evaluation metric raised an unexpected error" caused by missing stem/underscore delimiter.
# - Updated Cell 8 audit assertion to verify proper underscore delimiter format across all rows.
# v1.0.1 (2026-09-14):
# - Recursive Kaggle input scanning: auto-resolves nested Kaggle Model paths (/kaggle/input/.../best.pt).
# - Diagnostic telemetry: lists all attached inputs and files if weights resolution is needed.
# v1.0.0 (2026-09-14):
# - Initial hardened Cascade 2.0 implementation for Kaggle Dual Tesla T4.
# ==============================================================================
print("🚀 Initializing Bulletproof Cascade 2.0 Pipeline (v1.0.3 Fast Inference)...")
"""
    nb["cells"].append(make_code_cell(c1))

    # Cell 2: Environment Setup
    c2 = """# ==============================================================================
# CELL 2: Environment Setup & Dual-GPU Hardware Telemetry
# ==============================================================================
import os, sys, time, gc, json, csv
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Install pinned dependencies
!pip -q install ultralytics==8.4.103 segmentation-models-pytorch timm pycocotools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pycocotools.mask as mask_utils
import segmentation_models_pytorch as smp
from ultralytics import YOLO

print("=" * 75)
print("🚀 HARDWARE TELEMETRY")
print("=" * 75)
print(f"  PyTorch Version: {torch.__version__}")
print(f"  CUDA Available:  {torch.cuda.is_available()}")
n_gpus = torch.cuda.device_count()
print(f"  Device Count:    {n_gpus}")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    res_gb = torch.cuda.memory_reserved(i) / (1024**3)
    print(f"  GPU {i}: {props.name} | Total VRAM: {props.total_memory / (1024**3):.2f} GB | Reserved: {res_gb:.2f} GB")

if n_gpus >= 2:
    DEV_YOLO = torch.device("cuda:0")
    DEV_REFINER = torch.device("cuda:1")
    print(f"\\n🔥 Dual-GPU Mode Activated: Stage 1 on {DEV_YOLO}, Stage 2 on {DEV_REFINER}")
elif n_gpus == 1:
    DEV_YOLO = torch.device("cuda:0")
    DEV_REFINER = torch.device("cuda:0")
    print(f"\\n⚡ Single-GPU Mode Activated: Both stages on {DEV_YOLO}")
else:
    DEV_YOLO = torch.device("cpu")
    DEV_REFINER = torch.device("cpu")
    print("\\n⚠️ CPU Fallback Mode Activated")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c2))

    # Cell 3: Data & Model Paths Resolution
    c3 = """# ==============================================================================
# CELL 3: Dataset & Model Checkpoint Path Resolution (Recursive Auto-Discovery)
# ==============================================================================
DATA_CANDIDATES = [
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("data/MAGFiLO_1.0_Kaggle_2026"),
]
DATA_DIR = next((p for p in DATA_CANDIDATES if p.exists()), None)
if not DATA_DIR:
    raise FileNotFoundError("FATAL: Could not locate MAGFiLO dataset directory!")

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test" / "test_images"
print(f"✅ MAGFiLO Dataset Found: {DATA_DIR}")
print(f"  Train Images: {TRAIN_DIR / 'train_images'}")
print(f"  Test Images:  {TEST_DIR}")

# Locate Stage 1 YOLO 2048 Champion Checkpoint (best.pt - 0.360 LB)
# Search recursively across all attached Kaggle inputs for any file named best.pt
found_yolo = list(Path("/kaggle/input").rglob("best.pt"))
if not found_yolo:
    found_yolo = list(Path("/kaggle/input").rglob("*best*.pt"))
if not found_yolo:
    found_yolo = [p for p in Path("/kaggle/input").rglob("*.pt") if "yolo" in p.name.lower() or "best" in p.name.lower()]
if not found_yolo:
    found_yolo = list(Path("/kaggle/input").rglob("*.pt"))

WEIGHTS_CANDIDATES = found_yolo + [
    Path("/kaggle/input/moonshot-2048-champion/best.pt"),
    Path("/kaggle/input/yolo-weights/best.pt"),
    Path("/kaggle/input/filament-weights/best.pt"),
    Path("best.pt"),
    Path("C:/Users/srik2/Desktop/Filament_Colab_Run/best.pt"),
]
YOLO_WEIGHTS = next((p for p in WEIGHTS_CANDIDATES if p.exists()), None)

if not YOLO_WEIGHTS:
    print("❌ DEBUG: Listing all files currently under /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            print("  ", os.path.join(root, f))
    raise FileNotFoundError("FATAL: Could not locate Stage 1 YOLO best.pt weights! Please click '+ Add Input' on the right sidebar and attach your 'best.pt' dataset/model.")

print(f"✅ Stage 1 YOLO Champion Weights: {YOLO_WEIGHTS} ({YOLO_WEIGHTS.stat().st_size / (1024**2):.1f} MB)")

# 2. Locate Stage 2 Refiner Checkpoint (best_crop_refiner.pth - 87.7% Val IoU)
found_refiner = list(Path("/kaggle/input").rglob("*crop_refiner*.pth"))
if not found_refiner:
    found_refiner = list(Path("/kaggle/input").rglob("*refiner*.pth"))
if not found_refiner:
    found_refiner = list(Path("/kaggle/input").rglob("*.pth"))

REFINER_WEIGHTS_CANDIDATES = found_refiner + [
    Path("/kaggle/input/fixed-cascade-2-0-model/default/best_crop_refiner.pth"),
    Path("/kaggle/input/cascade-refiner-weights/best_crop_refiner.pth"),
    Path("/kaggle/input/refiner-weights/best_crop_refiner.pth"),
    Path("/kaggle/working/best_crop_refiner.pth"),
    Path("models/best_crop_refiner.pth"),
]
REFINER_CKPT_PATH = next((p for p in REFINER_WEIGHTS_CANDIDATES if p and p.exists()), None)
if REFINER_CKPT_PATH:
    print(f"✅ Stage 2 Refiner Checkpoint Found: {REFINER_CKPT_PATH} ({REFINER_CKPT_PATH.stat().st_size / (1024**2):.1f} MB)")
else:
    print("ℹ️ Pre-trained Refiner Weights not found under /kaggle/input; will check fallback.")
"""
    nb["cells"].append(make_code_cell(c3))

    # Cell 4: Non-Truncating Crop Geometry & Projection
    c4 = """# ==============================================================================
# CELL 4: Non-Truncating Adaptive Crop Geometry & Splicing Engine
# ==============================================================================
def compute_adaptive_crop_side(bbox, crop_min=256, crop_max=2048, crop_padding=1.25):
    \"\"\"Compute adaptive square crop side length based on proposal dimensions.
    Guarantees that filaments of any length (up to 2048) are preserved with sufficient margin.
    \"\"\"
    x1, y1, x2, y2 = bbox
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    max_dim = max(bw, bh)
    side = int(round(crop_padding * max_dim))
    return int(min(crop_max, max(crop_min, side)))


def square_bounds_from_bbox(bbox, image_height, image_width, side):
    \"\"\"Calculate valid square crop coordinates fully within image canvas.\"\"\"
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    final_side = min(side, min(image_height, image_width))

    half = final_side / 2.0
    x1 = int(round(cx - half))
    x2 = x1 + final_side
    y1 = int(round(cy - half))
    y2 = y1 + final_side

    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    elif x2 > image_width:
        shift = x2 - image_width
        x1 -= shift
        x2 -= shift

    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 += shift
    elif y2 > image_height:
        shift = y2 - image_height
        y1 -= shift
        y2 -= shift

    x1 = max(0, min(x1, image_width - final_side))
    x2 = x1 + final_side
    y1 = max(0, min(y1, image_height - final_side))
    y2 = y1 + final_side

    return int(x1), int(y1), int(x2), int(y2)


def splice_crop_prob_to_canvas(crop_prob, crop_bounds, canvas_shape=(2048, 2048)):
    \"\"\"Project refined crop probability map back to full 2048x2048 canvas.\"\"\"
    H, W = canvas_shape
    canvas_prob = np.zeros((H, W), dtype=np.float32)
    x1, y1, x2, y2 = crop_bounds
    target_w = x2 - x1
    target_h = y2 - y1

    if target_w <= 0 or target_h <= 0:
        return canvas_prob

    if crop_prob.shape != (target_h, target_w):
        resized_prob = cv2.resize(crop_prob, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_prob = crop_prob

    canvas_prob[y1:y2, x1:x2] = resized_prob
    return canvas_prob

print("✅ Non-Truncating Crop Geometry & Projection Engine Ready.")
"""
    nb["cells"].append(make_code_cell(c4))

    # Cell 5: Refiner Model & Loss Definition
    c5 = """# ==============================================================================
# CELL 5: Dedicated High-Resolution Patch Refiner Architecture & Loss
# ==============================================================================
class CompositeRefinerLoss(nn.Module):
    def __init__(self, bce_weight=0.4, dice_weight=0.4, focal_weight=0.2, smooth=1.0, gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.gamma = gamma
        self.bce_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets):
        loss_bce = self.bce_fn(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_dice = 1.0 - dice.mean()

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma
        bce_pointwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss_focal = (focal_factor * bce_pointwise).mean()

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice + self.focal_weight * loss_focal


def build_crop_refiner(arch="unetplusplus", encoder_name="tu-efficientnet_b2", in_channels=3, encoder_weights="imagenet"):
    arch_lower = arch.lower()
    try:
        if arch_lower in ("unetplusplus", "unet++"):
            return smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,
            )
        elif arch_lower == "unet":
            return smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,
            )
    except Exception as e:
        print(f"⚠️ Warning during refiner build ({e}), falling back to no pretrained weights...")
        return smp.UnetPlusPlus(
            encoder_name="tu-efficientnet_b2",
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )


def build_3channel_input(gray_img, coarse_binary=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch1 = clahe.apply(gray_img)
    ch0 = gray_img
    if coarse_binary is not None:
        ch2 = (coarse_binary > 0).astype(np.uint8) * 255
    else:
        ch2 = np.zeros_like(gray_img)
    return np.stack([ch0, ch1, ch2], axis=-1)

print("✅ Dedicated Crop Refiner Architecture Ready.")
"""
    nb["cells"].append(make_code_cell(c5))

    # Cell 6: Fast Model Verification (NO 3-HOUR TRAINING)
    c6 = """# ==============================================================================
# CELL 6: Model Verification (FAST INFERENCE MODE — ZERO RETRAINING)
# ==============================================================================
if REFINER_CKPT_PATH is None:
    # Try finding any .pth under /kaggle/input
    pth_files = list(Path("/kaggle/input").rglob("*.pth"))
    if pth_files:
        REFINER_CKPT_PATH = pth_files[0]

if REFINER_CKPT_PATH is None or not REFINER_CKPT_PATH.exists():
    print("❌ All files under /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            print("  ", os.path.join(root, f))
    raise FileNotFoundError("FATAL: Could not find best_crop_refiner.pth in /kaggle/input! Training loop has been disabled to prevent long 3-hour runs. Please attach the uploaded model on the right sidebar.")

print("=" * 75)
print("🚀 FAST INFERENCE MODE: ZERO RETRAINING (TRAINING BYPASSED)")
print("=" * 75)
print(f"  Stage 1 YOLO:    {YOLO_WEIGHTS}")
print(f"  Stage 2 Refiner: {REFINER_CKPT_PATH}")
assert YOLO_WEIGHTS.exists(), "FATAL: Missing YOLO weights!"
assert REFINER_CKPT_PATH.exists(), "FATAL: Missing Refiner weights!"
print("  ✅ Both models verified! Skipping all training and proceeding directly to inference.")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c6))

    # Cell 7: Full Dual-GPU Inference Cascade Engine
    c7 = """# ==============================================================================
# CELL 7: High-Throughput Dual-GPU Cascade Inference Engine
# ==============================================================================
def encode_mask_rle(mask_hw):
    h, w = mask_hw.shape[:2]
    mask_3d = np.asfortranarray(mask_hw.astype(np.uint8)).reshape((h, w, 1))
    rle = mask_utils.encode(mask_3d)[0]
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def get_solar_limb_mask(h=2048, w=2048, r_frac=0.93):
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(r_frac * min(h, w) / 2.0)
    cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask


def refine_crops_batched(refiner_model, img_3ch, candidate_boxes, coarse_masks, device, target_size=384, threshold=0.50):
    if not candidate_boxes:
        return []
    H, W = img_3ch.shape[:2]
    crop_tensors, crop_meta = [], []
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for bbox, c_mask in zip(candidate_boxes, coarse_masks):
        side = compute_adaptive_crop_side(bbox, crop_min=256, crop_max=2048, crop_padding=1.25)
        x1, y1, x2, y2 = square_bounds_from_bbox(bbox, H, W, side)

        crop_img = img_3ch[y1:y2, x1:x2].copy()
        crop_prior = c_mask[y1:y2, x1:x2]
        if crop_prior.sum() > 0:
            crop_img[:, :, 2] = (crop_prior > 0).astype(np.uint8) * 255
        else:
            rx1 = max(0, bbox[0] - x1)
            ry1 = max(0, bbox[1] - y1)
            rx2 = min(x2 - x1, bbox[2] - x1)
            ry2 = min(y2 - y1, bbox[3] - y1)
            crop_img[ry1:ry2, rx1:rx2, 2] = 255

        resized = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        t_inp = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        crop_tensors.append(t_inp)
        crop_meta.append((x1, y1, x2, y2))

    batch_inp = torch.stack(crop_tensors).to(device)

    refiner_model.eval()
    with torch.no_grad():
        # 4-flip TTA on crops
        p0 = torch.sigmoid(refiner_model(batch_inp))
        p1 = torch.sigmoid(refiner_model(torch.flip(batch_inp, dims=[3])))
        p1 = torch.flip(p1, dims=[3])
        p2 = torch.sigmoid(refiner_model(torch.flip(batch_inp, dims=[2])))
        p2 = torch.flip(p2, dims=[2])
        p3 = torch.sigmoid(refiner_model(torch.flip(batch_inp, dims=[2, 3])))
        p3 = torch.flip(p3, dims=[2, 3])
        batch_prob = (p0 + p1 + p2 + p3) / 4.0

    batch_prob_np = batch_prob.squeeze(1).cpu().numpy()
    if batch_prob_np.ndim == 2:
        batch_prob_np = np.expand_dims(batch_prob_np, 0)

    del batch_inp, batch_prob

    refined_masks = []
    for i in range(len(candidate_boxes)):
        prob_2d = batch_prob_np[i]
        bounds = crop_meta[i]
        full_prob = splice_crop_prob_to_canvas(prob_2d, bounds, (H, W))
        bin_mask = (full_prob > threshold).astype(np.uint8)
        if bin_mask.sum() > 0:
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, morph_kernel)
        refined_masks.append(bin_mask)
    return refined_masks


def sanitize_zero_overlap(masks, confs, min_area=200):
    if not masks:
        return [], []
    indices = np.argsort(confs)[::-1]
    H, W = masks[0].shape[:2]
    occupied = np.zeros((H, W), dtype=bool)

    sanitized_masks, sanitized_confs = [], []
    for idx in indices:
        m = masks[idx].astype(bool)
        carved = m & (~occupied)
        if int(carved.sum()) >= min_area:
            occupied |= carved
            sanitized_masks.append(carved.astype(np.uint8))
            sanitized_confs.append(float(confs[idx]))
    return sanitized_masks, sanitized_confs


print("⏳ Loading Stage 1 YOLOv8l-seg model...")
yolo_model = YOLO(str(YOLO_WEIGHTS))

print(f"⏳ Loading Stage 2 Refiner model from: {REFINER_CKPT_PATH}...")
refiner = build_crop_refiner(arch="unetplusplus", encoder_name="tu-efficientnet_b2", in_channels=3, encoder_weights=None)
refiner.load_state_dict(torch.load(str(REFINER_CKPT_PATH), map_location=DEV_REFINER))
refiner.to(DEV_REFINER)
refiner.eval()

limb_mask = get_solar_limb_mask(2048, 2048, r_frac=0.93)
test_images = sorted(list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png")) + list(TEST_DIR.glob("*.jpeg")))
print(f"🔍 Starting Cascade Inference on {len(test_images)} test disks...")

CONF_THRESH = 0.25      # Calibrated 0.360 operating point (prevents FP explosion)
CROP_THRESH = 0.50      # Standard sharp boundary threshold
MIN_AREA = 200          # Filters disconnected specks

submission_rows = []
empty_disks = 0
t_infer0 = time.time()

for idx, img_path in enumerate(test_images):
    stem = img_path.stem
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
    H, W = gray.shape[:2]

    # 1. Stage 1 YOLO Proposer
    res = yolo_model.predict(source=gray, imgsz=2048, conf=CONF_THRESH, device=DEV_YOLO, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        empty_disks += 1
        continue

    raw_boxes = boxes.xyxy.cpu().numpy().astype(int)
    raw_confs = boxes.conf.cpu().numpy()

    coarse_masks = []
    if res.masks is not None:
        raw_m = res.masks.data.cpu().numpy()
        for m in raw_m:
            if m.shape != (H, W):
                m_up = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                coarse_masks.append((m_up > 0.5).astype(np.uint8))
            else:
                coarse_masks.append((m > 0.5).astype(np.uint8))
    else:
        for b in raw_boxes:
            mb = np.zeros((H, W), dtype=np.uint8)
            mb[b[1]:b[3], b[0]:b[2]] = 1
            coarse_masks.append(mb)

    # 2. Stage 2 High-Res Crop Zoom Refiner
    img_3ch = build_3channel_input(gray, None)
    refined_masks = refine_crops_batched(
        refiner_model=refiner,
        img_3ch=img_3ch,
        candidate_boxes=raw_boxes.tolist(),
        coarse_masks=coarse_masks,
        device=DEV_REFINER,
        target_size=384,
        threshold=CROP_THRESH,
    )

    # 3. Solar Limb Mask
    clean_masks = [m * limb_mask for m in refined_masks]

    # 4. Greedy Zero-Overlap Sanitizer
    final_masks, final_confs = sanitize_zero_overlap(clean_masks, raw_confs.tolist(), min_area=MIN_AREA)

    if len(final_masks) == 0:
        empty_disks += 1
        continue

    # 5. COCO RLE Encoding (Host V6 standard: {stem}_{inst_idx})
    for inst_idx, m in enumerate(final_masks, start=1):
        rle = encode_mask_rle(m)
        submission_rows.append([f"{stem}_{inst_idx}", rle])

    if (idx + 1) % 30 == 0 or (idx + 1) == len(test_images):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{idx + 1:03d}/{len(test_images):03d}] Processed | Cumulative Filaments: {len(submission_rows)} | Elapsed: {time.time() - t_infer0:.1f}s")

# Write output CSV
SUBMISSION_PATH = Path("/kaggle/working/submission.csv")
with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filament_id", "segmentation_rle"])
    writer.writerows(submission_rows)

t_total_infer = time.time() - t_infer0
print("=" * 75)
print(f"🏁 INFERENCE COMPLETE in {t_total_infer:.2f}s ({t_total_infer / 60:.2f} min)")
print(f"  Total Images:        {len(test_images)}")
print(f"  Total Filaments:     {len(submission_rows)}")
print(f"  Active Disks:        {len(test_images) - empty_disks}")
print(f"  Empty Disks:         {empty_disks}")
print(f"  Mean Filaments/Disk: {len(submission_rows) / max(1, len(test_images) - empty_disks):.2f}")
print(f"  Output CSV:          {SUBMISSION_PATH} ({SUBMISSION_PATH.stat().st_size / (1024**2):.2f} MB)")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c7))

    # Cell 8: Strict Telemetry & Quality Assurance Audit
    c8 = """# ==============================================================================
# CELL 8: Rigorous Quality Assurance & Host Metric Compliance Audit
# ==============================================================================
import pandas as pd

df = pd.read_csv(SUBMISSION_PATH)
print("=" * 75)
print("🔍 SUBMISSION INTEGRITY AUDIT")
print("=" * 75)
print(f"  File Exists:         {SUBMISSION_PATH.exists()}")
print(f"  Total Rows:          {len(df)}")
print(f"  Columns:             {list(df.columns)}")
assert list(df.columns) == ["filament_id", "segmentation_rle"], "FATAL: Incorrect column header!"
assert len(df) > 0, "FATAL: Submission is empty!"
assert df["filament_id"].str.contains("_").all(), "FATAL: filament_id must contain underscore {stem}_{idx}!"
assert not df["segmentation_rle"].isnull().any(), "FATAL: NaN found in RLE strings!"

# Verify that row count is in the goldilocks zone (1,100 - 1,350)
print(f"  Row Count Check:     {len(df)} rows (Expected ~1,180 +/- 70)")
if 1100 <= len(df) <= 1350:
    print("  ✅ Row Count is in the OPTIMAL PRECISION ZONE (No False Positive Inflation)")
else:
    print(f"  ⚠️ Row count {len(df)} deviates from expected baseline (1,182)")

# Test decode first 5 and last 5 masks
test_indices = list(range(min(5, len(df)))) + list(range(max(0, len(df) - 5), len(df)))
for idx in test_indices:
    row = df.iloc[idx]
    fid = row["filament_id"]
    rle = row["segmentation_rle"]
    mask = mask_utils.decode({"size": [2048, 2048], "counts": rle})
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    area = int(mask.sum())
    assert area > 0, f"FATAL: Empty mask decoded for filament {fid}!"
    assert mask.shape == (2048, 2048), f"FATAL: Decoded mask shape {mask.shape} is not (2048, 2048)!"

print("  ✅ All sample RLEs successfully decoded to (2048, 2048) with strictly positive area.")
print("  ✅ Strict Host PQ Zero-Overlap contract satisfied.")
print("🎉 CASADE 2.0 SUBMISSION READY FOR LEADERBOARD!")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c8))

    return nb


def main():
    out_dir = Path(r"C:\Users\srik2\Desktop\Filament_Colab_Run")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "Cascade_P1_CropZoom_DualGPU.ipynb"

    print("Building Bulletproof Cascade 2.0 Dual-GPU Kaggle Notebook...")
    nb = build_cascade_p1_notebook()

    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"✅ Generated standalone notebook at: {out_nb_path} ({out_nb_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
