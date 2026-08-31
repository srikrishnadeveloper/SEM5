# =============================================================================
# Solar Filament Segmentation Challenge 2026 - SINGLE FILE TOP-50 PIPELINE
# =============================================================================
# Everything in one file: data caching, training, TTA inference, OOF threshold
# search, post-processing and submission.
#
# Why this is faster than a plain 5-fold U-Net++ notebook:
#   1. Images and masks are decoded ONCE into uint8 memmap caches. Every epoch
#      after that reads from RAM/disk instead of decoding 2048x2048 JPEGs and
#      re-rasterising COCO polygons. This alone is a ~5-10x data-loading win.
#   2. Train, validation and inference all run at ONE resolution, so there is no
#      scale mismatch and no wasted compute.
#   3. AMP + channels_last + cudnn.benchmark + gradient accumulation.
#   4. The expensive competition metric is computed on a cheap downsampled grid
#      during training and only at full precision when it matters.
#
# Why it should score higher:
#   1. 3-channel ImageNet input (keeps pretrained stem filters).
#   2. BCE(pos_weight) + Dice + Focal Tversky loss, tuned for ~0.2% positives.
#   3. EMA weights + warmup/cosine schedule.
#   4. Model selection directly on mIoU_multiscale (the real metric).
#   5. D4 (8-way) TTA at inference.
#   6. Threshold + min-area grid search on out-of-fold predictions.
#   7. Solar-disk masking, hole filling and small-object removal.
#   8. Probability maps saved as .npy so folds can be ensembled.
# =============================================================================

import gc
import glob
import json
import math
import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import albumentations as A
import pycocotools.mask as mask_util
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)  # avoid thread thrash inside DataLoader workers


# =============================================================================
# 1. CONFIG  -- edit this block only
# =============================================================================
class CFG:
    # ---- paths -------------------------------------------------------------
    base = os.environ.get("FILAMENT_BASE_PATH", "./data/MAGFiLO_1.0_Kaggle_2026")
    out_dir = os.environ.get("FILAMENT_OUT_DIR", "./filament_out")

    # ---- speed / quality preset -------------------------------------------
    # "max"      ~3 h/fold on a T4, TOP-50 competitive score (U-Net++ b4 @ 1024px)
    # "balanced" ~70 min/fold on a T4, solid fast model (U-Net b4 @ 1024px)
    preset = os.environ.get("FILAMENT_PRESET", "max")

    # ---- cross validation --------------------------------------------------
    n_folds = int(os.environ.get("FILAMENT_N_FOLDS", 5))
    fold = int(os.environ.get("FILAMENT_VAL_FOLD", 0))
    seed = int(os.environ.get("FILAMENT_SEED", 2026))

    # ---- these are set by apply_preset() -----------------------------------
    arch = "unetplusplus"
    encoder = "timm-efficientnet-b4"
    img_size = 1024
    batch_size = 1
    accum = 8
    epochs = 35

    # ---- fixed ------------------------------------------------------------
    full_res = 2048
    in_channels = 3
    lr = 3e-4
    min_lr = 1e-6
    weight_decay = 1e-5
    warmup_epochs = 2
    patience = 50              # full training without premature early stopping
    use_amp = True
    use_ema = True
    ema_decay = 0.999
    grad_checkpoint = True
    num_workers = 2
    heavy_aug = True
    use_copy_paste = False
    copy_paste_prob = 0.4
    copy_paste_min_area = 64
    imagenet_norm = True       # use ImageNet (0.485,0.456,0.406) stats — matches pretrained encoder BN stats
    decoder_attention = "scse"  # "scse" | "none" — spatial+channel attention in UNet/UNet++ decoders

    # loss weights — BCE dominates to counteract class imbalance; Dice+Tversky for overlap
    w_bce, w_dice, w_tversky = 0.40, 0.35, 0.25
    w_boundary = 0.0              # disabled: unstable under AMP and tiny contribution
    w_cldice = 0.0                # disabled: 10-iter soft-skel produces NaN spikes under fp16
    cldice_iter = 3
    tversky_alpha, tversky_beta, tversky_gamma = 0.3, 0.7, 1.3333
    pos_weight_cap = 15.0         # was 50 — too high causes BCE vs Dice gradient conflict

    # validation metric
    select_metric = "dice"          # "dice" | "miou" | "iou" (dice tracks continuous mask learning)
    miou_eval_res = 512            # downsample for the per-epoch metric (speed)

    # inference / post-processing
    tta_d4 = True
    tta_scales = (0.85, 1.0, 1.15) # multi-scale TTA: e.g. (0.85, 1.0, 1.15)
    threshold = 0.45
    min_area = 200
    max_area = 500_000
    morph_k = 3
    fill_holes = 128
    watershed_split = True
    no_filament_prob = 0.10
    use_crf = False             # requires pydensecrf; slow but can improve boundaries
    crf_sxy = 3                 # Gaussian spatial std
    crf_srgb = 13               # Bilateral colour std
    crf_compat = 10
    crf_iter = 5

    # threshold search grid
    search_thresholds = np.arange(0.20, 0.76, 0.05)
    search_min_areas = (50, 100, 200, 400, 800)

    save_probs = True


PRESETS = {
    "fast":     dict(arch="unet",         encoder="timm-efficientnet-b2", img_size=768,  batch_size=4, accum=2, epochs=20, tta_scales=(1.0,)),
    "balanced": dict(arch="unet",         encoder="timm-efficientnet-b4", img_size=1024, batch_size=2, accum=4, epochs=30, tta_scales=(0.9, 1.0, 1.1)),
    "max":      dict(arch="unetplusplus", encoder="timm-efficientnet-b4", img_size=1024, batch_size=1, accum=8, epochs=35, tta_scales=(0.85, 1.0, 1.15)),
    # in_channels=1: EdgeAttNet has no ImageNet-pretrained stem, so there is no
    # benefit to replicating the grayscale channel 3x (only extra compute).
    "edgeattnet": dict(arch="edgeattnet", encoder="none", img_size=512, batch_size=4, accum=4,
                       epochs=40, tta_scales=(0.9, 1.0, 1.1), in_channels=1),
}


def apply_preset(cfg=CFG, preset=None):
    """Apply a speed/quality preset. Call this BEFORE any manual override,
    because it overwrites arch/encoder/img_size/batch_size/accum/epochs."""
    name = preset or cfg.preset
    for k, v in PRESETS.get(name, PRESETS["max"]).items():
        setattr(cfg, k, v)
    cfg.preset = name
    return resolve(cfg)


def resolve(cfg=CFG):
    """Derive all paths from the current config. Idempotent and never
    overwrites model hyper-parameters, so it is safe to call after edits."""
    # cache at the training resolution so no resize happens per sample
    cfg.cache_res = cfg.img_size
    cfg.train_dir = os.path.join(cfg.base, "train")
    cfg.test_dir = os.path.join(cfg.base, "test")
    cfg.train_images = os.path.join(cfg.train_dir, "train_images")
    cfg.test_images = os.path.join(cfg.test_dir, "test_images")
    cfg.train_json = os.path.join(cfg.train_dir, "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    cfg.cache_dir = os.path.join(cfg.out_dir, f"cache_{cfg.cache_res}")
    cfg.model_dir = os.path.join(cfg.out_dir, "models")
    cfg.sub_dir = os.path.join(cfg.out_dir, "submissions")
    cfg.prob_dir = os.path.join(cfg.out_dir, "probs")
    for d in (cfg.out_dir, cfg.cache_dir, cfg.model_dir, cfg.sub_dir, cfg.prob_dir):
        os.makedirs(d, exist_ok=True)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # variable-free shapes -> safe and fast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_info(msg: str):
    """Flushed log message with human-readable timestamp, safe across all terminal encodings."""
    t_str = time.strftime("%H:%M:%S")
    text = f"[{t_str}] {msg}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        print(safe_text, flush=True)


# =============================================================================
# 1.5 SOLAR PREPROCESSING (EdgeAttNet-style)
# =============================================================================
# EdgeAttNet normalises, masks the disk, corrects limb darkening by radial
# flattening, smooths with a 3x3 Gaussian, then applies CLAHE inside the disk.
# Doing this once at full 2048 before resize/augmentation matches their pipeline.

def solar_disk_hough(gray, fallback_threshold=10):
    """Estimate the solar disk with Hough circles, falling back to a
    threshold-based connected component if Hough fails. Uses a fast thumbnail."""
    if gray is None or gray.size == 0:
        return None
    h, w = gray.shape[:2]
    scale = 256.0 / max(h, w)
    small_h, small_w = int(round(h * scale)), int(round(w * scale))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)

    min_dim = min(small_h, small_w)
    circles = None
    for dp in (2, 1, 3):
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dim // 2,
                                   param1=50, param2=50, minRadius=min_dim // 4,
                                   maxRadius=min_dim // 2 - 2)
        if circles is not None and len(circles[0]) > 0:
            break
    if circles is not None and len(circles[0]) > 0:
        x, y, r = circles[0][0]
        cx, cy, cr = int(round(x / scale)), int(round(y / scale)), int(round(r / scale))
        yy, xx = np.ogrid[:h, :w]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= cr * cr).astype(np.uint8)
        return dict(mask=disk, center=(cy, cx), radius=cr)

    # fallback on thumbnail
    _, b = cv2.threshold(small, fallback_threshold, 255, cv2.THRESH_BINARY)
    num, lab = cv2.connectedComponents(b)
    if num <= 1:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    disk_small = (lab == int(sizes.argmax())).astype(np.uint8)
    disk_small = cv2.morphologyEx(disk_small, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ys, xs = np.where(disk_small)
    if len(xs) == 0:
        return None
    cy = int(round(ys.mean() / scale))
    cx = int(round(xs.mean() / scale))
    cr = int(round(np.sqrt(((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2).max()) / scale))
    yy, xx = np.ogrid[:h, :w]
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= cr * cr).astype(np.uint8)
    return dict(mask=disk, center=(cy, cx), radius=cr)


def radial_flatten(gray_f, disk_info, n_bins=64, smooth_win=7):
    """Correct limb-darkening by dividing by a radial intensity profile.
    Vectorized with np.bincount for speed."""
    cy, cx = disk_info["center"]
    r = disk_info["radius"]
    h, w = gray_f.shape
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1)

    mask = disk_info["mask"].astype(bool)
    if not mask.any():
        return gray_f

    vals = gray_f[mask]
    dists = d[mask]
    bins = np.linspace(0, 1.2, n_bins + 1)
    bin_idx = np.clip(np.digitize(dists, bins) - 1, 0, n_bins - 1)
    bin_sums = np.bincount(bin_idx, weights=vals, minlength=n_bins)
    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = np.where(bin_counts > 0, bin_sums / np.maximum(bin_counts, 1), 0.0)

    # Smooth profile
    valid = profile > 0
    if valid.any():
        nz = np.where(valid)[0]
        if len(nz) > 1:
            for i in range(n_bins):
                if not valid[i]:
                    j = nz[np.argmin(np.abs(nz - i))]
                    profile[i] = profile[j]
    if smooth_win > 2 and len(profile) >= smooth_win:
        profile = savgol_filter(profile, smooth_win, 2)
    profile = np.maximum(profile, 1e-6)

    bin_idx2 = np.clip(np.digitize(d, bins) - 1, 0, n_bins - 1)
    bg = profile[bin_idx2]
    mean = vals.mean()
    out = np.zeros_like(gray_f)
    out[mask] = np.clip(gray_f[mask] / bg[mask] * mean, 0.0, 1.0)
    return out


def preprocess_solar(gray, do_clahe=True, do_flatten=True):
    """EdgeAttNet-style full-disk preprocessing.
    Input:  uint8 grayscale (any HxW)
    Output: uint8 grayscale, disk_info dict
    """
    if gray is None or gray.size == 0:
        return gray, None
    disk_info = solar_disk_hough(gray)
    if disk_info is None:
        disk_info = dict(mask=np.ones_like(gray, np.uint8), center=(gray.shape[0] // 2, gray.shape[1] // 2), radius=min(gray.shape) // 2)

    img = gray.astype(np.float32) / 255.0
    if do_flatten:
        img = radial_flatten(img, disk_info)
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    if do_clahe:
        img8 = (img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img8 = clahe.apply(img8)
        img = img8.astype(np.float32) / 255.0
    img *= disk_info["mask"].astype(np.float32)
    out = (img * 255).astype(np.uint8)
    return out, disk_info


# =============================================================================
# 2. ANNOTATIONS -> MASKS
# =============================================================================
def group_annotations(json_path, drop_ambiguous=True):
    """MAGFiLO has 1154 image_ids for only ~707 physical files (multi-annotator).
    Group every annotation by FILENAME so folds cannot leak."""
    with open(json_path, "r") as f:
        coco = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    file_to_anns = defaultdict(list)
    for fn in id_to_file.values():
        file_to_anns.setdefault(fn, [])
    for ann in coco["annotations"]:
        if drop_ambiguous and ann.get("category_id") == 4:
            continue
        fn = id_to_file.get(ann["image_id"])
        if fn is not None:
            file_to_anns[fn].append(ann)
    return dict(file_to_anns)


def polygons_to_mask(anns, h, w, scale=1.0):
    """Rasterise all polygons of one image into a single binary mask."""
    mask = np.zeros((h, w), np.uint8)
    for ann in anns:
        segs = ann.get("segmentation")
        if not isinstance(segs, list):
            continue
        for seg in segs:
            if not isinstance(seg, (list, tuple)) or len(seg) < 6:
                continue
            pts = np.asarray(seg, np.float64).reshape(-1, 2) * scale
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def make_folds(files, n_folds, seed):
    """GroupKFold by observation year."""
    files = sorted(files)
    years = [os.path.basename(f)[:4] for f in files]
    uniq = sorted(set(years))
    rng = random.Random(seed)
    shuffled = uniq[:]
    rng.shuffle(shuffled)
    sizes = {y: years.count(y) for y in shuffled}
    fold_of_year, load = {}, [0] * n_folds
    for y in sorted(shuffled, key=lambda y: -sizes[y]):
        k = int(np.argmin(load))
        fold_of_year[y] = k
        load[k] += sizes[y]
    return [fold_of_year[y] for y in years], files


# =============================================================================
# 3. ONE-TIME CACHE (Fast Vectorized Processing)
# =============================================================================
def build_cache(cfg):
    """Decode every JPEG + rasterise every mask directly at cache_res into uint8 memmaps."""
    meta_path = os.path.join(cfg.cache_dir, "meta.json")
    img_path = os.path.join(cfg.cache_dir, "images.npy")
    msk_path = os.path.join(cfg.cache_dir, "masks.npy")
    if os.path.exists(meta_path) and os.path.exists(img_path) and os.path.exists(msk_path):
        meta = json.load(open(meta_path))
        if meta.get("res") == cfg.cache_res:
            log_info(f"[Cache] Reusing existing cache in {cfg.cache_dir} ({meta['n']} images @ {meta['res']}x{meta['res']})")
            return meta

    file_to_anns = group_annotations(cfg.train_json)
    files = sorted(f for f in file_to_anns if os.path.exists(os.path.join(cfg.train_images, f)))
    if not files:
        raise FileNotFoundError(f"no training images found under {cfg.train_images}")
    r = cfg.cache_res
    n = len(files)
    log_info(f"[Cache] Building memory-mapped cache @ {r}x{r} for {n} training images...")
    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8, shape=(n, r, r))
    masks = np.lib.format.open_memmap(msk_path, mode="w+", dtype=np.uint8, shape=(n, r, r))

    pos_pixels = 0
    log_interval = max(1, n // 5)
    for i, fn in enumerate(tqdm(files, desc=f"[cache] building @{r}")):
        img = cv2.imread(os.path.join(cfg.train_images, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(os.path.join(cfg.train_images, fn))
        h, w = img.shape[:2]
        # Preprocess at FULL resolution first (match inference path exactly),
        # THEN resize to cache resolution. This is critical: Hough circle
        # detection and radial flattening produce completely different results
        # at 768 vs 2048.
        img_prep, _ = preprocess_solar(img)
        
        if (h, w) != (r, r):
            img_r = cv2.resize(img_prep, (r, r), interpolation=cv2.INTER_AREA)
        else:
            img_r = img_prep
        
        # Direct rasterization at r x r with scaling
        scale_x, scale_y = r / float(w), r / float(h)
        mask_r = np.zeros((r, r), np.uint8)
        for ann in file_to_anns[fn]:
            segs = ann.get("segmentation")
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if not isinstance(seg, (list, tuple)) or len(seg) < 6:
                    continue
                pts = np.asarray(seg, np.float64).reshape(-1, 2)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                cv2.fillPoly(mask_r, [np.round(pts).astype(np.int32)], 1)
                
        images[i] = img_r
        masks[i] = mask_r
        pos_pixels += int(mask_r.sum())
        if (i + 1) % log_interval == 0 or (i + 1) == n:
            log_info(f"  [Cache] Processed {i+1:>3}/{n} images ({(i+1)/n*100:.1f}%) | Pos pixel count: {pos_pixels:,}")

    images.flush(); masks.flush()
    del images, masks; gc.collect()
    fold_ids, files = make_folds(files, cfg.n_folds, cfg.seed)
    meta = {"res": r, "n": n, "files": files, "folds": fold_ids,
            "pos_frac": pos_pixels / float(n * r * r)}
    json.dump(meta, open(meta_path, "w"))
    log_info(f"[Cache] Complete: {n} images @ {r}x{r}, positive pixel fraction: {meta['pos_frac']:.5f}")
    return meta


def _downsample_mask_max(mask, out_res):
    """Max-pool style downsample: never loses a thin structure."""
    h = mask.shape[0]
    if h == out_res:
        return mask.astype(np.uint8)
    if h % out_res == 0:
        f = h // out_res
        return mask.reshape(out_res, f, out_res, f).max(axis=(1, 3)).astype(np.uint8)
    f = max(1, int(round(h / out_res)))
    dil = cv2.dilate(mask, np.ones((f, f), np.uint8))
    return (cv2.resize(dil, (out_res, out_res), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


# =============================================================================
# 4. DATASET
# =============================================================================
def build_transforms(cfg, train):
    """Images stay uint8 through augmentation so CLAHE/ISONoise are legal,
    then Normalize converts to float. This ordering is a common crash source."""
    ops = []
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.2), translate_percent=(-0.06, 0.06),
                     rotate=(-30, 30), border_mode=cv2.BORDER_CONSTANT, p=0.7),
        ]
        if cfg.heavy_aug:
            try:
                gauss = A.GaussNoise(std_range=(0.01, 0.05), p=0.2)
            except TypeError:
                gauss = A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
            ops += [
                A.OneOf([A.ElasticTransform(alpha=40, sigma=8),
                         A.GridDistortion(num_steps=5, distort_limit=0.2)], p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                gauss,
            ]
    if cfg.imagenet_norm:
        mean = (0.485, 0.456, 0.406)[:cfg.in_channels]
        std = (0.229, 0.224, 0.225)[:cfg.in_channels]
        ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    else:
        ops += [A.Normalize(mean=(0.5,) * cfg.in_channels, std=(0.5,) * cfg.in_channels),
                ToTensorV2()]
    return A.Compose(ops)


class FilamentDataset(Dataset):
    def __init__(self, cfg, meta, indices, train):
        self.cfg, self.indices, self.train = cfg, list(indices), train
        self.images = np.load(os.path.join(cfg.cache_dir, "images.npy"), mmap_mode="r")
        self.masks = np.load(os.path.join(cfg.cache_dir, "masks.npy"), mmap_mode="r")
        self.files = meta["files"]
        self.tf = build_transforms(cfg, train)

    def __len__(self):
        return len(self.indices)

    def _load_sample(self, idx, for_copy=False):
        img = np.asarray(self.images[idx])                      # uint8 (H, W)
        mask = np.asarray(self.masks[idx]).astype(np.uint8)     # uint8 (H, W)
        if self.cfg.in_channels == 3:
            img = np.repeat(img[:, :, None], 3, axis=2)
        else:
            img = img[:, :, None]
        if for_copy:
            # deterministic spatial crop/flip so pasted objects are not too distorted
            s = self.tf(image=img, mask=mask)
        else:
            s = self.tf(image=img, mask=mask)
        x = s["image"].float()
        y = s["mask"]
        y = y.float().unsqueeze(0) if y.ndim == 2 else y.float()
        return x, y

    def _random_copy_paste(self, x, y):
        """Paste a single filament instance from a random donor image."""
        cfg = self.cfg
        H, W = y.shape[-2:]
        donor_idx = random.choice(self.indices)
        # Do not use the same index more than a few times to avoid leakage
        dx, dy = self._load_sample(donor_idx, for_copy=True)
        # find a donor component large enough
        dy_bin = (dy[0] > 0.5).cpu().numpy() if isinstance(dy, torch.Tensor) else (dy[0] > 0.5)
        lab, n = ndimage.label(dy_bin)
        comps = [(lab == k) for k in range(1, n + 1) if (lab == k).sum() >= cfg.copy_paste_min_area]
        if not comps:
            return x, y
        comp = random.choice(comps)
        ys, xs = np.where(comp)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        comp_patch = comp[y0:y1, x0:x1]
        dx_patch = dx[:, y0:y1, x0:x1]
        dy_patch = dy[:, y0:y1, x0:x1]
        ph, pw = comp_patch.shape
        # optional random flip/rotate of the patch
        if random.random() < 0.5:
            comp_patch = np.flip(comp_patch, axis=1)
            dx_patch = torch.flip(dx_patch, dims=[2])
            dy_patch = torch.flip(dy_patch, dims=[2])
        if random.random() < 0.5:
            comp_patch = np.flip(comp_patch, axis=0)
            dx_patch = torch.flip(dx_patch, dims=[1])
            dy_patch = torch.flip(dy_patch, dims=[1])
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            comp_patch = np.rot90(comp_patch, k)
            dx_patch = torch.rot90(dx_patch, k, [1, 2])
            dy_patch = torch.rot90(dy_patch, k, [1, 2])
            # shape may have swapped
            ph, pw = comp_patch.shape
        # random placement
        max_ty = max(1, H - ph)
        max_tx = max(1, W - pw)
        ty = random.randint(0, max_ty - 1)
        tx = random.randint(0, max_tx - 1)
        # paste: object pixels only. np.flip/np.rot90 return negative-stride
        # views (torch.from_numpy rejects those, hence ascontiguousarray),
        # and torch.where() needs a BOOL condition, not x.dtype (float).
        comp_t = torch.from_numpy(np.ascontiguousarray(comp_patch)).to(x.device).bool()
        for c in range(x.shape[0]):
            target = x[c, ty:ty + ph, tx:tx + pw]
            x[c, ty:ty + ph, tx:tx + pw] = torch.where(
                comp_t, dx_patch[c, :ph, :pw].to(x.device, x.dtype), target)
        y[0, ty:ty + ph, tx:tx + pw] = torch.where(
            comp_t, dy_patch[0, :ph, :pw].to(y.device, y.dtype), y[0, ty:ty + ph, tx:tx + pw])
        return x, y

    def __getitem__(self, i):
        idx = self.indices[i]
        x, y = self._load_sample(idx)
        if self.train and self.cfg.use_copy_paste and random.random() < self.cfg.copy_paste_prob:
            x, y = self._random_copy_paste(x, y)
        return x, y, os.path.splitext(self.files[idx])[0]


def build_loaders(cfg, meta):
    folds = np.asarray(meta["folds"])
    tr_idx = np.where(folds != cfg.fold)[0]
    va_idx = np.where(folds == cfg.fold)[0]
    if len(va_idx) == 0:  # degenerate fold -> fall back to a deterministic slice
        va_idx = np.arange(0, len(folds), max(1, len(folds) // 5))
        tr_idx = np.setdiff1d(np.arange(len(folds)), va_idx)
    tr = FilamentDataset(cfg, meta, tr_idx, True)
    va = FilamentDataset(cfg, meta, va_idx, False)
    common = dict(num_workers=cfg.num_workers, pin_memory=(DEVICE.type == "cuda"))
    if cfg.num_workers > 0:
        common.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common)
    # Validation uses no gradients, so we can fit ~4x larger batch for faster eval
    val_bs = max(4, cfg.batch_size * 4)
    val_loader = DataLoader(va, batch_size=val_bs, shuffle=False, **common)
    print(f"[data] fold {cfg.fold}: {len(tr)} train / {len(va)} val @ {cfg.img_size} (val_bs={val_bs})")
    return train_loader, val_loader


# =============================================================================
# 5. MODEL
# =============================================================================
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _EdgeAttnBlock(nn.Module):
    """Edge-Guided Multi-Head Self-Attention from the EdgeAttNet paper.
    Edge priors are added to the Query and Key before attention; Value stays
    unchanged.  No positional encodings are used."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                         batch_first=True, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, edge):
        # x, edge: (B, N, C)
        q = x + edge
        k = x + edge
        v = x
        out, _ = self.mha(q, k, v, need_weights=False)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class UNetEdgeTransformer(nn.Module):
    """Re-implementation of the EdgeAttNet architecture from arXiv:2509.02964.
    4-stage U-Net encoder/decoder with two edge-guided MHSA blocks at the
    bottleneck.  Trained from scratch, no ImageNet pretraining."""
    def __init__(self, in_channels=1, channels=(64, 128, 256, 512), drop=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.enc1 = _DoubleConv(in_channels, channels[0], drop)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(channels[0], channels[1], drop)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _DoubleConv(channels[1], channels[2], drop)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _DoubleConv(channels[2], channels[3], drop)
        self.pool4 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck_conv = _DoubleConv(channels[3], channels[3], drop)
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.edge_proj = nn.Conv2d(1, channels[3], 1)
        self.eg_mhsa = nn.ModuleList([
            _EdgeAttnBlock(channels[3], num_heads=4, dropout=drop) for _ in range(2)
        ])

        # decoder: transposed-conv + skip concatenation + double conv
        self.up4 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec4 = _DoubleConv(channels[2] + channels[3], channels[2], drop)
        self.up3 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec3 = _DoubleConv(channels[1] + channels[2], channels[1], drop)
        self.up2 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec2 = _DoubleConv(channels[0] + channels[1], channels[0], drop)
        self.up1 = nn.ConvTranspose2d(channels[0], channels[0], 2, stride=2)
        self.dec1 = _DoubleConv(channels[0] + channels[0], channels[0], drop)
        self.final = nn.Conv2d(channels[0], 1, 1)

    def _to_flat(self, t):
        B, C, H, W = t.shape
        return t.reshape(B, C, -1).permute(0, 2, 1), (H, W)

    def forward(self, x):
        # x: (B, C, H, W), must be divisible by 16
        x1 = self.enc1(x); p1 = self.pool1(x1)
        x2 = self.enc2(p1); p2 = self.pool2(x2)
        x3 = self.enc3(p2); p3 = self.pool3(x3)
        x4 = self.enc4(p3); p4 = self.pool4(x4)

        # bottleneck
        z = self.bottleneck_conv(p4)
        # extract edge map from the original image and resize to bottleneck size
        edge = self.edge_extractor(x)
        edge = F.interpolate(edge, size=z.shape[-2:], mode="bilinear", align_corners=False)
        edge = self.edge_proj(edge)
        zf, (H, W) = self._to_flat(z)
        ef, _ = self._to_flat(edge)
        for blk in self.eg_mhsa:
            zf = blk(zf, ef)
        z = zf.permute(0, 2, 1).reshape(-1, z.shape[1], H, W)

        # decoder with skip connections
        d4 = self.up4(z)
        d4 = self._pad_cat(d4, x4)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = self._pad_cat(d3, x3)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, x2)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self._pad_cat(d1, x1)
        d1 = self.dec1(d1)
        return self.final(d1)

    def _pad_cat(self, small, big):
        diff_h = big.shape[2] - small.shape[2]
        diff_w = big.shape[3] - small.shape[3]
        if diff_h or diff_w:
            small = F.pad(small, [diff_w // 2, diff_w - diff_w // 2,
                                  diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([small, big], dim=1)


def build_model(cfg):
    if cfg.arch.lower() == "edgeattnet":
        net = UNetEdgeTransformer(in_channels=cfg.in_channels)
        n_par = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"[model] edgeattnet in_channels={cfg.in_channels}: {n_par:.1f}M params")
        return net

    kwargs = dict(encoder_name=cfg.encoder, encoder_weights="imagenet",
                  in_channels=cfg.in_channels, classes=1, activation=None)
    if cfg.decoder_attention.lower() != "none" and cfg.arch.lower() in ("unet", "unetplusplus"):
        try:
            kwargs["decoder_attention_type"] = cfg.decoder_attention
        except Exception:
            pass
    factory = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus,
               "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus}
    fn = factory.get(cfg.arch.lower(), smp.Unet)
    try:
        net = fn(**kwargs)
    except Exception as e:  # no internet for pretrained weights or unsupported arg
        print(f"[model] first init failed ({e}); retrying without decoder attention / pretrained")
        kwargs.pop("decoder_attention_type", None)
        kwargs["encoder_weights"] = None
        net = fn(**kwargs)
    if cfg.grad_checkpoint:
        enc = getattr(net, "encoder", None)
        if hasattr(enc, "set_grad_checkpointing"):
            try:
                enc.set_grad_checkpointing(True)
                print("[model] gradient checkpointing enabled")
            except Exception:
                pass
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[model] {cfg.arch} + {cfg.encoder}: {n_par:.1f}M params")
    return net


class EMA:
    """Plain EMA of float parameters and buffers. Avoids AveragedModel's
    BN-statistics pitfalls by copying buffers straight across."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.other = {k: v.detach().clone()
                      for k, v in model.state_dict().items()
                      if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.other[k] = v.detach().clone()

    def state_dict(self):
        sd = {k: v.clone() for k, v in self.shadow.items()}
        sd.update({k: v.clone() for k, v in self.other.items()})
        return sd

    def copy_to(self, model):
        model.load_state_dict(self.state_dict(), strict=True)


# =============================================================================
# 6. LOSS
# =============================================================================
class TverskyFocalLoss(nn.Module):
    """Focal Tversky. beta > alpha penalises false negatives, which is what you
    want when only ~0.2% of pixels are filament."""
    def __init__(self, alpha, beta, gamma, smooth=1.0):
        super().__init__()
        self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tv = (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)
        return torch.pow(1.0 - tv, self.g).mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.s = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        num = 2 * (p * t).sum(1) + self.s
        den = p.sum(1) + t.sum(1) + self.s
        return (1 - num / den).mean()


class BoundaryLoss(nn.Module):
    """Sobel-edge L1 loss. Pushes predicted probability boundaries to align
    with the ground-truth boundaries, helping with thin filament barbs."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", sobel_x)
        self.register_buffer("ky", sobel_y)

    def grad_norm(self, x):
        # x: (B,1,H,W) probabilities or binary masks
        gx = F.conv2d(x, self.kx.to(x.device), padding=1)
        gy = F.conv2d(x, self.ky.to(x.device), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ge = self.grad_norm(target)
        pe = self.grad_norm(p)
        # L1 on edge magnitude; normalise by the number of edge pixels
        num = (ge > 0).float().sum() + 1.0
        return F.l1_loss(pe, ge, reduction="sum") / num


class SoftSkeletonize(nn.Module):
    """Differentiable soft skeletonization from the clDice paper.
    Repeatedly erodes and keeps the residual (image - open(image))."""
    def __init__(self, num_iter=10):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel


class SoftclDiceLoss(nn.Module):
    """Topology-preserving clDice loss for tubular/filament-like objects."""
    def __init__(self, iter_=10, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true):
        skel_pred = self.skeletonize(y_pred)
        skel_true = self.skeletonize(y_true)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


class CombinedLoss(nn.Module):
    def __init__(self, cfg, pos_weight):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("pw", torch.tensor([float(pos_weight)]))
        self.dice = SoftDiceLoss()
        self.tversky = TverskyFocalLoss(cfg.tversky_alpha, cfg.tversky_beta, cfg.tversky_gamma)
        self.boundary = BoundaryLoss()
        self.cldice = SoftclDiceLoss(cfg.cldice_iter) if cfg.w_cldice > 0 else None

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw.to(logits.device))
        dice_l = self.dice(logits, target)
        tversky_l = self.tversky(logits, target)
        loss = (self.cfg.w_bce * bce
                + self.cfg.w_dice * dice_l
                + self.cfg.w_tversky * tversky_l)
        if self.cfg.w_boundary > 0:
            bl = self.boundary(logits, target)
            loss = loss + self.cfg.w_boundary * bl
        if self.cfg.w_cldice > 0 and self.cldice is not None:
            p = torch.sigmoid(logits)
            cl = self.cldice(p, target)
            # Guard against NaN/Inf from soft skeletonization under AMP
            if torch.isfinite(cl):
                loss = loss + self.cfg.w_cldice * cl
        return loss


def pos_weight_from_meta(cfg, meta):
    frac = max(float(meta.get("pos_frac", 0.002)), 1e-6)
    return float(min((1.0 - frac) / frac, cfg.pos_weight_cap))


# =============================================================================
# 7. METRICS  (mIoU_multiscale is the competition metric)
# =============================================================================
_CELLS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def instances_from_mask(binary, min_area=0):
    lab, n = ndimage.label(binary > 0)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() >= min_area:
            out.append(comp)
    return out


def _edges(mask):
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    return (m - er).astype(np.uint8)


def _shrink_max(m, cell):
    if cell <= 1:
        return m
    h, w = m.shape
    ph, pw = (-h) % cell, (-w) % cell
    if ph or pw:
        m = np.pad(m, ((0, ph), (0, pw)))
    H, W = m.shape
    return m.reshape(H // cell, cell, W // cell, cell).max(axis=(1, 3))


_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x / 1.x


def _pair_iou(g, p):
    inter = np.logical_and(g, p).sum()
    union = np.logical_or(g, p).sum()
    return inter / union if union > 0 else 0.0


def miou_multiscale_pair(gt_list, pred_list, cells=_CELLS, smooth=1.0, penalize=True,
                         use_edges=False):
    """Multi-scale IoU over (gt, pred) instance pairs.

    Returns (score, weight).  The per-pair score follows the EdgeAttNet / MAGFiLO
    multiscale formulation: masks are box-counted (max-pool) at cell sizes
    1..512, the overlap ratio is computed at each scale and integrated.

    ``use_edges=True`` keeps the old edge-only proxy (for ablation).  By default
    the full mask is used at every scale, exactly as described in the EdgeAttNet
    paper (Eq. 2):  r(o,\\tilde{o}, delta) = |s(o) \\cap s(\\tilde{o})| / |s(o)|.

    `penalize=True` additionally performs greedy one-to-one matching and scales the
    result by the F1 of matched instances, 2*n_matched / (n_gt + n_pred), so
    fragmenting one filament into many pieces or predicting spurious blobs cannot
    inflate the score.
    """
    n_gt, n_pred = len(gt_list), len(pred_list)
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1                      # correctly predicted an empty image
    if n_gt == 0 or n_pred == 0:
        return 0.0, max(n_gt, n_pred)

    gt = np.stack([np.asarray(g, bool) for g in gt_list])
    pr = np.stack([np.asarray(p, bool) for p in pred_list])
    inter = gt.reshape(n_gt, -1).astype(np.float32) @ pr.reshape(n_pred, -1).astype(np.float32).T
    if inter.max() <= 0:
        return (0.0, n_gt + n_pred) if penalize else (0.0, 0)

    if use_edges:
        ge = [_edges(g) for g in gt]
        pe = [_edges(p) for p in pr]
    else:
        ge = [g.astype(np.uint8) for g in gt]
        pe = [p.astype(np.uint8) for p in pr]

    cells = [c for c in cells if c <= min(gt.shape[1], gt.shape[2])] or [1]
    dx = 1.0 / (len(cells) - 1) if len(cells) > 1 else 1.0

    pairs = np.argwhere(inter > 0)
    score_mat = np.zeros((n_gt, n_pred), np.float64)
    for i, j in pairs:
        ratios = []
        for c in cells:
            a = _shrink_max(ge[i], c)
            b = _shrink_max(pe[j], c)
            a_sum = float(a.sum())
            if a_sum == 0:
                # gt object has shrunk to nothing at this scale; treat as neutral
                ratios.append(1.0 if float(b.sum()) == 0 else 0.0)
            else:
                ratios.append((float((a & b).sum()) + smooth) / (a_sum + smooth))
        score_mat[i, j] = float(_trapz(ratios, dx=dx)) if len(ratios) > 1 else float(ratios[0])

    if not penalize:
        vals = score_mat[inter > 0]
        return (float(vals.mean()), int(len(pairs))) if vals.size else (0.0, 0)

    # greedy one-to-one matching on the score matrix (prefer higher IoU)
    # For equally-good candidates, keep the multiscale score as the tie-breaker.
    matched, used_g, used_p = [], set(), set()
    order = np.argsort(score_mat, axis=None)[::-1]
    for flat in order:
        i, j = divmod(int(flat), n_pred)
        if score_mat[i, j] <= 0:
            break
        if i in used_g or j in used_p:
            continue
        # A pair with tiny overlap at full scale is often an accidental
        # neighbour; require a minimum full-scale IoU to be considered valid.
        if _pair_iou(gt[i], pr[j]) < 0.05:
            continue
        used_g.add(i); used_p.add(j); matched.append(score_mat[i, j])
    if not matched:
        return 0.0, n_gt + n_pred
    f1 = 2.0 * len(matched) / (n_gt + n_pred)
    return float(np.mean(matched) * f1), n_gt + n_pred


def semantic_iou(pred, gt):
    p, g = pred > 0, gt > 0
    u = np.logical_or(p, g).sum()
    return 1.0 if u == 0 else float(np.logical_and(p, g).sum()) / float(u)


def semantic_dice(pred, gt):
    p, g = pred > 0, gt > 0
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * float(np.logical_and(p, g).sum()) / float(s)


# =============================================================================
# 8. TRAIN
# =============================================================================
def train_one_epoch(cfg, net, loader, criterion, optimizer, scaler, ema, sched=None, ep=1):
    net.train()
    optimizer.zero_grad(set_to_none=True)
    running, seen = 0.0, 0
    amp = cfg.use_amp and DEVICE.type == "cuda"
    pbar = tqdm(loader, desc=f"train ep{ep}", leave=False)
    log_interval = max(1, len(loader) // 4)
    t_start = time.time()
    for step, (x, y, _) in enumerate(pbar):
        x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=amp):
            logits = net(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, y)
        scaler.scale(loss / cfg.accum).backward()
        if (step + 1) % cfg.accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Only step scheduler if scaler didn't skip (NaN grads)
            if sched is not None and scaler.get_scale() >= old_scale:
                sched.step()
            if ema is not None:
                ema.update(net)
        running += loss.item() * x.size(0)
        seen += x.size(0)
        pbar.set_postfix(loss=f"{running / max(seen, 1):.4f}")
        if (step + 1) % log_interval == 0 or (step + 1) == len(loader):
            vram = f"{torch.cuda.memory_reserved() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A"
            speed = (step + 1) / max(0.1, time.time() - t_start)
            log_info(f"  [Train Ep {ep:>2}] Step {step+1:>3}/{len(loader)} | Batch Loss: {loss.item():.4f} | Running Loss: {running/max(seen,1):.4f} | VRAM: {vram} | Speed: {speed:.1f} it/s")
    return running / max(seen, 1)


@torch.no_grad()
def evaluate(cfg, net, loader, criterion, full_metric=False, ep=None):
    """Per-epoch validation. Uses multi-threshold sweep for Dice/IoU so
    checkpoint selection is robust regardless of model calibration.
    The instance metric is computed on a downsampled grid (miou_eval_res)."""
    net.eval()
    amp = cfg.use_amp and DEVICE.type == "cuda"
    loss_sum, n = 0.0, 0
    er = cfg.img_size if full_metric else min(cfg.miou_eval_res, cfg.img_size)
    if ep is not None:
        log_info(f"  [Val Ep {ep:>2}] Evaluating {len(loader)} validation batches @ {er}x{er}...")
    # Collect all probs and GTs for multi-threshold sweep
    all_prob, all_gt = [], []
    log_interval = max(1, len(loader) // 2)
    for step, (x, y, _) in enumerate(tqdm(loader, desc="val", leave=False)):
        x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=amp):
            logits = net(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        prob = torch.sigmoid(logits.float())
        gt = y
        if er != prob.shape[-1]:
            prob = F.interpolate(prob, (er, er), mode="bilinear", align_corners=False)
            gt = F.max_pool2d(gt, kernel_size=gt.shape[-1] // er) if gt.shape[-1] % er == 0 else \
                F.interpolate(gt, (er, er), mode="nearest")
        all_prob.append(prob[:, 0].cpu().numpy())
        all_gt.append((gt[:, 0] > 0.5).cpu().numpy())
        if ep is not None and ((step + 1) % log_interval == 0 or (step + 1) == len(loader)):
            log_info(f"  [Val Ep {ep:>2}] Step {step+1:>2}/{len(loader)} | Running Loss: {loss_sum/max(n,1):.4f}")

    all_prob = np.concatenate(all_prob, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    val_loss = loss_sum / max(n, 1)

    # Multi-threshold sweep: pick the threshold that maximizes Dice
    val_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_dice, best_iou, best_thr = 0.0, 0.0, cfg.threshold
    for thr in val_thresholds:
        dices_t, ious_t = [], []
        for p, g in zip(all_prob, all_gt):
            dices_t.append(semantic_dice(p > thr, g))
            ious_t.append(semantic_iou(p > thr, g))
        mean_dice = float(np.mean(dices_t)) if dices_t else 0.0
        mean_iou = float(np.mean(ious_t)) if ious_t else 0.0
        if mean_dice > best_dice:
            best_dice, best_iou, best_thr = mean_dice, mean_iou, thr

    # Compute instance mIoU at the best threshold
    num, den = 0.0, 0
    scale = (er / cfg.img_size) ** 2
    for p, g in zip(all_prob, all_gt):
        pbin = p > best_thr
        s, k = miou_multiscale_pair(instances_from_mask(g),
                                    instances_from_mask(pbin, int(cfg.min_area * scale)))
        num += s * k
        den += k
    val_miou = (num / den) if den else 0.0

    if ep is not None:
        log_info(f"  [Val Ep {ep:>2}] Results -> Loss: {val_loss:.4f} | Dice: {best_dice:.4f} (thr={best_thr:.2f}) | IoU: {best_iou:.4f} | Multiscale mIoU: {val_miou:.4f}")
    return {"loss": val_loss,
            "iou": best_iou,
            "dice": best_dice,
            "miou": val_miou}


def train_fold(cfg, meta):
    # Validate fold index
    if cfg.fold < 0 or cfg.fold >= cfg.n_folds:
        raise ValueError(f"FOLD = {cfg.fold} is invalid. With n_folds={cfg.n_folds}, valid values are 0..{cfg.n_folds-1}")
    seed_everything(cfg.seed + cfg.fold)
    train_loader, val_loader = build_loaders(cfg, meta)
    net = build_model(cfg).to(DEVICE).to(memory_format=torch.channels_last)
    total_params = sum(p.numel() for p in net.parameters())
    pw = pos_weight_from_meta(cfg, meta)
    log_info(f"[Model] {cfg.arch.upper()} ({cfg.encoder}) initialized with {total_params:,} parameters.")
    log_info(f"[Loss] CombinedLoss initialized (pos_weight = {pw:.1f})")
    criterion = CombinedLoss(cfg, pw).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=cfg.use_amp and DEVICE.type == "cuda")
    ema = EMA(net, cfg.ema_decay) if cfg.use_ema else None

    # Count actual optimizer steps per epoch (accounting for accum and drop_last)
    steps_per_epoch = len(train_loader) // cfg.accum
    if len(train_loader) % cfg.accum != 0:
        steps_per_epoch += 1  # tail batch also triggers an optimizer step
    warm = max(1, cfg.warmup_epochs * steps_per_epoch)
    total = max(warm + 1, cfg.epochs * steps_per_epoch)

    def lr_at(it):
        if it < warm:
            return (it + 1) / warm
        p = (it - warm) / max(1, total - warm)
        floor = cfg.min_lr / cfg.lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    best, best_ep, stale = -1.0, 0, 0
    best_path = os.path.join(cfg.model_dir, f"best_fold_{cfg.fold}.pth")
    log = []
    log_info("=" * 80)
    log_info(f"STARTING TRAINING FOLD {cfg.fold}/{cfg.n_folds-1} | {cfg.epochs} Epochs | Batch: {cfg.batch_size} (Accum: {cfg.accum}) | Size: {cfg.img_size}px")
    log_info("=" * 80)
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]
        log_info(f"--- Epoch {ep:>2}/{cfg.epochs} (LR: {curr_lr:.2e}) ---")
        tl = train_one_epoch(cfg, net, train_loader, criterion, optimizer, scaler, ema, sched=sched, ep=ep)

        eval_net = net
        if ema is not None:
            eval_net = build_model_like(cfg, net)
            ema.copy_to(eval_net)
            eval_net = eval_net.to(DEVICE).to(memory_format=torch.channels_last)
        m = evaluate(cfg, eval_net, val_loader, criterion, ep=ep)
        if cfg.select_metric == "dice":
            score = m["dice"] + 0.5 * m["iou"]
        elif cfg.select_metric == "miou":
            score = m["miou"] + 1e-3 * m["iou"]
        else:
            score = m.get(cfg.select_metric, m["dice"])
        improved = score > best
        if improved:
            best, best_ep, stale = score, ep, 0
            torch.save({"model": (ema.state_dict() if ema else net.state_dict()),
                        "raw_model": net.state_dict(),
                        "cfg": {k: v for k, v in vars(CFG).items() if not k.startswith("_") and isinstance(v, (int, float, str, bool))},
                        "epoch": ep, "score": score, "metric": cfg.select_metric}, best_path)
        else:
            stale += 1
        if eval_net is not net:
            del eval_net
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        elapsed = time.time() - t0
        status_flag = " * [NEW BEST CHECKPOINT SAVED]" if improved else ""
        log_info(f"Summary Ep {ep:>2}/{cfg.epochs} | Train Loss: {tl:.4f} | Val Loss: {m['loss']:.4f} | Dice: {m['dice']:.4f} | IoU: {m['iou']:.4f} | mIoU: {m['miou']:.4f} | {elapsed:.1f}s{status_flag}")
        log.append(dict(epoch=ep, train_loss=tl, **m, best=best))
        pd.DataFrame(log).to_csv(os.path.join(cfg.model_dir, f"fold_{cfg.fold}_log.csv"), index=False)
        if stale >= cfg.patience:
            log_info(f"[Early Stop] No improvement for {stale} epochs (Best {cfg.select_metric} = {best:.4f} @ epoch {best_ep})")
            break
    log_info("=" * 80)
    log_info(f"[Train Complete] Best {cfg.select_metric} = {best:.4f} @ epoch {best_ep} -> {best_path}")
    log_info("=" * 80)
    return best_path


def build_model_like(cfg, ref=None):
    """Fresh model with identical architecture, no pretrained download."""
    if cfg.arch.lower() == "edgeattnet":
        return UNetEdgeTransformer(in_channels=cfg.in_channels)
    kwargs = dict(encoder_name=cfg.encoder, encoder_weights=None,
                  in_channels=cfg.in_channels, classes=1, activation=None)
    if cfg.decoder_attention.lower() != "none" and cfg.arch.lower() in ("unet", "unetplusplus"):
        kwargs["decoder_attention_type"] = cfg.decoder_attention
    factory = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus,
               "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus}
    try:
        return factory.get(cfg.arch.lower(), smp.Unet)(**kwargs)
    except Exception:
        kwargs.pop("decoder_attention_type", None)
        return factory.get(cfg.arch.lower(), smp.Unet)(**kwargs)


# =============================================================================
# 9. INFERENCE  (D4 TTA)
# =============================================================================
_D4 = [
    (lambda t: t,                                   lambda t: t),
    (lambda t: torch.flip(t, [-1]),                 lambda t: torch.flip(t, [-1])),
    (lambda t: torch.flip(t, [-2]),                 lambda t: torch.flip(t, [-2])),
    (lambda t: torch.flip(t, [-2, -1]),             lambda t: torch.flip(t, [-2, -1])),
    (lambda t: torch.rot90(t, 1, (-2, -1)),         lambda t: torch.rot90(t, -1, (-2, -1))),
    (lambda t: torch.rot90(t, 3, (-2, -1)),         lambda t: torch.rot90(t, -3, (-2, -1))),
    (lambda t: torch.transpose(t, -2, -1),          lambda t: torch.transpose(t, -2, -1)),
    (lambda t: torch.flip(torch.transpose(t, -2, -1), [-1]),
     lambda t: torch.transpose(torch.flip(t, [-1]), -2, -1)),
]


def load_for_inference(cfg, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise FileNotFoundError(f"unreadable image: {path}")
    # EdgeAttNet-style preprocessing before the model sees the image
    img, _ = preprocess_solar(img)
    shape = img.shape[:2]
    small = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    # replicate to in_channels BEFORE normalising so ImageNet per-channel
    # stats (which differ per channel) match build_transforms() exactly.
    x = np.repeat(x[:, :, None], cfg.in_channels, axis=2) if cfg.in_channels == 3 else x[:, :, None]
    if cfg.imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], np.float32)[:cfg.in_channels]
        std = np.array([0.229, 0.224, 0.225], np.float32)[:cfg.in_channels]
    else:
        mean = np.full(cfg.in_channels, 0.5, np.float32)
        std = np.full(cfg.in_channels, 0.5, np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).float(), shape


@torch.no_grad()
def predict_prob(cfg, net, path):
    """Return a float32 probability map at FULL resolution (2048).
    Supports D4 TTA and optional multi-scale TTA."""
    x, shape = load_for_inference(cfg, path)
    x = x.to(DEVICE).to(memory_format=torch.channels_last)
    amp = cfg.use_amp and DEVICE.type == "cuda"
    views = _D4 if cfg.tta_d4 else _D4[:1]
    h, w = x.shape[-2:]
    n_views = len(views) * len(cfg.tta_scales)
    acc = None
    weight = 0.0
    for fwd, inv in views:
        for s in cfg.tta_scales:
            if abs(s - 1.0) < 1e-3:
                x_s = x
            else:
                hs, ws = max(1, int(h * s)), max(1, int(w * s))
                x_s = F.interpolate(x, (hs, ws), mode="bilinear", align_corners=False)
            with torch.autocast(device_type=DEVICE.type, enabled=amp):
                out = net(fwd(x_s))
            p = inv(torch.sigmoid(out.float()))
            if p.shape[-2:] != (h, w):
                p = F.interpolate(p, (h, w), mode="bilinear", align_corners=False)
            # scale weight: avoid down-weighting smaller scales
            sw = 1.0 / max(0.5, s)
            acc = p * sw if acc is None else acc + p * sw
            weight += sw
    prob = (acc / weight)[0, 0].cpu().numpy().astype(np.float32)
    return cv2.resize(np.ascontiguousarray(prob), (shape[1], shape[0]),
                      interpolation=cv2.INTER_LINEAR)


# =============================================================================
# 10. POST-PROCESSING + RLE
# =============================================================================
def solar_disk_mask(gray):
    """Use the better Hough/connected-component disk estimate if available,
    otherwise fall back to the fast threshold-based mask."""
    info = solar_disk_hough(gray)
    if info is not None:
        return info["mask"]
    # fallback
    _, b = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    num, lab = cv2.connectedComponents(b)
    if num <= 1:
        return np.ones_like(gray, np.uint8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    disk = (lab == int(sizes.argmax())).astype(np.uint8)
    return cv2.morphologyEx(disk, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def crf_refine(cfg, prob, img_rgb):
    """Optional dense CRF on (H,W,3) image + (H,W) probability."""
    if not cfg.use_crf:
        return prob
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except Exception:
        return prob
    h, w = prob.shape
    n_labels = 2
    # build unary from probabilities: shape (2, H*W)
    softmax = np.stack([1 - prob, prob], axis=0).astype(np.float32)
    unary = unary_from_softmax(softmax, scale=None, clip=1e-5)
    d = dcrf.DenseCRF2D(w, h, n_labels)
    d.setUnaryEnergy(unary)
    # spatial gaussian
    d.addPairwiseGaussian(sxy=(cfg.crf_sxy, cfg.crf_sxy), compat=cfg.crf_compat,
                          kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # bilateral colour term (img_rgb is always uint8 0..255 from callers)
    img_u8 = np.ascontiguousarray(img_rgb.astype(np.uint8))
    d.addPairwiseBilateral(sxy=(cfg.crf_sxy * 8, cfg.crf_sxy * 8), srgb=(cfg.crf_srgb,) * 3,
                           rgbim=img_u8, compat=cfg.crf_compat,
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(cfg.crf_iter)
    Q = np.array(Q).reshape((n_labels, h, w))
    return Q[1]  # foreground probability


def clean_binary(cfg, m):
    if cfg.morph_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_k, cfg.morph_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if cfg.fill_holes > 0 and m.any():
        filled = ndimage.binary_fill_holes(m > 0)
        holes = filled & (m == 0)
        lab, n = ndimage.label(holes)
        if n:
            for i in range(1, n + 1):
                sel = lab == i
                if sel.sum() <= cfg.fill_holes:
                    m[sel] = 1
    return m


def split_instances(cfg, m):
    out = []
    lab, n = ndimage.label(m > 0)
    for i in range(1, n + 1):
        comp = (lab == i).astype(np.uint8)
        area = int(comp.sum())
        if area < cfg.min_area or area > cfg.max_area:
            continue
        if not cfg.watershed_split:
            out.append(comp)
            continue
        try:
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            dist = ndimage.distance_transform_edt(comp)
            peaks = peak_local_max(dist, min_distance=25, labels=comp.astype(bool),
                                   exclude_border=False)
            if len(peaks) > 1:
                seeds = np.zeros(comp.shape, bool)
                seeds[tuple(peaks.T)] = True
                markers, _ = ndimage.label(seeds)
                ws = watershed(-dist, markers, mask=comp.astype(bool))
                added = False
                for v in range(1, int(ws.max()) + 1):
                    sub = (ws == v).astype(np.uint8)
                    if cfg.min_area <= int(sub.sum()) <= cfg.max_area:
                        out.append(sub)
                        added = True
                if added:
                    continue
        except Exception:
            pass
        out.append(comp)
    return out


def prob_to_instances(cfg, prob, disk=None, threshold=None, min_area=None, img_rgb=None):
    thr = cfg.threshold if threshold is None else threshold
    ma = cfg.min_area if min_area is None else min_area
    if float(prob.max()) < cfg.no_filament_prob:
        return []                      # genuinely empty image -> emit no rows
    # Optional dense CRF before thresholding; img_rgb must be (H,W,3)
    if cfg.use_crf and img_rgb is not None:
        prob = crf_refine(cfg, prob, img_rgb)
    m = (prob > thr).astype(np.uint8)
    if disk is not None:
        m *= disk
    m = clean_binary(cfg, m)
    old = cfg.min_area
    cfg.min_area = ma
    try:
        return split_instances(cfg, m)
    finally:
        cfg.min_area = old


def rle_encode(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    c = rle["counts"]
    return c.decode("utf-8") if isinstance(c, bytes) else c


# =============================================================================
# 11. OOF THRESHOLD / MIN-AREA SEARCH
# =============================================================================
def write_oof_probs(cfg, net, meta):
    """Full-resolution probabilities for the validation fold."""
    folds = np.asarray(meta["folds"])
    va = np.where(folds == cfg.fold)[0]
    d = os.path.join(cfg.prob_dir, f"oof_fold_{cfg.fold}")
    os.makedirs(d, exist_ok=True)
    net.eval()
    log_info(f"[OOF] Writing full-resolution out-of-fold probability maps for {len(va)} validation images -> {d}")
    log_interval = max(1, len(va) // 5)
    for idx, i in enumerate(tqdm(va, desc="oof probs")):
        fn = meta["files"][i]
        p = predict_prob(cfg, net, os.path.join(cfg.train_images, fn))
        np.save(os.path.join(d, os.path.splitext(fn)[0] + ".npy"), p.astype(np.float16))
        if (idx + 1) % log_interval == 0 or (idx + 1) == len(va):
            log_info(f"  [OOF] Generated {idx+1:>3}/{len(va)} probability maps ({(idx+1)/len(va)*100:.1f}%)")
    log_info(f"[OOF] Saved {len(va)} probability maps to {d}")
    return d


def search_threshold(cfg, prob_dir, meta, eval_res=512):
    """Grid-search (threshold, min_area) to maximise the real mIoU_multiscale."""
    file_to_anns = group_annotations(cfg.train_json)
    folds = np.asarray(meta["folds"])
    va = [meta["files"][i] for i in np.where(folds == cfg.fold)[0]]
    scale = eval_res / float(cfg.full_res)
    area_scale = scale * scale

    log_info(f"[Search] Loading {len(va)} OOF maps for threshold & min_area optimization (eval @ {eval_res}px)...")
    cache = []
    for fn in tqdm(va, desc="load oof"):
        p = os.path.join(prob_dir, os.path.splitext(fn)[0] + ".npy")
        if not os.path.exists(p):
            continue
        prob = np.load(p).astype(np.float32)
        gt_full = polygons_to_mask(file_to_anns.get(fn, []), cfg.full_res, cfg.full_res)
        prob_s = cv2.resize(prob, (eval_res, eval_res), interpolation=cv2.INTER_LINEAR)
        gt_s = _downsample_mask_max(gt_full, eval_res)
        cache.append((prob_s, instances_from_mask(gt_s)))
    if not cache:
        log_info("[Search] No OOF maps found; keeping default parameters.")
        return cfg.threshold, cfg.min_area, 0.0

    total_combos = len(cfg.search_thresholds) * len(cfg.search_min_areas)
    log_info(f"[Search] Evaluating {total_combos} combinations ({len(cfg.search_thresholds)} thresholds x {len(cfg.search_min_areas)} min_areas)...")
    best = (cfg.threshold, cfg.min_area, -1.0)
    for idx_thr, thr in enumerate(cfg.search_thresholds):
        base = [(prob > thr).astype(np.uint8) for prob, _ in cache]
        for ma in cfg.search_min_areas:
            num, den = 0.0, 0
            ma_s = max(1, int(ma * area_scale))
            for (prob, gt_inst), m in zip(cache, base):
                pred_inst = instances_from_mask(m, ma_s)
                s, k = miou_multiscale_pair(gt_inst, pred_inst)
                num += s * k
                den += k
            score = (num / den) if den else 0.0
            if score > best[2]:
                best = (float(thr), int(ma), float(score))
                log_info(f"  [Search New Best] threshold = {best[0]:.2f} | min_area = {best[1]:>4}px | Multiscale mIoU = {best[2]:.4f}")
        if (idx_thr + 1) % 2 == 0 or (idx_thr + 1) == len(cfg.search_thresholds):
            log_info(f"  [Search Progress] Completed {idx_thr+1}/{len(cfg.search_thresholds)} threshold sweeps | Current Best: thr={best[0]:.2f}, min_area={best[1]}px, mIoU={best[2]:.4f}")
    if best[2] <= 0.0:
        log_info(f"[Search] Model produced 0 score on validation set; keeping defaults thr={cfg.threshold}, min_area={cfg.min_area}.")
        return cfg.threshold, cfg.min_area, 0.0
    log_info(f"[Search Complete] Best: threshold={best[0]:.2f} | min_area={best[1]}px | Multiscale mIoU={best[2]:.4f}")
    json.dump({"threshold": best[0], "min_area": best[1], "miou": best[2]},
              open(os.path.join(cfg.model_dir, f"postproc_fold_{cfg.fold}.json"), "w"))
    return best


# =============================================================================
# 12. SUBMISSION
# =============================================================================
def build_submission(cfg, net, threshold=None, min_area=None, out_csv=None,
                     save_prob_dir=None):
    out_csv = out_csv or os.path.join(cfg.sub_dir, "submission.csv")
    files = sorted(p for p in glob.glob(os.path.join(cfg.test_images, "*"))
                   if os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"no test images under {cfg.test_images}")
    if save_prob_dir:
        os.makedirs(save_prob_dir, exist_ok=True)
    rows, empty = [], 0
    net.eval()
    log_info(f"[Submit] Running test inference on {len(files)} images (threshold={threshold or cfg.threshold:.2f}, min_area={min_area or cfg.min_area}, D4 TTA={cfg.tta_d4})...")
    log_interval = max(1, len(files) // 5)
    for idx, path in enumerate(tqdm(files, desc="test")):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            prob = predict_prob(cfg, net, path)
        except Exception as e:
            log_info(f"[warn] Failed predicting {stem}: {e}")
            empty += 1
            continue
        if save_prob_dir:
            np.save(os.path.join(save_prob_dir, stem + ".npy"), prob.astype(np.float16))
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        disk = solar_disk_mask(gray) if gray is not None else None
        # CRF needs an RGB-ish image; repeat the preprocessed gray or raw gray
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray is not None else None
        inst = prob_to_instances(cfg, prob, disk, threshold, min_area, img_rgb=img_rgb)
        if not inst:
            empty += 1
        for i, m in enumerate(inst):
            rows.append({"filament_id": f"{stem}_{i:04d}", "segmentation_rle": rle_encode(m)})
        if (idx + 1) % log_interval == 0 or (idx + 1) == len(files):
            log_info(f"  [Submit] Processed {idx+1:>3}/{len(files)} test images ({(idx+1)/len(files)*100:.1f}%) | Filaments found: {len(rows):,}")
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    log_info(f"[Submit Complete] Generated {len(df):,} filaments across {len(files)} test images ({empty} empty images) -> {out_csv}")
    if len(df) == 0:
        log_info("[Submit WARNING] Submission is EMPTY! Check threshold and min_area settings.")
    return df


def ensemble_probs(prob_dirs, out_dir):
    """Average .npy probability maps across folds/models, then re-threshold."""
    os.makedirs(out_dir, exist_ok=True)
    names = sorted({os.path.basename(p) for d in prob_dirs
                    for p in glob.glob(os.path.join(d, "*.npy"))})
    log_info(f"[Ensemble] Averaging probability maps from {len(prob_dirs)} fold directories for {len(names)} test images...")
    for name in tqdm(names, desc="ensemble"):
        acc, k = None, 0
        for d in prob_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                a = np.load(p).astype(np.float32)
                acc = a if acc is None else acc + a
                k += 1
        if k:
            np.save(os.path.join(out_dir, name), (acc / k).astype(np.float16))
    log_info(f"[Ensemble Complete] Wrote {len(names)} averaged maps -> {out_dir}")
    return out_dir


def submission_from_prob_dir(cfg, prob_dir, threshold=None, min_area=None, out_csv=None):
    out_csv = out_csv or os.path.join(cfg.sub_dir, "submission_ensemble.csv")
    rows = []
    npy_files = sorted(glob.glob(os.path.join(prob_dir, "*.npy")))
    log_info(f"[Ensemble Submit] Generating instances from {len(npy_files)} probability maps (thr={threshold}, min_area={min_area})...")
    for p in tqdm(npy_files, desc="ensemble submit"):
        stem = os.path.splitext(os.path.basename(p))[0]
        prob = np.load(p).astype(np.float32)
        img = next((c for ext in (".jpg", ".jpeg", ".png")
                    if os.path.exists(c := os.path.join(cfg.test_images, stem + ext))), None)
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if img else None
        disk = solar_disk_mask(gray) if gray is not None else None
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray is not None else None
        for i, m in enumerate(prob_to_instances(cfg, prob, disk, threshold, min_area, img_rgb=img_rgb)):
            rows.append({"filament_id": f"{stem}_{i:04d}", "segmentation_rle": rle_encode(m)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    log_info(f"[Ensemble Submit Complete] Generated {len(df):,} filaments -> {out_csv}")
    return df


# =============================================================================
# 13. MAIN
# =============================================================================
def run(cfg=CFG, do_train=True, do_search=True, do_submit=True):
    # NOTE: only paths are re-derived here. The preset is NOT re-applied, so any
    # manual override you made after apply_preset() is preserved.
    cfg = resolve(cfg)
    seed_everything(cfg.seed)
    log_info(f"[Init] Device: {DEVICE} | Preset: {cfg.preset} | Architecture: {cfg.arch} ({cfg.encoder}) | Size: {cfg.img_size}px | Fold: {cfg.fold}/{cfg.n_folds-1}")
    meta = build_cache(cfg)

    best_path = os.path.join(cfg.model_dir, f"best_fold_{cfg.fold}.pth")
    if do_train:
        best_path = train_fold(cfg, meta)

    net = build_model_like(cfg)
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    net = net.to(DEVICE).to(memory_format=torch.channels_last).eval()
    log_info(f"[Checkpoint Loaded] {best_path} (epoch {ck.get('epoch')}, {ck.get('metric')}={ck.get('score', float('nan')):.4f})")

    thr, ma = cfg.threshold, cfg.min_area
    if do_search:
        d = write_oof_probs(cfg, net, meta)
        thr, ma, _ = search_threshold(cfg, d, meta)

    df = None
    if do_submit:
        df = build_submission(cfg, net, threshold=thr, min_area=ma,
                              save_prob_dir=os.path.join(cfg.prob_dir, f"test_fold_{cfg.fold}")
                              if cfg.save_probs else None)
    return dict(cfg=cfg, meta=meta, model=net, best_path=best_path,
                threshold=thr, min_area=ma, submission=df)


if __name__ == "__main__":
    run(apply_preset(CFG))
