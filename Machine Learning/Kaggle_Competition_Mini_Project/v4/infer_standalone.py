"""
Standalone Inference Engine — Solar Filament Segmentation Challenge 2026
Loads any trained .pth checkpoint and generates submission.csv in <90 seconds.
"""

import os
import sys
import glob
import time
from typing import List
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import pycocotools.mask as mask_utils
import segmentation_models_pytorch as smp


def log_info(msg: str):
    print(f"{time.strftime('[%H:%M:%S]')} [INFO] {msg}", flush=True)


def create_solar_features(img_raw: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_raw)
    blurred = cv2.GaussianBlur(img_raw, (0, 0), sigmaX=3.0)
    unsharp = cv2.addWeighted(img_raw, 1.5, blurred, -0.5, 0)
    return np.stack([img_raw, img_clahe, unsharp], axis=-1)


def generate_solar_limb_mask(h: int = 2048, w: int = 2048, radius_ratio: float = 0.93) -> np.ndarray:
    cy, cx = h / 2.0, w / 2.0
    r = min(h, w) / 2.0 * radius_ratio
    y, x = np.ogrid[:h, :w]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).astype(np.uint8)


def encode_coco_rle(binary_mask: np.ndarray) -> str:
    if binary_mask.sum() == 0:
        return "PPP2"
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    encoded = mask_utils.encode(fortran_mask)
    if isinstance(encoded["counts"], bytes):
        return encoded["counts"].decode("utf-8")
    return encoded["counts"]


def split_filament_instances_watershed(binary_mask: np.ndarray, min_area: int = 200, min_distance: int = 15) -> List[np.ndarray]:
    if binary_mask.sum() == 0:
        return []
    dist = distance_transform_edt(binary_mask)
    coords = peak_local_max(dist, min_distance=min_distance, labels=binary_mask)
    if len(coords) <= 1:
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        instances = []
        for lbl in range(1, n_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                instances.append((labels == lbl).astype(np.uint8))
        return instances

    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, pt in enumerate(coords):
        markers[pt[0], pt[1]] = i + 1

    labels = watershed(-dist, markers, mask=binary_mask)
    instances = []
    for lbl in range(1, labels.max() + 1):
        inst_mask = (labels == lbl).astype(np.uint8)
        if inst_mask.sum() >= min_area:
            instances.append(inst_mask)

    if not instances:
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        for lbl in range(1, n_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                instances.append((labels == lbl).astype(np.uint8))

    return instances


def build_model_from_checkpoint(ckpt_path: str, device: str = "cuda"):
    log_info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "UnetPlusPlus")
    encoder = ckpt.get("encoder", "efficientnet-b4")
    attention = ckpt.get("attention", "scse")

    if arch.lower() == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_attention_type=attention if attention else None
        )
    elif arch.lower() == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    else:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

    raw_sd = ckpt["model_state_dict"]
    clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    log_info(f"Loaded {arch} ({encoder}) successfully! (Trained Val Dice: {ckpt.get('val_dice', 0.0):.4f})")
    return model


@torch.no_grad()
def run_fast_inference(
    ckpt_paths: List[str],
    test_img_dir: str,
    output_csv: str = "submission.csv",
    img_size: int = 1024,
    threshold: float = 0.45,
    min_area: int = 200,
    use_tta: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    models = [build_model_from_checkpoint(p, device) for p in ckpt_paths]
    test_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpeg")) + glob.glob(os.path.join(test_img_dir, "*.jpg")))
    log_info(f"Found {len(test_files)} test images in {test_img_dir}")

    solar_mask = generate_solar_limb_mask(2048, 2048, radius_ratio=0.93)
    records = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    t0 = time.time()
    for fpath in tqdm(test_files, desc="Inference"):
        fn = os.path.basename(fpath)
        stem = os.path.splitext(fn)[0]

        img_raw = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        h, w = img_raw.shape[:2]

        feats = create_solar_features(img_raw)
        feats_resized = cv2.resize(feats, (img_size, img_size), interpolation=cv2.INTER_AREA)
        norm_img = (feats_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        ensemble_prob = np.zeros((img_size, img_size), dtype=np.float32)

        for model in models:
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                if use_tta:
                    p1 = torch.sigmoid(model(tensor))
                    p2 = torch.sigmoid(model(torch.flip(tensor, dims=[2])))
                    p2 = torch.flip(p2, dims=[2])
                    p3 = torch.sigmoid(model(torch.flip(tensor, dims=[3])))
                    p3 = torch.flip(p3, dims=[3])
                    p4 = torch.sigmoid(model(torch.flip(tensor, dims=[2, 3])))
                    p4 = torch.flip(p4, dims=[2, 3])
                    prob = (p1 + p2 + p3 + p4) / 4.0
                else:
                    prob = torch.sigmoid(model(tensor))

            ensemble_prob += prob.squeeze().cpu().numpy() / len(models)

        prob_2048 = cv2.resize(ensemble_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_2048 = prob_2048 * solar_mask

        bin_mask = (prob_2048 > threshold).astype(np.uint8)
        instances = split_filament_instances_watershed(bin_mask, min_area=min_area)

        if not instances:
            records.append({
                "filament_id": f"{stem}_1",
                "segmentation_rle": "PPP2"
            })
        else:
            for inst_idx, inst_mask in enumerate(instances, 1):
                rle = encode_coco_rle(inst_mask)
                records.append({
                    "filament_id": f"{stem}_{inst_idx}",
                    "segmentation_rle": rle
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    log_info(f"[SUCCESS] Generated {len(df)} predictions across {len(test_files)} images -> {output_csv} [{time.time() - t0:.1f}s]")
    return output_csv
