"""Dual-GPU Inference Engine for Cascade 2.0 (Crop & Zoom Paradigm).

Hardened against all historical failure modes:
1. Anchored Stage 1 precision: conf=0.25 on Native 2048 YOLOv8l-seg (eliminates the results (7) FP spike).
2. Non-truncating geometry: adaptive crop bounds up to 2048px (eliminates V8.1 512px clipping).
3. 3-channel prior conditioning: guides refiner on specific instance, resolving dense clusters.
4. Morphological polish: 3x3 circular closing to eliminate granular chromospheric noise.
5. Solar limb mask: removes edge flare outside r=0.93*R.
6. Greedy zero-overlap sanitizer: guarantees strictly 0 shared pixels per disk.
7. Host V6 compliant RLE: Fortran 3D shape (H, W, 1), 0 rows on empty disks.
8. Anti-leak memory management: gc.collect() and torch.cuda.empty_cache() checkpoints.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import csv
import sys
import time
import gc
import cv2
import numpy as np
import torch
import pycocotools.mask as mask_utils
from ultralytics import YOLO

from cascade.geometry import compute_adaptive_crop_side, square_bounds_from_bbox, splice_crop_prob_to_canvas
from cascade.dataset import build_3channel_input
from cascade.model_refiner import build_crop_refiner


def encode_mask_rle(mask_hw: np.ndarray) -> str:
    """Encode binary mask (H, W) to COCO RLE string via 3D Fortran array."""
    h, w = mask_hw.shape[:2]
    mask_3d = np.asfortranarray(mask_hw.astype(np.uint8)).reshape((h, w, 1))
    rle = mask_utils.encode(mask_3d)[0]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts


def get_solar_limb_mask(h: int = 2048, w: int = 2048, r_frac: float = 0.93) -> np.ndarray:
    """Generate circular solar disk mask centered at (w/2, h/2)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(r_frac * min(h, w) / 2.0)
    cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask


def refine_crops_batched(
    refiner_model: torch.nn.Module,
    img_3ch: np.ndarray,
    candidate_boxes: List[List[int]],
    coarse_masks: List[np.ndarray],
    device: torch.device,
    target_size: int = 384,
    use_tta: bool = True,
    threshold: float = 0.50,
) -> List[np.ndarray]:
    """Refine all candidate crops for a disk through the patch refiner."""
    if not candidate_boxes:
        return []

    H, W = img_3ch.shape[:2]
    crop_tensors = []
    crop_meta = []
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for bbox, c_mask in zip(candidate_boxes, coarse_masks):
        side = compute_adaptive_crop_side(bbox, crop_min=256, crop_max=2048, crop_padding=1.25)
        x1, y1, x2, y2 = square_bounds_from_bbox(bbox, H, W, side)

        # Slice 3-channel input
        crop_img = img_3ch[y1:y2, x1:x2].copy()

        # Update Channel 2 with instance coarse prior (or bounding box if empty)
        crop_prior = c_mask[y1:y2, x1:x2]
        if crop_prior.sum() > 0:
            crop_img[:, :, 2] = (crop_prior > 0).astype(np.uint8) * 255
        else:
            # Fallback: fill bounding box region inside crop
            rx1 = max(0, bbox[0] - x1)
            ry1 = max(0, bbox[1] - y1)
            rx2 = min(x2 - x1, bbox[2] - x1)
            ry2 = min(y2 - y1, bbox[3] - y1)
            crop_img[ry1:ry2, rx1:rx2, 2] = 255

        # Resize to network target size
        resized = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        t_inp = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        crop_tensors.append(t_inp)
        crop_meta.append((x1, y1, x2, y2))

    batch_inp = torch.stack(crop_tensors).to(device)

    refiner_model.eval()
    with torch.no_grad():
        if use_tta:
            # 4-flip TTA: orig, fliplr, flipud, both
            p0 = torch.sigmoid(refiner_model(batch_inp))
            p1 = torch.sigmoid(refiner_model(torch.flip(batch_inp, dims=[3])))
            p1 = torch.flip(p1, dims=[3])
            p2 = torch.sigmoid(refiner_model(torch.flip(batch_inp, dims=[2])))
            p2 = torch.flip(p2, dims=[2])
            p3 = torch.sigmoid(refiner_model(torch.flip(batch_inp, dims=[2, 3])))
            p3 = torch.flip(p3, dims=[2, 3])
            batch_prob = (p0 + p1 + p2 + p3) / 4.0
        else:
            batch_prob = torch.sigmoid(refiner_model(batch_inp))

    batch_prob_np = batch_prob.squeeze(1).cpu().numpy()
    if batch_prob_np.ndim == 2:
        batch_prob_np = np.expand_dims(batch_prob_np, 0)

    # Cleanup GPU memory immediately
    del batch_inp, batch_prob

    refined_masks = []
    for i in range(len(candidate_boxes)):
        prob_2d = batch_prob_np[i]
        bounds = crop_meta[i]
        full_prob = splice_crop_prob_to_canvas(prob_2d, bounds, (H, W))
        bin_mask = (full_prob > threshold).astype(np.uint8)

        # Morphological polish (closes small 1px granular holes in dark fibrils)
        if bin_mask.sum() > 0:
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, morph_kernel)

        refined_masks.append(bin_mask)

    return refined_masks


def sanitize_zero_overlap(
    masks: List[np.ndarray],
    confs: List[float],
    min_area: int = 200,
) -> Tuple[List[np.ndarray], List[float]]:
    """Greedy pixel-carve sanitizer guaranteeing 0 shared pixels on the disk."""
    if not masks:
        return [], []

    indices = np.argsort(confs)[::-1]
    H, W = masks[0].shape[:2]
    occupied = np.zeros((H, W), dtype=bool)

    sanitized_masks = []
    sanitized_confs = []

    for idx in indices:
        m = masks[idx].astype(bool)
        carved = m & (~occupied)
        area = int(carved.sum())
        if area >= min_area:
            occupied |= carved
            sanitized_masks.append(carved.astype(np.uint8))
            sanitized_confs.append(float(confs[idx]))

    return sanitized_masks, sanitized_confs


def run_cascade_inference(
    test_images_dir: Path,
    yolo_weights: Path,
    refiner_weights: Optional[Path],
    output_csv: Path,
    conf_threshold: float = 0.25,
    crop_threshold: float = 0.50,
    min_area: int = 200,
    target_size: int = 384,
    arch: str = "unetplusplus",
    encoder: str = "tu-efficientnet_b2",
):
    print("=" * 75)
    print("🚀 CASCADE 2.0: BULLETPROOF DUAL-GPU INFERENCE PIPELINE")
    print("=" * 75)
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        dev_yolo = torch.device("cuda:0")
        dev_refiner = torch.device("cuda:1")
        print(f"  Dual-GPU Mode: Stage 1 on {dev_yolo} | Stage 2 on {dev_refiner}")
    elif n_gpus == 1:
        dev_yolo = torch.device("cuda:0")
        dev_refiner = torch.device("cuda:0")
        print(f"  Single-GPU Mode: Both stages on {dev_yolo}")
    else:
        dev_yolo = torch.device("cpu")
        dev_refiner = torch.device("cpu")
        print("  CPU Mode: Execution on CPU")

    print(f"  Stage 1 YOLO Model:     {yolo_weights} (conf={conf_threshold})")
    print(f"  Stage 2 Refiner Model:  {refiner_weights or 'Disabled (YOLO Pass-through)'}")
    print(f"  Refiner Crop Threshold: {crop_threshold}")
    print(f"  Min Area Threshold:     {min_area} px")
    print(f"  Output CSV:             {output_csv}")
    print("=" * 75)

    # 1. Load Stage 1 YOLO
    print("⏳ Loading Stage 1 YOLOv8l-seg champion model...")
    yolo_model = YOLO(str(yolo_weights))

    # 2. Load Stage 2 Refiner
    refiner_model = None
    if refiner_weights and Path(refiner_weights).exists():
        print(f"⏳ Loading Stage 2 Refiner model from: {refiner_weights}...")
        refiner_model = build_crop_refiner(arch=arch, encoder_name=encoder, in_channels=3, encoder_weights=None)
        ckpt = torch.load(str(refiner_weights), map_location=dev_refiner)
        state = ckpt.get("model_state", ckpt)
        refiner_model.load_state_dict(state)
        refiner_model.to(dev_refiner)
        refiner_model.eval()
        print("✅ Stage 2 Refiner loaded successfully.")
    else:
        print("ℹ️ Stage 2 Refiner weights not provided; operating in Stage 1 Direct mode.")

    limb_mask = get_solar_limb_mask(2048, 2048, r_frac=0.93)
    img_files = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpeg")))
    print(f"🔍 Found {len(img_files)} test images in {test_images_dir}")

    submission_rows = []
    global_instance_id = 0
    empty_disks = 0
    t_start = time.time()

    for img_idx, img_p in enumerate(img_files):
        gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        H, W = gray.shape[:2]

        # Stage 1: YOLO Proposer
        results = yolo_model.predict(
            source=gray,
            imgsz=2048,
            conf=conf_threshold,
            device=dev_yolo,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            empty_disks += 1
            continue

        raw_boxes = boxes.xyxy.cpu().numpy().astype(int)
        raw_confs = boxes.conf.cpu().numpy()

        # Extract YOLO coarse masks
        coarse_masks = []
        if results.masks is not None:
            raw_masks = results.masks.data.cpu().numpy()
            for m in raw_masks:
                if m.shape != (H, W):
                    m_full = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    coarse_masks.append((m_full > 0.5).astype(np.uint8))
                else:
                    coarse_masks.append((m > 0.5).astype(np.uint8))
        else:
            for b in raw_boxes:
                m_box = np.zeros((H, W), dtype=np.uint8)
                m_box[b[1]:b[3], b[0]:b[2]] = 1
                coarse_masks.append(m_box)

        # Stage 2: Crop Refiner
        if refiner_model is not None:
            img_3ch = build_3channel_input(gray, None)
            candidate_masks = refine_crops_batched(
                refiner_model=refiner_model,
                img_3ch=img_3ch,
                candidate_boxes=raw_boxes.tolist(),
                coarse_masks=coarse_masks,
                device=dev_refiner,
                target_size=target_size,
                use_tta=True,
                threshold=crop_threshold,
            )
        else:
            candidate_masks = coarse_masks

        # Apply Solar Limb Mask
        clean_masks = [m * limb_mask for m in candidate_masks]

        # Greedy Zero Overlap Sanitizer
        final_masks, final_confs = sanitize_zero_overlap(clean_masks, raw_confs.tolist(), min_area=min_area)

        if len(final_masks) == 0:
            empty_disks += 1
            continue

        # Encode to RLE with Host V6 format: {stem}_{instance_index}
        stem = img_p.stem
        for inst_idx, m in enumerate(final_masks, start=1):
            rle_str = encode_mask_rle(m)
            submission_rows.append([f"{stem}_{inst_idx}", rle_str])

        if (img_idx + 1) % 30 == 0 or (img_idx + 1) == len(img_files):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [{img_idx + 1:03d}/{len(img_files):03d}] Processed | Cumulative Filaments: {len(submission_rows)} | Elapsed: {time.time() - t_start:.1f}s")

    # Write submission CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(submission_rows)

    total_time = time.time() - t_start
    print("=" * 75)
    print(f"🏁 Cascade Inference Complete in {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"  Total Images:        {len(img_files)}")
    print(f"  Total Filaments:     {len(submission_rows)}")
    print(f"  Active Disks:        {len(img_files) - empty_disks}")
    print(f"  Empty Disks:         {empty_disks}")
    print(f"  Mean Filaments/Disk: {len(submission_rows) / max(1, len(img_files) - empty_disks):.2f}")
    print(f"  Saved to:            {output_csv}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cascade 2.0 Inference")
    parser.add_argument("--test-dir", type=str, default="data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    parser.add_argument("--yolo-weights", type=str, default=r"C:\Users\srik2\Desktop\Filament_Colab_Run\best.pt")
    parser.add_argument("--refiner-weights", type=str, default="models/best_crop_refiner.pth")
    parser.add_argument("--out", type=str, default="submissions/submission_cascade_p1.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--crop-thresh", type=float, default=0.50)
    parser.add_argument("--min-area", type=int, default=200)
    args = parser.parse_args()

    run_cascade_inference(
        test_images_dir=Path(args.test_dir),
        yolo_weights=Path(args.yolo_weights),
        refiner_weights=Path(args.refiner_weights) if Path(args.refiner_weights).exists() else None,
        output_csv=Path(args.out),
        conf_threshold=args.conf,
        crop_threshold=args.crop_thresh,
        min_area=args.min_area,
    )
