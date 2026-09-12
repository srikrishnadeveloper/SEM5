"""
v8_1 Stage 3 — Instance-Segmentation Cascade Inference Engine.

Master: ChatGPT | Builder: Antigravity
Target: Kaggle Solar Filament Segmentation Challenge 2026

Frozen Rules:
- Pipeline: YOLO proposals -> adaptive pad -> crop refiner (4-flip TTA) -> residual (IoU<0.30) -> greedy score non-overlap
- Residual OFF by default (--residual 0).
- Adaptive crop formula: min(512, max(256, int(1.2 * max(w, h)))).
- 4-flip TTA on crops only.
- Empty disk: 0 rows (strictly adheres to host contract; every row represents a real filament).
- Output: submissions/v8_1_cascade.csv.
- Statistics audit: n_images, n_rows, rows/image (mean, p50, p90), n_empty, elapsed seconds.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.pq import encode_mask, decode_rle
from v8_1.config import CascadeConfig, CropRefinerConfig, YOLOConfig, MAGFILO_DIR, SUBMISSIONS_DIR
from v8_1.geometry import compute_adaptive_crop_side, square_bounds_from_bbox

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_infer_config(args):
    """Print the frozen Inference Cascade specification for review."""
    print("=" * 75)
    print("V8_1 STAGE 3: CASCADE INFERENCE (CONFIGURATION)")
    print("=" * 75)
    print(f"  Stage 1 YOLO Proposer: {YOLOConfig.WEIGHTS_PATH} (conf={args.conf})")
    print(f"  Stage 2 Crop Refiner:  {CropRefinerConfig.CKPT_PATH} (4-flip TTA={args.tta})")
    print(f"  Residual Discovery:    DISABLED (Pure YOLO -> Crop Refiner cascade)")
    print(f"  Solar Disk Mask:       Radius = {CascadeConfig.SOLAR_DISK_R_FRAC} * 1024 (~952 px)")
    print(f"  Min Area Threshold:    {getattr(args, 'min_area', 100)} px")
    print(f"  YOLO Mask Fallback:    {getattr(args, 'yolo_fallback', 0)}")
    print(f"  Overlap Strategy:      {getattr(args, 'overlap_mode', 'trim')}")
    print(f"  Empty Disk Policy:     0 rows emitted (per official host PQ metric contract)")
    print(f"  Output Submission:     {args.out}")
    print(f"  CUDA Available:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'})")
    print("=" * 75)


def get_solar_disk_mask(h: int = 2048, w: int = 2048, r_frac: float = 0.93) -> np.ndarray:
    """Generate circular solar disk mask centered at (w/2, h/2)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(r_frac * min(h, w) / 2.0)
    cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask


def preprocess_3ch(gray: np.ndarray) -> np.ndarray:
    """Build [Raw, CLAHE, Unsharp] 3-channel input."""
    ch_raw = gray.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5)
    ch_unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return np.stack([ch_raw, ch_clahe, ch_unsharp], axis=-1)


def refine_crop_tta(
    refiner_model,
    img_3ch: np.ndarray,
    bbox: list[int],
    device: torch.device,
    target_size: int = 384,
    use_tta: bool = True,
    threshold: float = 0.50,
) -> np.ndarray:
    """Refine a single candidate using Crop U-Net with 4-flip TTA."""
    h, w = img_3ch.shape[:2]

    # Strictly square adaptive crop window (identical contract to training)
    side = compute_adaptive_crop_side(
        bbox,
        crop_min=CropRefinerConfig.CROP_MIN,
        crop_max=CropRefinerConfig.CROP_MAX,
        crop_padding=CropRefinerConfig.CROP_PADDING,
    )
    c_x1, c_y1, c_x2, c_y2 = square_bounds_from_bbox(bbox, h, w, side)

    crop_img = img_3ch[c_y1:c_y2, c_x1:c_x2]
    orig_ch, orig_cw = crop_img.shape[:2]

    # Square resize to target_size
    resized = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    inp = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    inp = inp.to(device)

    refiner_model.eval()
    with torch.no_grad():
        if use_tta:
            # 4-flip TTA: [original, fliplr, flipud, fliplr+flipud]
            p0 = torch.sigmoid(refiner_model(inp))
            p1 = torch.sigmoid(refiner_model(torch.flip(inp, dims=[3])))
            p1 = torch.flip(p1, dims=[3])
            p2 = torch.sigmoid(refiner_model(torch.flip(inp, dims=[2])))
            p2 = torch.flip(p2, dims=[2])
            p3 = torch.sigmoid(refiner_model(torch.flip(inp, dims=[2, 3])))
            p3 = torch.flip(p3, dims=[2, 3])
            prob_map = (p0 + p1 + p2 + p3) / 4.0
        else:
            prob_map = torch.sigmoid(refiner_model(inp))

    prob_np = prob_map.squeeze().cpu().numpy()
    crop_prob = cv2.resize(prob_np, (orig_cw, orig_ch), interpolation=cv2.INTER_LINEAR)
    bin_crop = (crop_prob > threshold).astype(np.uint8)

    # Paste back into full 2048x2048 canvas
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[c_y1:c_y2, c_x1:c_x2] = bin_crop
    return full_mask


def sanitize_instances_zero_overlap(
    instances: list[dict],
    min_area: int = 400,
) -> list[dict]:
    """Strict host-contract sanitizer: guarantees ZERO shared pixels per disk.
    
    1. Sort accepted instances by confidence descending, then area descending.
    2. Greedy pixel carve: mask[occupied > 0] = 0.
    3. Drop if area < min_area after carve.
    4. Update occupied buffer.
    5. Per disk ASSERT: sum(mask_i & mask_j) == 0 for all i != j.
       Fail the notebook immediately if any overlap remains.
    """
    if not instances:
        return []

    # 1. Sort by confidence desc, then area desc
    sorted_inst = sorted(
        instances,
        key=lambda x: (float(x["confidence"]), int(x["mask"].sum())),
        reverse=True,
    )

    occupied = None
    sanitized = []

    for inst in sorted_inst:
        m = inst["mask"].copy()
        if occupied is not None:
            m[occupied > 0] = 0

        area = int(m.sum())
        if area < min_area:
            continue

        if occupied is None:
            occupied = (m > 0).astype(np.uint8)
        else:
            occupied = np.maximum(occupied, (m > 0).astype(np.uint8))

        sanitized.append({
            "confidence": float(inst["confidence"]),
            "mask": m,
        })

    # 5. Strict Pairwise & Union Disjoint Assertions
    n = len(sanitized)
    for i in range(n):
        for j in range(i + 1, n):
            overlap = int(np.logical_and(sanitized[i]["mask"], sanitized[j]["mask"]).sum())
            assert overlap == 0, f"FATAL: Overlap detected between instance {i} and {j}: {overlap} pixels!"

    if n > 0 and occupied is not None:
        disjoint_sum = sum(int(inst["mask"].sum()) for inst in sanitized)
        union_sum = int(occupied.sum())
        assert disjoint_sum == union_sum, f"FATAL: Disjoint sum mismatch: disjoint={disjoint_sum} vs union={union_sum}!"

    return sanitized


def arbitrate_instances(
    candidates: list[dict],
    disk_mask: np.ndarray = None,
    min_area: int = 100,
    overlap_mode: str = "trim",
) -> list[dict]:
    """Arbitrate candidate instances sorted by confidence descending.
    
    Args:
        candidates: list of dicts with 'confidence' and 'mask'.
        disk_mask: circular mask (h, w) for solar disk clipping.
        min_area: minimum pixel area threshold.
        overlap_mode: 'trim' (greedy pixel carve) or 'allow' (instance NMS at IoU>0.5).
    """
    sorted_cands = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    occupied = None
    accepted_instances = []

    for c in sorted_cands:
        mask = c["mask"].copy()
        if disk_mask is not None:
            mask[disk_mask == 0] = 0

        if mask.sum() < min_area:
            continue

        if overlap_mode == "trim":
            if occupied is not None:
                mask[occupied > 0] = 0
            if mask.sum() < min_area:
                continue
            if occupied is None:
                occupied = mask.copy()
            else:
                occupied = np.maximum(occupied, mask)
            accepted_instances.append({
                "confidence": c["confidence"],
                "mask": mask,
            })
        elif overlap_mode == "allow":
            # Instance-level NMS at IoU > 0.5 without carving valid pixels
            suppress = False
            for accepted in accepted_instances:
                acc_mask = accepted["mask"]
                intersection = np.logical_and(mask, acc_mask).sum()
                if intersection > 0:
                    union = np.logical_or(mask, acc_mask).sum()
                    iou = float(intersection) / float(union) if union > 0 else 0.0
                    if iou > 0.5:
                        suppress = True
                        break
            if not suppress:
                accepted_instances.append({
                    "confidence": c["confidence"],
                    "mask": mask,
                })
        else:
            raise ValueError(f"Unknown overlap_mode: {overlap_mode}")

    return accepted_instances


# Legacy alias for backward compatibility
greedy_non_overlap_arbitration = arbitrate_instances


def run_cascade_inference(args):
    """Execute end-to-end inference cascade across test images."""
    test_img_dir = MAGFILO_DIR / "test" / "test_images"
    test_jpegs = sorted(list(test_img_dir.glob("*.jpeg")) + list(test_img_dir.glob("*.jpg")))
    if not test_jpegs:
        raise FileNotFoundError(f"No test images found in {test_img_dir}")

    # Dual-T4 Model Parallelism: YOLO on cuda:0, Crop U-Net on cuda:1 (if available)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_yolo = 0 if n_gpus > 0 else "cpu"
    refiner_gpu_idx = 1 if n_gpus >= 2 else (0 if n_gpus == 1 else "cpu")
    device_refiner = torch.device(f"cuda:{refiner_gpu_idx}" if n_gpus > 0 else "cpu")
    disk_mask = get_solar_disk_mask(2048, 2048, r_frac=CascadeConfig.SOLAR_DISK_R_FRAC)

    print(f"  Target test disks to process: {len(test_jpegs)}")
    print(f"  Model-Parallel Placement: YOLO -> cuda:{device_yolo} | Refiner -> {device_refiner}")

    # Load Stage 1 YOLO model
    from ultralytics import YOLO
    yolo_ckpt = YOLOConfig.resolve_weights(args.weights if hasattr(args, "weights") else None)
    if not yolo_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 YOLO weights not found at {yolo_ckpt}")
    print(f"  Stage 1 YOLO Proposer loaded from: {yolo_ckpt}")
    yolo_model = YOLO(str(yolo_ckpt))

    # Load Stage 2 Crop U-Net refiner
    import segmentation_models_pytorch as smp
    refiner_ckpt = CropRefinerConfig.resolve_ckpt(args.refiner_weights if hasattr(args, "refiner_weights") else None)
    if not refiner_ckpt.exists():
        raise FileNotFoundError(f"Stage 2 Crop Refiner checkpoint not found at {refiner_ckpt}")
    print(f"  Stage 2 Crop Refiner loaded from: {refiner_ckpt}")

    refiner_model = smp.Unet(
        encoder_name=CropRefinerConfig.ENCODER,
        in_channels=CropRefinerConfig.IN_CHANNELS,
        classes=CropRefinerConfig.CLASSES,
    ).to(device_refiner)
    refiner_model.load_state_dict(torch.load(refiner_ckpt, map_location=device_refiner))
    refiner_model.eval()

    all_rows = []
    rows_per_disk = []
    empty_disk_count = 0
    t0 = time.perf_counter()

    min_area = getattr(args, "min_area", 400)
    yolo_fallback = getattr(args, "yolo_fallback", 0)
    overlap_mode = getattr(args, "overlap_mode", "trim")

    if overlap_mode != "trim":
        raise ValueError(
            f"FATAL: overlap_mode='{overlap_mode}' is FORBIDDEN for submission generation! "
            f"Host strictly requires ZERO shared pixels between predicted masks. Must use overlap_mode='trim'."
        )

    for idx, p in enumerate(test_jpegs):
        stem = p.stem
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"FATAL: Unable to decode test image at {p}! Official test image is corrupt or missing.")

        h, w = raw.shape[:2]
        img_3ch = preprocess_3ch(raw)

        # 1. YOLO Proposals (Input is 2048 JPEG; Ultralytics maps boxes back to 2048 source frame)
        yolo_preds = yolo_model.predict(
            source=str(p),
            imgsz=YOLOConfig.IMGSZ,
            conf=args.conf,
            device=device_yolo,
            verbose=False,
        )

        candidates = []
        if yolo_preds and yolo_preds[0].boxes is not None and len(yolo_preds[0].boxes) > 0:
            boxes = yolo_preds[0].boxes.xyxy.cpu().numpy()
            confs = yolo_preds[0].boxes.conf.cpu().numpy()
            yolo_masks_raw = yolo_preds[0].masks.data.cpu().numpy() if (yolo_preds and yolo_preds[0].masks is not None) else None

            # One-line debug check on first image to guarantee boxes are in 2048 space
            if idx == 0 and len(boxes) > 0:
                min_c = float(boxes.min())
                max_c = float(boxes.max())
                print(f"  [DEBUG Image 0] YOLO raw xyxy range: min={min_c:.1f}, max={max_c:.1f} (assert <= {max(h, w)})")
                assert max_c <= max(h, w) + 50, f"FATAL: YOLO box coordinates exceed image bounds! Got max={max_c}"

            for b_idx, (box, conf) in enumerate(zip(boxes, confs)):
                # Clip directly to native 2048 coordinate frame (DO NOT multiply by scale!)
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(w, int(box[2]))
                y2 = min(h, int(box[3]))
                bbox_2048 = [x1, y1, x2, y2]

                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue

                # Refine with Crop U-Net on device_refiner
                refined_mask = refine_crop_tta(
                    refiner_model=refiner_model,
                    img_3ch=img_3ch,
                    bbox=bbox_2048,
                    device=device_refiner,
                    target_size=CropRefinerConfig.TARGET_SIZE,
                    use_tta=args.tta,
                )

                # YOLO mask fallback: if refiner mask is below min_area, use YOLO-seg mask if valid
                if yolo_fallback == 1 and refined_mask.sum() < min_area:
                    if yolo_masks_raw is not None and b_idx < len(yolo_masks_raw):
                        yolo_m = cv2.resize(yolo_masks_raw[b_idx].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        if yolo_m.sum() >= min_area:
                            refined_mask = yolo_m

                if refined_mask.sum() > 0:
                    candidates.append({
                        "confidence": float(conf),
                        "mask": refined_mask,
                    })

        # 2. Instance arbitration (trim)
        final_instances = arbitrate_instances(
            candidates,
            disk_mask=disk_mask,
            min_area=min_area,
            overlap_mode="trim",
        )

        # Mandatory Host Sanitizer: zero shared pixels guaranteed & asserted
        final_instances = sanitize_instances_zero_overlap(
            final_instances,
            min_area=min_area,
        )

        # 3. Pycocotools RLE encoding: ONE ROW PER ACTUAL DETECTED FILAMENT
        if len(final_instances) == 0:
            empty_disk_count += 1
            rows_per_disk.append(0)
            # Emit ZERO rows for images with zero detected filaments (per official host PQ metric contract)
        else:
            for k, inst in enumerate(final_instances):
                mask = inst["mask"]
                assert mask.sum() > 0, f"FATAL: Found zero-area predicted instance on {stem}_{k + 1}"
                rle_str = encode_mask(mask)
                all_rows.append({"filament_id": f"{stem}_{k + 1}", "segmentation_rle": rle_str})
            rows_per_disk.append(len(final_instances))

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_jpegs):
            elapsed = time.perf_counter() - t0
            print(f"  [{idx + 1:03d}/{len(test_jpegs):03d}] Processed {stem} | Total rows: {len(all_rows)} ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0

    # Safety assertion: model must produce positive predictions
    assert len(all_rows) > 0, "FATAL: Submission is completely empty! No filaments predicted."

    # Write submission CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Statistics breakdown
    r_arr = np.array(rows_per_disk)
    covered_stems = sum(1 for r in rows_per_disk if r > 0)
    empty_stems = sum(1 for r in rows_per_disk if r == 0)
    mean_r = float(np.mean(r_arr))
    p50_r = float(np.median(r_arr))
    p90_r = float(np.percentile(r_arr, 90))

    print("\n" + "=" * 75)
    print("SUBMISSION AUDIT & STATISTICS")
    print("=" * 75)
    print(f"  Submission CSV:           {out_path}")
    print(f"  Total Test Images:        {len(test_jpegs)}")
    print(f"  Covered Stems (>0 preds): {covered_stems} ({covered_stems / len(test_jpegs) * 100:.1f}%)")
    print(f"  Empty Stems (0 preds):    {empty_stems} ({empty_stems / len(test_jpegs) * 100:.1f}%)")
    print(f"  Total Filament Rows:      {len(all_rows)}")
    print(f"  Rows per Image:           Mean={mean_r:.2f} | p50={p50_r:.1f} | p90={p90_r:.1f}")
    print(f"  Total Elapsed Time:       {total_time:.2f} seconds ({len(test_jpegs) / total_time:.2f} img/s)")
    print("=" * 75)

    # Memory cleanup & VRAM logging
    del yolo_model, refiner_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for dev_idx in range(torch.cuda.device_count()):
            res_mb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
            alloc_mb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
            print(f"  [GPU {dev_idx}] Memory Reserved: {res_mb:.1f} MB | Allocated: {alloc_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Run v8_1 Instance-Segmentation Cascade Inference")
    parser.add_argument("--dry-run", action="store_true", help="Print frozen config and exit 0 without running")
    parser.add_argument("--residual", type=int, default=0, choices=[0, 1], help="Enable residual discovery branch (default: 0)")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO proposal confidence threshold (default: 0.20)")
    parser.add_argument("--min-area", type=int, default=400, help="Minimum filament area in pixels (default: 400)")
    parser.add_argument("--yolo-fallback", type=int, default=0, choices=[0, 1], help="Fallback to YOLO-seg mask if refiner output < min_area (default: 0)")
    parser.add_argument("--overlap-mode", type=str, default="trim", choices=["trim", "allow"], help="Overlap arbitration: only 'trim' permitted for submissions (default: trim)")
    parser.add_argument("--tta", action="store_true", default=True, help="Use 4-flip TTA on crops")
    parser.add_argument("--weights", type=str, default=None, help="Explicit path to YOLO weights (best.pt)")
    parser.add_argument("--refiner-weights", type=str, default=None, help="Explicit path to Crop Refiner weights (crop_refiner_r34.pth)")
    parser.add_argument("--out", type=str, default=str(CascadeConfig.SUBMISSION_PATH), help="Output submission CSV path")
    args = parser.parse_args()

    if args.dry_run:
        print_infer_config(args)
        print("Dry run completed successfully. Exiting 0 on CPU.\n")
        sys.exit(0)

    if args.overlap_mode != "trim":
        print(f"[FATAL ERROR] overlap_mode='{args.overlap_mode}' is FORBIDDEN for submission generation!")
        print("Host rejected submissions containing overlapping masks. Only overlap_mode='trim' is permitted.")
        sys.exit(1)

    if args.residual == 1:
        print("[NOTICE] Residual discovery branch is intentionally DISABLED for Baseline 1 (pure YOLO -> Crop Refiner).")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available! Cascade inference requires a GPU.")
        print("Run with --dry-run to inspect configuration on CPU.")
        sys.exit(1)

    print_infer_config(args)
    run_cascade_inference(args)


if __name__ == "__main__":
    main()
