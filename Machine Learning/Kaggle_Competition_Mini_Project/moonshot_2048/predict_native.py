"""
moonshot_2048/predict_native.py — Native-2048 YOLO Inference Engine & Zero-Overlap Sanitizer.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Frozen Submission & Prediction Truths:
1. Native Resolution: Predict directly on 2048x2048 images (or fallback 1792/1536).
2. Host Zero-Overlap Rule: Strictly zero shared pixels between any two masks on the same disk.
   - Enforce greedy pixel carve: sort instances descending by confidence, then area.
   - mask[occupied > 0] = 0; drop if post-carve area < min_area.
   - Per-disk assertion: sum(mask_i & mask_j) == 0 for all i != j.
3. Zero-Prediction Rule:
   - Disks with zero predicted filaments emit ZERO rows in submission.csv.
   - Never emit dummy zero masks (PPP2, PPP8, etc.).
4. Encoding:
   - Valid pycocotools COCO Fortran RLE.
   - Output format: exactly two columns: filament_id,segmentation_rle.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from metrics.pq import encode_mask, decode_rle
from moonshot_2048.config import MoonshotConfig, MAGFILO_DIR, SUBMISSIONS_DIR


def apply_solar_limb_mask(mask: np.ndarray, r_frac: float = MoonshotConfig.SOLAR_DISK_R_FRAC) -> np.ndarray:
    """Zero out any predicted pixels falling outside the solar chromosphere limb."""
    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    max_r = int((min(w, h) / 2.0) * r_frac)
    
    # Fast distance check: zero out outside circle
    y_indices, x_indices = np.ogrid[:h, :w]
    dist_from_center_sq = (x_indices - cx)**2 + (y_indices - cy)**2
    mask[dist_from_center_sq > max_r**2] = 0
    return mask


def sanitize_instances_zero_overlap(
    masks: List[np.ndarray],
    confidences: List[float],
    min_area: int = MoonshotConfig.MIN_AREA,
    r_frac: float = MoonshotConfig.SOLAR_DISK_R_FRAC,
) -> Tuple[List[np.ndarray], List[float]]:
    """Strict Greedy Pixel-Carve Zero-Overlap Sanitizer.
    
    Order: Confidence descending, then area descending.
    Each mask carves away pixels already claimed by higher-priority masks.
    Post-carve masks with area < min_area are discarded.
    Finally, strictly asserts that no two masks share any pixels.
    """
    if not masks:
        return [], []

    # Pair and sort
    items = []
    for m, c in zip(masks, confidences):
        m_clipped = apply_solar_limb_mask(m.copy(), r_frac=r_frac)
        area = int(m_clipped.sum())
        if area >= min_area:
            items.append({"mask": m_clipped, "conf": float(c), "area": area})

    if not items:
        return [], []

    # Sort descending by confidence, then area
    items.sort(key=lambda x: (x["conf"], x["area"]), reverse=True)

    h, w = items[0]["mask"].shape[:2]
    occupied = np.zeros((h, w), dtype=np.uint8)
    sanitized_masks = []
    sanitized_confs = []

    for it in items:
        m = it["mask"]
        # Carve out already occupied pixels
        m[occupied > 0] = 0
        carved_area = int(m.sum())

        if carved_area >= min_area:
            occupied |= m
            sanitized_masks.append(m)
            sanitized_confs.append(it["conf"])

    # Strict Zero-Overlap Assertion per Host Rule
    n_kept = len(sanitized_masks)
    for i in range(n_kept):
        for j in range(i + 1, n_kept):
            shared = int(np.logical_and(sanitized_masks[i], sanitized_masks[j]).sum())
            if shared > 0:
                raise AssertionError(f"FATAL: Overlap sanitizer failed! {shared} shared pixels between mask {i} and {j}.")

    return sanitized_masks, sanitized_confs


def run_native_inference_on_disk(
    model,
    img_path: Path,
    imgsz: int = MoonshotConfig.IMGSZ,
    conf: float = MoonshotConfig.CONF,
    nms_iou: float = MoonshotConfig.NMS_IOU,
    min_area: int = MoonshotConfig.MIN_AREA,
    device: str = "0",
) -> Tuple[List[np.ndarray], List[float]]:
    """Run native YOLO-seg inference on a single 2048x2048 image and sanitize."""
    preds = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        iou=nms_iou,
        max_det=MoonshotConfig.MAX_DET,
        device=device,
        verbose=False,
    )

    pred_masks = []
    pred_confs = []

    if preds and preds[0].masks is not None:
        raw_masks = preds[0].masks.data.cpu().numpy()
        raw_confs = preds[0].boxes.conf.cpu().numpy()

        for rm, c in zip(raw_masks, raw_confs):
            # Upscale mask to native 2048x2048 if predicted at smaller imgsz
            if rm.shape[:2] != (2048, 2048):
                m_2048 = cv2.resize(rm.astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST)
            else:
                m_2048 = rm.astype(np.uint8)

            if m_2048.sum() > 0:
                pred_masks.append(m_2048)
                pred_confs.append(float(c))

    # Apply greedy zero-overlap sanitizer
    clean_masks, clean_confs = sanitize_instances_zero_overlap(
        pred_masks, pred_confs, min_area=min_area
    )
    return clean_masks, clean_confs


def generate_submission(
    model,
    test_dir: Path,
    out_csv: Path,
    imgsz: int = MoonshotConfig.IMGSZ,
    conf: float = MoonshotConfig.CONF,
    nms_iou: float = MoonshotConfig.NMS_IOU,
    min_area: int = MoonshotConfig.MIN_AREA,
    device: str = "0",
    assert_180: bool = True,
) -> pd.DataFrame:
    """Generate official Kaggle submission CSV for all test images."""
    import hashlib
    import json

    test_images = sorted(list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg")))
    print("=" * 75)
    print("MOONSHOT 2048: NATIVE TEST INFERENCE & SUBMISSION BUILD (DIRECTIVE 18)")
    print("=" * 75)
    print(f"  Test Images Found:   {len(test_images)}")
    print(f"  Inference imgsz:     {imgsz}")
    print(f"  Confidence:          {conf}")
    print(f"  NMS IoU:             {nms_iou}")
    print(f"  Min Area:            {min_area} px")
    print(f"  Output CSV:          {out_csv}")
    print("=" * 75)

    if assert_180:
        assert len(test_images) == 180, f"FATAL: Expected exactly 180 test images, found {len(test_images)} in {test_dir}!"
        print("  [DISCOVERY VERIFIED] Exactly 180 test images confirmed.")

    rows = []
    zero_pred_disks = 0
    total_filaments = 0
    processed_stems = []
    zero_pred_stems = []

    for idx, img_p in enumerate(test_images):
        stem = img_p.stem
        processed_stems.append(stem)
        masks, confs = run_native_inference_on_disk(
            model=model,
            img_path=img_p,
            imgsz=imgsz,
            conf=conf,
            nms_iou=nms_iou,
            min_area=min_area,
            device=device,
        )

        if not masks:
            zero_pred_disks += 1
            zero_pred_stems.append(stem)
            # Zero-prediction rule: emit 0 rows for this disk
            continue

        for inst_idx, mask in enumerate(masks):
            rle_str = encode_mask(mask)
            filament_id = f"{stem}_{inst_idx}"
            rows.append({
                "filament_id": filament_id,
                "segmentation_rle": rle_str,
            })
            total_filaments += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(test_images):
            print(f"  [{idx + 1}/{len(test_images)}] processed | Filaments so far: {total_filaments}")

    df_sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(out_csv, index=False)

    sha256 = hashlib.sha256(out_csv.read_bytes()).hexdigest()

    # Record inference manifest
    manifest = {
        "status": "SUCCESS",
        "total_test_images": len(test_images),
        "total_filaments_predicted": total_filaments,
        "represented_disks": len(test_images) - zero_pred_disks,
        "zero_detection_disks": zero_pred_disks,
        "test_stems": processed_stems,
        "zero_pred_stems": zero_pred_stems,
        "submission_path": str(out_csv.resolve()),
        "submission_sha256": sha256,
        "inference_imgsz": imgsz,
        "conf": conf,
        "nms_iou": nms_iou,
        "min_area": min_area,
    }
    manifest_path = out_csv.parent / "inference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 75)
    print("SUBMISSION SUMMARY & MANIFEST:")
    print(f"  Total Processed Images: {len(test_images)}")
    print(f"  Total Submitted Rows:   {len(df_sub)}")
    print(f"  Zero-Detection Disks:   {zero_pred_disks} (emitted 0 rows per host rule)")
    print(f"  Wrote Submission To:    {out_csv} (SHA256: {sha256})")
    print(f"  Wrote Manifest To:      {manifest_path}")
    print("=" * 75)
    return df_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native-2048 YOLO Inference")
    parser.add_argument("--weights", type=str, default=str(MoonshotConfig.V8L_WEIGHTS_PATH))
    parser.add_argument("--test-dir", type=str, default=str(MAGFILO_DIR / "test" / "test_images"))
    parser.add_argument("--out", type=str, default=str(MoonshotConfig.SUBMISSION_PATH))
    parser.add_argument("--imgsz", type=int, default=MoonshotConfig.IMGSZ)
    parser.add_argument("--conf", type=float, default=MoonshotConfig.CONF)
    parser.add_argument("--iou", type=float, default=MoonshotConfig.NMS_IOU)
    parser.add_argument("--min-area", type=int, default=MoonshotConfig.MIN_AREA)
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] Native-2048 inference configured successfully. Exiting 0.")
        sys.exit(0)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    generate_submission(
        model=model,
        test_dir=Path(args.test_dir),
        out_csv=Path(args.out),
        imgsz=args.imgsz,
        conf=args.conf,
        nms_iou=args.iou,
        min_area=args.min_area,
        device=args.device,
    )
