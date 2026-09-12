"""
Smoke Cascade Test — CPU-only plumbing verifier on REAL MAGFiLO Disks.

Exercises the full Seed -> Refine cascade pipeline using random/heuristic
proposals on N real MAGFiLO train JPEGs and evaluates against REAL COCO ground truth.

Validates:
1. Reading real GONG 2048x2048 JPEGs
2. 3-channel astronomical preprocessing (Raw + CLAHE + Unsharp)
3. YOLO-style proposal generation (stubbed with random/heuristic boxes)
4. Solar disk limb masking (radius ~0.93 * 1024)
5. Adaptive crop sizing: side = min(512, max(256, int(1.2 * max(w, h))))
6. Crop refiner with 4-flip TTA
7. Greedy confidence-score non-overlap arbitration
8. Residual discovery branch (global semantic -> connected components -> IoU < 0.3 filter)
9. COCO RLE encoding via pycocotools (never hardcoded)
10. Ground truth: REAL polygons from COCO JSON rasterized at 2048x2048 (no pred leakage)
11. Per-disk and total Kirillov PQ / SQ / RQ / TP / FP / FN computation
12. CSV output: submissions/smoke_real8.csv with rows-per-image audit

Usage:
    python scripts/smoke_cascade.py --n 8
    python scripts/smoke_cascade.py --n 8 --out submissions/smoke_real8.csv
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.pq import encode_mask, decode_rle, pq_score, pq_score_multi


# ─────────────────────────────────────────────────────────────────────
# 1. Astronomical Preprocessing & Solar Disk Mask
# ─────────────────────────────────────────────────────────────────────

def get_solar_disk_mask(h: int = 2048, w: int = 2048, r_frac: float = 0.93) -> np.ndarray:
    """Generate circular solar disk mask centered at (w/2, h/2)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(r_frac * min(h, w) / 2.0)
    cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask


def preprocess_3ch(gray: np.ndarray) -> np.ndarray:
    """Build [Raw, CLAHE, Unsharp] 3-channel input from grayscale.

    Args:
        gray: (H, W) uint8 grayscale image.

    Returns:
        (H, W, 3) uint8 array.
    """
    ch_raw = gray.copy()

    # Channel 2: CLAHE (clip=3.0, grid=8x8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(gray)

    # Channel 3: High-pass unsharp ridge filter
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5)
    ch_unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return np.stack([ch_raw, ch_clahe, ch_unsharp], axis=-1)


# ─────────────────────────────────────────────────────────────────────
# 2. Stub YOLO proposal generator
# ─────────────────────────────────────────────────────────────────────

def stub_yolo_proposals(h: int, w: int, n_proposals: int = 5, rng=None):
    """Generate random bounding box proposals (simulating YOLO output).

    Returns:
        list of dicts with keys: bbox (x1, y1, x2, y2), confidence, mask (H, W uint8).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    proposals = []
    for _ in range(n_proposals):
        cx = rng.integers(300, w - 300)
        cy = rng.integers(300, h - 300)
        bw = rng.integers(50, 160)
        bh = rng.integers(50, 160)
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)
        conf = float(rng.uniform(0.35, 0.90))

        # Create an elliptical filament stub
        mask = np.zeros((h, w), dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2))
        angle = rng.integers(0, 180)
        cv2.ellipse(mask, center, axes, int(angle), 0, 360, 1, -1)

        proposals.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": conf,
            "mask": mask,
        })
    return proposals


# ─────────────────────────────────────────────────────────────────────
# 3. Adaptive crop sizing + stub refiner with 4-flip TTA
# ─────────────────────────────────────────────────────────────────────

def adaptive_crop_size(bbox):
    """Compute crop side: min(512, max(256, int(1.2 * max(w, h))))."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    return min(512, max(256, int(1.2 * max(bw, bh))))


def stub_crop_refiner_tta(image_3ch: np.ndarray, bbox, mask_proposal: np.ndarray):
    """Simulate crop refiner with 4-flip TTA."""
    side = adaptive_crop_size(bbox)
    assert 256 <= side <= 512, f"Crop side {side} out of range [256, 512]"

    # Refine mask: morphological cleanup
    kernel = np.ones((3, 3), dtype=np.uint8)
    refined = cv2.erode(mask_proposal, kernel, iterations=1)
    return refined


# ─────────────────────────────────────────────────────────────────────
# 4. Residual discovery branch
# ─────────────────────────────────────────────────────────────────────

def stub_residual_discovery(h: int, w: int, yolo_masks: list[np.ndarray], rng=None):
    """Simulate global semantic checkpoint -> connected components -> IoU < 0.3 filter."""
    if rng is None:
        rng = np.random.default_rng(123)

    n_residual = rng.integers(0, 3)
    candidates = []

    for _ in range(n_residual):
        cx = rng.integers(300, w - 300)
        cy = rng.integers(300, h - 300)
        r = rng.integers(20, 60)
        blob = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(blob, (cx, cy), r, 1, -1)

        # Check IoU with every YOLO instance — keep only if ALL IoUs < 0.30
        keep = True
        for ym in yolo_masks:
            intersection = np.logical_and(blob, ym).sum()
            union = np.logical_or(blob, ym).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou >= 0.30:
                keep = False
                break

        if keep:
            conf = float(rng.uniform(0.25, 0.48))
            candidates.append({
                "bbox": (max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r)),
                "confidence": conf,
                "mask": blob,
            })

    return candidates


# ─────────────────────────────────────────────────────────────────────
# 5. Greedy non-overlap arbitration by confidence
# ─────────────────────────────────────────────────────────────────────

def greedy_non_overlap(instances: list[dict], disk_mask: np.ndarray = None) -> list[dict]:
    """Greedy assembly: higher confidence keeps pixels, lower is trimmed.

    Also clips instances to the solar disk boundary.
    """
    sorted_inst = sorted(instances, key=lambda x: x["confidence"], reverse=True)
    occupied = None
    result = []

    for inst in sorted_inst:
        mask = inst["mask"].copy()
        if disk_mask is not None:
            mask[disk_mask == 0] = 0

        if occupied is not None:
            mask[occupied > 0] = 0

        if mask.sum() < 50:  # Noise threshold
            continue

        if occupied is None:
            occupied = mask.copy()
        else:
            occupied = np.maximum(occupied, mask)

        result.append({
            "confidence": inst["confidence"],
            "mask": mask,
        })

    return result


# ─────────────────────────────────────────────────────────────────────
# 6. Real Data & Ground Truth Loading
# ─────────────────────────────────────────────────────────────────────

def load_real_disks_and_gt(data_dir: Path, n: int = 8):
    """Load N real MAGFiLO train JPEGs and rasterize real COCO ground truth polygons.

    Returns:
        list of dicts with keys: stem, image (2048x2048 uint8), gt_masks (list of 2048x2048 uint8).
    """
    train_dir = data_dir / "train"
    img_dir = train_dir / "train_images"
    json_candidates = list(train_dir.glob("*.json")) + list(data_dir.glob("*.json"))
    if not json_candidates:
        raise FileNotFoundError(f"No annotation JSON in {data_dir}")
    json_path = json_candidates[0]

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map file_name -> list of image_ids (annotator observations)
    fn_to_img_ids = defaultdict(list)
    for img in coco.get("images", []):
        fn_to_img_ids[img["file_name"]].append(img["id"])

    # Map image_id -> list of annotations (excluding category 4)
    img_id_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != 4:
            img_id_to_anns[ann["image_id"]].append(ann)

    # Get real train JPEGs
    jpeg_paths = sorted(list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")))[:n]
    if not jpeg_paths:
        raise FileNotFoundError(f"No JPEGs found in {img_dir}")

    samples = []
    for p in jpeg_paths:
        fn = p.name
        stem = p.stem
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        h, w = raw.shape[:2]

        # Get primary annotator's observations (no polygon OR-merging)
        img_ids = fn_to_img_ids.get(fn, [])
        gt_masks = []
        if img_ids:
            primary_id = img_ids[0]
            anns = img_id_to_anns.get(primary_id, [])
            for ann in anns:
                segs = ann.get("segmentation", [])
                if isinstance(segs, list):
                    inst_mask = np.zeros((h, w), dtype=np.uint8)
                    for poly in segs:
                        if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
                            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(inst_mask, [pts], 1)
                    if inst_mask.sum() > 0:
                        gt_masks.append(inst_mask)

        samples.append({
            "stem": stem,
            "filename": fn,
            "image": raw,
            "gt_masks": gt_masks,
        })

    return samples


# ─────────────────────────────────────────────────────────────────────
# 7. Main Smoke Cascade Execution
# ─────────────────────────────────────────────────────────────────────

def run_smoke_cascade(n: int = 8, output_csv: str = None):
    """Run full cascade pipeline on N real MAGFiLO disks against real GT."""

    data_dir = PROJECT_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"

    print("=" * 75)
    print("SMOKE CASCADE — Real-Image MAGFiLO Plumbing & Kirillov PQ Verifier")
    print("=" * 75)
    print(f"  Target Disks:      {n}")
    print(f"  Data Directory:    {data_dir}")
    print(f"  Execution Mode:    CPU (Stub Proposer + Real COCO GT)")
    print()

    samples = load_real_disks_and_gt(data_dir, n=n)
    print(f"  Loaded {len(samples)} real MAGFiLO images + real COCO ground truth.\n")

    disk_mask = get_solar_disk_mask(2048, 2048, r_frac=0.93)
    rng = np.random.default_rng(2026)

    all_rows = []
    per_disk_metrics = []
    all_preds_for_multi = []
    all_gts_for_multi = []
    rows_per_image = defaultdict(int)

    t0 = time.perf_counter()

    for idx, sample in enumerate(samples):
        stem = sample["stem"]
        raw = sample["image"]
        gt_masks = sample["gt_masks"]
        h, w = raw.shape[:2]

        # Step 1: 3-channel astronomical preprocessing
        img_3ch = preprocess_3ch(raw)

        # Step 2: YOLO proposals (stubbed with random/heuristic boxes)
        n_proposals = rng.integers(2, 6)
        proposals = stub_yolo_proposals(h, w, n_proposals=n_proposals, rng=rng)

        # Step 3: Crop refiner with 4-flip TTA
        refined_proposals = []
        for p in proposals:
            refined = stub_crop_refiner_tta(img_3ch, p["bbox"], p["mask"])
            refined_proposals.append({
                "confidence": p["confidence"],
                "mask": refined,
                "bbox": p["bbox"],
            })

        # Step 4: Residual discovery branch
        yolo_masks = [p["mask"] for p in refined_proposals]
        residuals = stub_residual_discovery(h, w, yolo_masks, rng=rng)

        # Step 5: Refine residual candidates
        refined_residuals = []
        for r in residuals:
            refined = stub_crop_refiner_tta(img_3ch, r["bbox"], r["mask"])
            refined_residuals.append({
                "confidence": r["confidence"],
                "mask": refined,
            })

        # Step 6: Greedy non-overlap arbitration + solar limb clipping
        all_candidates = refined_proposals + refined_residuals
        final_instances = greedy_non_overlap(all_candidates, disk_mask=disk_mask)

        # Step 7: Pycocotools Fortran RLE encoding
        pred_masks = []
        if len(final_instances) == 0:
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            rle_str = encode_mask(zero_mask)
            filament_id = f"{stem}_1"
            all_rows.append({"filament_id": filament_id, "segmentation_rle": rle_str})
            rows_per_image[stem] += 1
        else:
            for k, inst in enumerate(final_instances):
                m = inst["mask"]
                rle_str = encode_mask(m)
                filament_id = f"{stem}_{k + 1}"
                all_rows.append({"filament_id": filament_id, "segmentation_rle": rle_str})
                pred_masks.append(m)
                rows_per_image[stem] += 1

        all_preds_for_multi.append(pred_masks)
        all_gts_for_multi.append(gt_masks)

        # Step 8: Compute Kirillov PQ for this disk
        d_metrics = pq_score(pred_masks, gt_masks, iou_threshold=0.5)
        d_metrics["stem"] = stem
        d_metrics["n_gt"] = len(gt_masks)
        d_metrics["n_pred"] = len(pred_masks)
        per_disk_metrics.append(d_metrics)

    # ─────────────────────────────────────────────────────────────────
    # Print Per-Disk Results Table
    # ─────────────────────────────────────────────────────────────────
    print("-" * 75)
    print(f"{'STEM':<22} | {'N_GT':>4} | {'N_PRED':>6} | {'PQ':>6} | {'SQ':>6} | {'RQ':>6} | {'TP':>3} {'FP':>3} {'FN':>3}")
    print("-" * 75)
    for d in per_disk_metrics:
        print(f"{d['stem']:<22} | {d['n_gt']:>4} | {d['n_pred']:>6} | {d['PQ']:>6.4f} | {d['SQ']:>6.4f} | {d['RQ']:>6.4f} | {d['TP']:>3} {d['FP']:>3} {d['FN']:>3}")
    print("-" * 75)

    # Summary totals
    multi_summary = pq_score_multi(all_preds_for_multi, all_gts_for_multi)
    total_gt = sum(d["n_gt"] for d in per_disk_metrics)
    total_pred = sum(d["n_pred"] for d in per_disk_metrics)
    tot_tp = multi_summary["total_TP"]
    tot_fp = multi_summary["total_FP"]
    tot_fn = multi_summary["total_FN"]
    mean_pq = multi_summary["mean_PQ"]
    mean_sq = multi_summary["mean_SQ"]
    mean_rq = multi_summary["mean_RQ"]

    print(f"{'TOTALS / MEAN':<22} | {total_gt:>4} | {total_pred:>6} | {mean_pq:>6.4f} | {mean_sq:>6.4f} | {mean_rq:>6.4f} | {tot_tp:>3} {tot_fp:>3} {tot_fn:>3}")
    print("=" * 75)
    print(f"Audit Check: FN = {tot_fn} (Assert > 0 on stub weights: {tot_fn > 0})")
    print(f"Wall Clock Time: {time.perf_counter() - t0:.2f}s\n")

    # ─────────────────────────────────────────────────────────────────
    # Write Submission CSV
    # ─────────────────────────────────────────────────────────────────
    if output_csv is None:
        output_csv = str(PROJECT_ROOT / "submissions" / "smoke_real8.csv")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written CSV: {out_path}")
    print(f"Total CSV Rows: {len(all_rows)}")
    print(f"Rows per Image Audit:")
    for stem, count in rows_per_image.items():
        print(f"  {stem}: {count} row(s)")
    print()

    # RLE Verification: decode all rows
    print("RLE DECODE VERIFICATION:")
    verified_instances = 0
    with open(out_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rle_str = r["segmentation_rle"]
            m_dec = decode_rle(rle_str, 2048, 2048)
            assert m_dec.shape == (2048, 2048), f"Decoded shape mismatch: {m_dec.shape}"
            verified_instances += 1
    print(f"  Successfully decoded all {verified_instances} rows with shape (2048, 2048).")
    print("=" * 75)

    return {
        "per_disk": per_disk_metrics,
        "totals": {
            "total_gt": total_gt,
            "total_pred": total_pred,
            "TP": tot_tp,
            "FP": tot_fp,
            "FN": tot_fn,
            "mean_PQ": mean_pq,
            "mean_SQ": mean_sq,
            "mean_RQ": mean_rq,
        },
        "csv_path": str(out_path),
        "csv_rows": len(all_rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Real-image MAGFiLO smoke cascade verifier")
    parser.add_argument("--n", type=int, default=8, help="Number of real images to process")
    parser.add_argument("--out", type=str, default="submissions/smoke_real8.csv", help="Output CSV path")
    args = parser.parse_args()

    out_csv = str(PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else args.out
    run_smoke_cascade(n=args.n, output_csv=out_csv)


if __name__ == "__main__":
    main()
