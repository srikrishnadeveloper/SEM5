"""
v8_1 Stage 4 — Inference Diagnostic & Operating Point Grid Sweep (Directive R2).

Master: Grok | Executor: Antigravity
Target: Kaggle Solar Filament Segmentation Challenge 2026

R2 Objectives:
1. Stage A (Diagnose): Score validation physical JPEGs across each annotator observation separately.
2. Stage B (Inference Sweep): Grid search over conf, min_area, yolo_mask_fallback, overlap_mode.
   Default baseline: conf=0.25, min_area=100, yolo_mask_fallback=0, overlap_mode=trim (0.350 setting).
   Select argmax pq_mean operating point and dump histogram.
3. Stage C (Test Submit): Run inference on 180 test disks with frozen winner and verify contract.
"""

import argparse
import csv
import json
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

from metrics.pq import pq_score, encode_mask, decode_rle
from v8_1.config import (
    YOLOConfig,
    CropRefinerConfig,
    CascadeConfig,
    MAGFILO_DIR,
    YOLO_DATA_DIR,
    OUT,
    RUNS_DIR,
    MODELS_DIR,
)
from v8_1.geometry import compute_adaptive_crop_side, square_bounds_from_bbox
import importlib
_infer_module = importlib.import_module("v8_1.3_infer_cascade")
preprocess_3ch = _infer_module.preprocess_3ch
get_solar_disk_mask = _infer_module.get_solar_disk_mask
refine_crop_tta = _infer_module.refine_crop_tta
arbitrate_instances = _infer_module.arbitrate_instances
run_cascade_inference = _infer_module.run_cascade_inference

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_validation_ground_truth(data_dir: Path, yolo_data_dir: Path):
    """Load the 128 unique validation physical JPEGs and their ground truth annotator masks."""
    ann_path = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"MAGFiLO training JSON not found at: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Map image_id -> file_name and file_name -> list of image_ids
    img_id_to_fn = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
    fn_to_img_ids = defaultdict(list)
    for iid, fn in img_id_to_fn.items():
        fn_to_img_ids[fn].append(iid)

    # Collect segmentation polygons per image_id (all categories 1-4 are filaments)
    img_id_to_polys = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        segs = ann.get("segmentation", [])
        if isinstance(segs, list):
            for poly in segs:
                if len(poly) >= 6 and len(poly) % 2 == 0:
                    img_id_to_polys[ann["image_id"]].append(poly)

    # Identify val stems from yolo_data_dir labels
    val_labels_dir = yolo_data_dir / "labels" / "val"
    if val_labels_dir.exists():
        val_label_files = list(val_labels_dir.glob("*.txt"))
        val_physical_stems = sorted(list({p.stem.split("_ann_")[0] for p in val_label_files}))
    else:
        # Fallback to year GroupKFold logic if yolo_seg directory not yet materialized
        all_stems = sorted(list(fn_to_img_ids.keys()))
        years = [s[:4] for s in all_stems]
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(all_stems, groups=years))
        _, val_idx = splits[0]
        val_physical_stems = sorted([Path(all_stems[i]).stem for i in val_idx])

    val_img_dir = data_dir / "train" / "train_images"
    val_gt_by_stem = {}

    for stem in val_physical_stems:
        fn = f"{stem}.jpeg" if (val_img_dir / f"{stem}.jpeg").exists() else f"{stem}.jpg"
        img_ids = fn_to_img_ids.get(fn, [])
        annotator_polys = [img_id_to_polys.get(iid, []) for iid in img_ids]
        val_gt_by_stem[stem] = {
            "fn": fn,
            "img_path": val_img_dir / fn,
            "annotators_polys": annotator_polys,
        }

    return val_physical_stems, val_gt_by_stem


def extract_val_candidates_pool(
    val_physical_stems: list[str],
    val_gt_by_stem: dict,
    yolo_model,
    refiner_model,
    device_yolo,
    device_refiner,
    base_conf: float = 0.10,
) -> dict:
    """Run YOLO proposer and Crop Refiner once at base_conf=0.10 on 128 val images.
    
    Caches candidates with RLE-compressed masks to eliminate redundant neural evaluations.
    """
    print(f"\n[STAGE A] Extracting candidate pools on {len(val_physical_stems)} val disks (base_conf={base_conf:.2f})...")
    candidates_by_stem = {}
    t0 = time.perf_counter()

    for idx, stem in enumerate(val_physical_stems):
        img_path = val_gt_by_stem[stem]["img_path"]
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"FATAL: Unable to decode validation image at {img_path}")

        h, w = raw.shape[:2]
        img_3ch = preprocess_3ch(raw)

        # Run YOLO at lowest sweep confidence
        yolo_preds = yolo_model.predict(
            source=str(img_path),
            imgsz=YOLOConfig.IMGSZ,
            conf=base_conf,
            device=device_yolo,
            verbose=False,
        )

        stem_candidates = []
        if yolo_preds and yolo_preds[0].boxes is not None and len(yolo_preds[0].boxes) > 0:
            boxes = yolo_preds[0].boxes.xyxy.cpu().numpy()
            confs = yolo_preds[0].boxes.conf.cpu().numpy()
            yolo_masks_raw = yolo_preds[0].masks.data.cpu().numpy() if (yolo_preds and yolo_preds[0].masks is not None) else None

            for b_idx, (box, conf) in enumerate(zip(boxes, confs)):
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(w, int(box[2]))
                y2 = min(h, int(box[3]))
                bbox_2048 = [x1, y1, x2, y2]

                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue

                # Refine with Crop U-Net + 4-flip TTA
                refined_mask = refine_crop_tta(
                    refiner_model=refiner_model,
                    img_3ch=img_3ch,
                    bbox=bbox_2048,
                    device=device_refiner,
                    target_size=CropRefinerConfig.TARGET_SIZE,
                    use_tta=True,
                )

                # Store YOLO-seg mask if available
                yolo_m = None
                if yolo_masks_raw is not None and b_idx < len(yolo_masks_raw):
                    yolo_m = cv2.resize(yolo_masks_raw[b_idx].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                # Cache candidate as COCO RLE to keep RAM < 10 MB
                stem_candidates.append({
                    "confidence": float(conf),
                    "bbox": bbox_2048,
                    "refined_rle": encode_mask(refined_mask) if refined_mask.sum() > 0 else None,
                    "refined_area": int(refined_mask.sum()),
                    "yolo_rle": encode_mask(yolo_m) if (yolo_m is not None and yolo_m.sum() > 0) else None,
                    "yolo_area": int(yolo_m.sum()) if yolo_m is not None else 0,
                })

        candidates_by_stem[stem] = stem_candidates

        if (idx + 1) % 32 == 0 or (idx + 1) == len(val_physical_stems):
            elapsed = time.perf_counter() - t0
            print(f"  [{idx + 1:03d}/{len(val_physical_stems):03d}] Cached {stem} ({elapsed:.1f}s)")

    print(f"[STAGE A] Extraction complete in {time.perf_counter() - t0:.1f}s.")
    return candidates_by_stem


def evaluate_grid_point(
    conf: float,
    min_area: int,
    yolo_fallback: int,
    overlap_mode: str,
    val_physical_stems: list[str],
    val_gt_by_stem: dict,
    candidates_by_stem: dict,
    disk_mask: np.ndarray,
) -> dict:
    """Evaluate one parameter combination across all 128 validation disks."""
    disk_pq_means, disk_pq_maxs = [], []
    disk_sq_means, disk_rq_means = [], []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    tot_pred_instances = 0
    preds_per_disk = []

    for stem in val_physical_stems:
        # 1. Rasterize ground-truth instances for all annotator observations on this physical image
        annotators_polys = val_gt_by_stem[stem]["annotators_polys"]
        annotator_instances = []
        for polys in annotators_polys:
            masks = []
            for p in polys:
                m = np.zeros((2048, 2048), dtype=np.uint8)
                pts = np.asarray(p, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(m, [pts], 1)
                if m.sum() > 0:
                    masks.append(m)
            annotator_instances.append(masks)

        # 2. Select candidates for this operating point
        raw_candidates = candidates_by_stem.get(stem, [])
        filtered_candidates = []

        for c in raw_candidates:
            if c["confidence"] < conf:
                continue

            ref_area = c["refined_area"]
            yolo_area = c["yolo_area"]

            # YOLO mask fallback logic:
            if yolo_fallback == 1 and ref_area < min_area and yolo_area >= min_area:
                mask = decode_rle(c["yolo_rle"], 2048, 2048)
            elif ref_area >= min_area and c["refined_rle"] is not None:
                mask = decode_rle(c["refined_rle"], 2048, 2048)
            else:
                continue

            filtered_candidates.append({
                "confidence": c["confidence"],
                "mask": mask,
            })

        # 3. Arbitrate instances using overlap_mode and min_area
        accepted = arbitrate_instances(
            filtered_candidates,
            disk_mask=disk_mask,
            min_area=min_area,
            overlap_mode=overlap_mode,
        )

        pred_masks = [inst["mask"] for inst in accepted]
        tot_pred_instances += len(pred_masks)
        preds_per_disk.append(len(pred_masks))

        # 4. Score against each annotator independently
        ann_pqs, ann_sqs, ann_rqs = [], [], []
        for ann_gt in annotator_instances:
            score_dict = pq_score(pred_masks, ann_gt, iou_threshold=0.5)
            ann_pqs.append(score_dict["PQ"])
            ann_sqs.append(score_dict["SQ"])
            ann_rqs.append(score_dict["RQ"])
            tot_tp += score_dict["TP"]
            tot_fp += score_dict["FP"]
            tot_fn += score_dict["FN"]

        if ann_pqs:
            disk_pq_means.append(np.mean(ann_pqs))
            disk_pq_maxs.append(np.max(ann_pqs))
            disk_sq_means.append(np.mean(ann_sqs))
            disk_rq_means.append(np.mean(ann_rqs))

    mean_pq = float(np.mean(disk_pq_means)) if disk_pq_means else 0.0
    max_pq = float(np.mean(disk_pq_maxs)) if disk_pq_maxs else 0.0
    mean_sq = float(np.mean(disk_sq_means)) if disk_sq_means else 0.0
    mean_rq = float(np.mean(disk_rq_means)) if disk_rq_means else 0.0

    return {
        "conf": conf,
        "min_area": min_area,
        "yolo_fallback": yolo_fallback,
        "overlap_mode": overlap_mode,
        "pq_mean": mean_pq,
        "pq_max": max_pq,
        "sq_mean": mean_sq,
        "rq_mean": mean_rq,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "n_pred": tot_pred_instances,
        "preds_per_disk": preds_per_disk,
    }


def run_r2_sweep(args):
    """Execute complete Stage A, Stage B, and Stage C workflow."""
    print("=" * 85)
    print("V8_1 STAGE 4: GROK DIRECTIVE R2 INFERENCE & OPERATING POINT SWEEP")
    print("=" * 85)

    data_dir = MAGFILO_DIR
    yolo_data_dir = YOLO_DATA_DIR
    disk_mask = get_solar_disk_mask(2048, 2048, r_frac=CascadeConfig.SOLAR_DISK_R_FRAC)

    # 1. Resolve weights
    yolo_ckpt = YOLOConfig.resolve_weights(args.weights if hasattr(args, "weights") else None)
    refiner_ckpt = CropRefinerConfig.resolve_ckpt(args.refiner_weights if hasattr(args, "refiner_weights") else None)

    print(f"  YOLO Weights Resolved:    {yolo_ckpt} (exists: {yolo_ckpt.exists()})")
    print(f"  Refiner Checkpoint:       {refiner_ckpt} (exists: {refiner_ckpt.exists()})")

    if not yolo_ckpt.exists() or not refiner_ckpt.exists():
        if args.check_weights_only:
            print("\n[CHECK-WEIGHTS-ONLY] Weights missing. Retraining required.")
            return None
        raise FileNotFoundError(
            f"FATAL: Weights missing! YOLO: {yolo_ckpt} ({yolo_ckpt.exists()}), Refiner: {refiner_ckpt} ({refiner_ckpt.exists()}). "
            f"Attach completed kernel output or run training cells first."
        )

    if args.check_weights_only:
        print("\n[CHECK-WEIGHTS-ONLY] Weights confirmed present! Skipping retraining.")
        return True

    # 2. Load Models
    from ultralytics import YOLO
    import segmentation_models_pytorch as smp

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_yolo = 0 if n_gpus > 0 else "cpu"
    refiner_gpu_idx = 1 if n_gpus >= 2 else (0 if n_gpus == 1 else "cpu")
    device_refiner = torch.device(f"cuda:{refiner_gpu_idx}" if n_gpus > 0 else "cpu")

    print(f"  Model Placement: YOLO -> cuda:{device_yolo} | Refiner -> {device_refiner}")

    yolo_model = YOLO(str(yolo_ckpt))
    refiner_model = smp.Unet(
        encoder_name=CropRefinerConfig.ENCODER,
        in_channels=CropRefinerConfig.IN_CHANNELS,
        classes=CropRefinerConfig.CLASSES,
    ).to(device_refiner)
    refiner_model.load_state_dict(torch.load(refiner_ckpt, map_location=device_refiner))
    refiner_model.eval()

    # 3. Load Validation Set & Annotations
    val_physical_stems, val_gt_by_stem = load_validation_ground_truth(data_dir, yolo_data_dir)
    print(f"  Holdout unique physical validation JPEGs: {len(val_physical_stems)}")

    # Ground truth instances per image statistics
    gt_counts_per_disk = []
    for stem in val_physical_stems:
        for polys in val_gt_by_stem[stem]["annotators_polys"]:
            gt_counts_per_disk.append(len(polys))
    gt_counts_arr = np.array(gt_counts_per_disk)
    print(f"  GT Instances per Image (all annotators): Mean={gt_counts_arr.mean():.2f} | Median={np.median(gt_counts_arr):.1f} | Min={gt_counts_arr.min()} | Max={gt_counts_arr.max()}")

    # 4. Stage A: Extract candidate pools (runs heavy neural forward passes once)
    candidates_by_stem = extract_val_candidates_pool(
        val_physical_stems,
        val_gt_by_stem,
        yolo_model,
        refiner_model,
        device_yolo,
        device_refiner,
        base_conf=0.10,
    )

    # 5. Stage B: Inference Grid Sweep
    confs = [0.10, 0.15, 0.20, 0.25, 0.35]
    min_areas = [50, 100, 200, 400]
    fallbacks = [0, 1]
    overlap_modes = ["trim", "allow"]

    total_points = len(confs) * len(min_areas) * len(fallbacks) * len(overlap_modes)
    print(f"\n[STAGE B] Running Inference Grid Search across {total_points} operating points...")

    results_table = []
    t_sweep_start = time.perf_counter()

    for conf in confs:
        for ma in min_areas:
            for fb in fallbacks:
                for om in overlap_modes:
                    res = evaluate_grid_point(
                        conf=conf,
                        min_area=ma,
                        yolo_fallback=fb,
                        overlap_mode=om,
                        val_physical_stems=val_physical_stems,
                        val_gt_by_stem=val_gt_by_stem,
                        candidates_by_stem=candidates_by_stem,
                        disk_mask=disk_mask,
                    )
                    results_table.append(res)

    print(f"[STAGE B] Sweep completed in {time.perf_counter() - t_sweep_start:.2f}s.\n")

    # 6. Format and Print Required VAL_PQ_TABLE
    print("=" * 115)
    print("## VAL_PQ_TABLE")
    print("=" * 115)
    header = f"{'conf':<6} | {'min_area':<8} | {'fallback':<8} | {'overlap':<7} | {'PQ_mean':<8} | {'PQ_max':<8} | {'SQ':<7} | {'RQ':<7} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'n_pred':<6}"
    print(header)
    print("-" * 115)

    baseline_result = None
    for r in results_table:
        is_baseline = (r["conf"] == 0.25 and r["min_area"] == 100 and r["yolo_fallback"] == 0 and r["overlap_mode"] == "trim")
        if is_baseline:
            baseline_result = r
        marker = " <-- [0.350 BASELINE]" if is_baseline else ""
        row_str = (
            f"{r['conf']:<6.2f} | {r['min_area']:<8} | {r['yolo_fallback']:<8} | {r['overlap_mode']:<7} | "
            f"{r['pq_mean']:<8.4f} | {r['pq_max']:<8.4f} | {r['sq_mean']:<7.4f} | {r['rq_mean']:<7.4f} | "
            f"{r['tp']:<6} | {r['fp']:<6} | {r['fn']:<6} | {r['n_pred']:<6}{marker}"
        )
        print(row_str)
    print("=" * 115)

    # 7. Select Argmax Operating Point
    best_config = max(results_table, key=lambda x: x["pq_mean"])
    print("\n## CHOSEN_OPERATING_POINT")
    print(f"  Optimal Configuration by pq_mean:")
    print(f"    conf:                {best_config['conf']}")
    print(f"    min_area:            {best_config['min_area']}")
    print(f"    yolo_mask_fallback:  {best_config['yolo_fallback']}")
    print(f"    overlap_mode:        {best_config['overlap_mode']}")
    print(f"    Validation PQ_mean:  {best_config['pq_mean']:.4f} (Baseline 0.350: {baseline_result['pq_mean']:.4f})")
    print(f"    Validation PQ_max:   {best_config['pq_max']:.4f}")
    print(f"    Validation SQ:       {best_config['sq_mean']:.4f}")
    print(f"    Validation RQ:       {best_config['rq_mean']:.4f}")
    print(f"    TP / FP / FN:        {best_config['tp']} / {best_config['fp']} / {best_config['fn']}")
    print(f"    Total Pred Filaments:{best_config['n_pred']}")

    # 8. Histogram: Predictions per Image vs Ground Truth Instances
    pred_counts_arr = np.array(best_config["preds_per_disk"])
    print("\n--- INSTANCE HISTOGRAM COMPARISON (Fold-0 Validation) ---")
    print(f"  Ground Truth / Disk: Mean={gt_counts_arr.mean():.2f} | Median={np.median(gt_counts_arr):.1f} | Min={gt_counts_arr.min()} | Max={gt_counts_arr.max()}")
    print(f"  Chosen Preds / Disk: Mean={pred_counts_arr.mean():.2f} | Median={np.median(pred_counts_arr):.1f} | Min={pred_counts_arr.min()} | Max={pred_counts_arr.max()}")

    # Save summary JSON
    summary_path = OUT / "v8_1_r2_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {k: v for k, v in baseline_result.items() if k != "preds_per_disk"},
            "chosen_operating_point": {k: v for k, v in best_config.items() if k != "preds_per_disk"},
            "all_results": [{k: v for k, v in r.items() if k != "preds_per_disk"} for r in results_table],
        }, f, indent=2)
    print(f"Saved sweep summary to {summary_path}")

    # 9. Stage C: Test Submission Generation
    if args.run_test:
        print("\n" + "=" * 85)
        print("[STAGE C] Generating Test Submission using Frozen Operating Point...")
        print("=" * 85)
        
        # Prepare namespace for run_cascade_inference
        test_args = argparse.Namespace(
            conf=best_config["conf"],
            min_area=best_config["min_area"],
            yolo_fallback=best_config["yolo_fallback"],
            overlap_mode=best_config["overlap_mode"],
            tta=True,
            weights=str(yolo_ckpt),
            refiner_weights=str(refiner_ckpt),
            out=args.out,
        )
        run_cascade_inference(test_args)

    return best_config


def main():
    parser = argparse.ArgumentParser(description="v8_1 R2 Diagnostic & Inference Sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit 0")
    parser.add_argument("--check-weights-only", action="store_true", help="Check if weights exist without running sweep")
    parser.add_argument("--weights", type=str, default=None, help="Explicit path to YOLO weights (best.pt)")
    parser.add_argument("--refiner-weights", type=str, default=None, help="Explicit path to Refiner weights (crop_refiner_r34.pth)")
    parser.add_argument("--run-test", action="store_true", default=True, help="Execute Stage C test inference with winning operating point")
    parser.add_argument("--out", type=str, default=str(OUT / "submission.csv"), help="Output submission CSV path")
    args = parser.parse_args()

    if args.dry_run:
        print("v8_1 4_sweep_inference.py dry run OK. Exiting 0 on CPU.")
        sys.exit(0)

    if not torch.cuda.is_available() and not args.check_weights_only:
        print("[ERROR] CUDA is not available! R2 inference sweep requires GPU.")
        sys.exit(1)

    run_r2_sweep(args)


if __name__ == "__main__":
    main()
