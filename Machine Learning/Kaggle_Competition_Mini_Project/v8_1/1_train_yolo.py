"""
v8_1 Stage 1 — Train YOLO11s-seg Proposer on MAGFiLO (Frozen Specification).

Master: ChatGPT | Builder: Antigravity
Target: Kaggle Solar Filament Segmentation Challenge 2026

Frozen Rules:
- Model: yolo11s-seg.pt @ 1024
- 20 epochs, batch 2, single GPU cuda:0, AMP=True, mosaic=0.5, close_mosaic=2
- Refuse to start if not torch.cuda.is_available() (unless --dry-run)
- Multi-annotator validation PQ: evaluate on unique file_names, computing
  pq_mean and pq_max across annotators. Sweep conf in {0.15, 0.25, 0.35}.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.pq import pq_score, pq_score_multi
from v8_1.config import YOLOConfig, MAGFILO_DIR, YOLO_DATA_DIR, RUNS_DIR


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_frozen_config():
    """Print the frozen YOLO11s-seg specification for master review."""
    print("=" * 75)
    print("V8_1 STAGE 1: YOLO11s-seg PROPOSER (FROZEN CONFIGURATION)")
    print("=" * 75)
    print(f"  Model Architecture:    {YOLOConfig.MODEL}")
    print(f"  Input Resolution:      {YOLOConfig.IMGSZ}x{YOLOConfig.IMGSZ}")
    print(f"  Epochs:                {YOLOConfig.EPOCHS}")
    print(f"  Batch Size:            {YOLOConfig.BATCH}")
    print(f"  Training Device:       cuda:{YOLOConfig.DEVICE} (Single GPU pass 1)")
    print(f"  Mixed Precision (AMP): {YOLOConfig.AMP}")
    print(f"  Mosaic Augmentation:   {YOLOConfig.MOSAIC} (close_mosaic: last {YOLOConfig.CLOSE_MOSAIC} epochs)")
    print(f"  Geometry Augmentation: degrees={YOLOConfig.DEGREES}, fliplr={YOLOConfig.FLIPLR}, flipud={YOLOConfig.FLIPUD}")
    print(f"  Copy-Paste:            DISABLED (until 's' baseline holds)")
    print(f"  Overlap Mask:          {YOLOConfig.OVERLAP_MASK}")
    print(f"  Mask Ratio:            {YOLOConfig.MASK_RATIO}")
    print(f"  Workers:               {YOLOConfig.WORKERS}")
    print(f"  Dataset Config:        {YOLOConfig.DATA_YAML}")
    print(f"  Project Run Dir:       {YOLOConfig.PROJECT}/{YOLOConfig.NAME}")
    print(f"  Target Checkpoint:     {YOLOConfig.WEIGHTS_PATH}")
    print(f"  CUDA Available:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'})")
    print("=" * 75)


def evaluate_val_pq(model, data_dir: Path, yolo_data_dir: Path, conf_thresholds=(0.15, 0.25, 0.35)):
    """Evaluate YOLO-seg predictions against unique physical JPEGs.

    For each val file_name:
    1. Rasterize each annotator observation separately.
    2. Compute Kirillov PQ vs each annotator.
    3. Record pq_mean and pq_max.
    Sweep conf in {0.15, 0.25, 0.35} on pq_mean.
    """
    print("\n" + "=" * 75)
    print("HOLDOUT MULTI-ANNOTATOR KIRILLOV PQ EVALUATION")
    print("=" * 75)

    # Load ground truth annotations
    train_dir = data_dir / "train"
    json_path = next(train_dir.glob("*.json"))
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map file_name -> list of annotator observation image_ids
    fn_to_img_ids = defaultdict(list)
    for img in coco.get("images", []):
        fn_to_img_ids[img["file_name"]].append(img["id"])

    # Map image_id -> list of polygons (all categories 1, 2, 3, 4 are solar filaments)
    img_id_to_polys = defaultdict(list)
    for ann in coco.get("annotations", []):
        segs = ann.get("segmentation", [])
        if isinstance(segs, list):
            for poly in segs:
                if len(poly) >= 6 and len(poly) % 2 == 0:
                    img_id_to_polys[ann["image_id"]].append(poly)

    # Identify val files from yolo_data_dir
    val_label_files = list((yolo_data_dir / "labels" / "val").glob("*.txt"))
    val_physical_stems = sorted(list({p.stem.split("_ann_")[0] for p in val_label_files}))
    val_img_dir = data_dir / "train" / "train_images"

    print(f"  Holdout unique physical JPEGs to score: {len(val_physical_stems)}")

    # Record polygon coordinates per val file (streaming, zero RAM overhead)
    val_gt_by_file = {}
    for stem in val_physical_stems:
        fn = f"{stem}.jpeg" if (val_img_dir / f"{stem}.jpeg").exists() else f"{stem}.jpg"
        img_ids = fn_to_img_ids.get(fn, [])
        annotator_polys = [img_id_to_polys.get(iid, []) for iid in img_ids]
        val_gt_by_file[stem] = {"fn": fn, "annotators_polys": annotator_polys}

    results_table = []

    for conf in conf_thresholds:
        print(f"\n--- Sweeping Confidence Threshold: {conf:.2f} ---")
        disk_pq_means, disk_pq_maxs = [], []
        disk_sq_means, disk_rq_means = [], []
        tot_tp, tot_fp, tot_fn = 0, 0, 0

        for stem in val_physical_stems:
            fn = val_gt_by_file[stem]["fn"]
            img_path = val_img_dir / fn
            annotators_polys = val_gt_by_file[stem]["annotators_polys"]

            # Stream-rasterize GT masks on-the-fly for this physical image only
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

            # Run YOLO prediction
            preds = model.predict(
                source=str(img_path),
                imgsz=YOLOConfig.IMGSZ,
                conf=conf,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )

            pred_masks = []
            if preds and preds[0].masks is not None:
                # Resize YOLO masks back to 2048x2048
                raw_masks = preds[0].masks.data.cpu().numpy()
                for rm in raw_masks:
                    m_2048 = cv2.resize(rm.astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST)
                    if m_2048.sum() > 0:
                        pred_masks.append(m_2048)

            # Score against each annotator independently
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

            # Release full 2048 masks immediately
            del pred_masks, annotator_instances

        mean_pq = float(np.mean(disk_pq_means)) if disk_pq_means else 0.0
        max_pq = float(np.mean(disk_pq_maxs)) if disk_pq_maxs else 0.0
        mean_sq = float(np.mean(disk_sq_means)) if disk_sq_means else 0.0
        mean_rq = float(np.mean(disk_rq_means)) if disk_rq_means else 0.0
        results_table.append({
            "conf": conf,
            "pq_mean": mean_pq,
            "pq_max": max_pq,
            "sq_mean": mean_sq,
            "rq_mean": mean_rq,
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
        })
        print(f"  Confidence {conf:.2f} -> pq_mean: {mean_pq:.4f} | pq_max: {max_pq:.4f} | SQ: {mean_sq:.4f} | RQ: {mean_rq:.4f} (TP={tot_tp}, FP={tot_fp}, FN={tot_fn})")

    print("\n" + "=" * 85)
    print("SWEEP SUMMARY TABLE (Unique Physical Disks)")
    print("=" * 85)
    print(f"{'CONFIDENCE':<12} | {'pq_mean (Selector)':<20} | {'pq_max':<12} | {'SQ':<10} | {'RQ':<10} | {'TP/FP/FN'}")
    print("-" * 85)
    for r in results_table:
        print(f"{r['conf']:<12.2f} | {r['pq_mean']:<20.4f} | {r['pq_max']:<12.4f} | {r['sq_mean']:<10.4f} | {r['rq_mean']:<10.4f} | {r['tp']}/{r['fp']}/{r['fn']}")
    print("=" * 85)

    best_conf = max(results_table, key=lambda x: x["pq_mean"])
    print(f"[OPTIMAL] Optimal Threshold by pq_mean: conf={best_conf['conf']} (pq_mean={best_conf['pq_mean']:.4f})")
    return results_table
    return results_table


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11s-seg Stage 1 Proposer")
    parser.add_argument("--dry-run", action="store_true", help="Print frozen config and exit 0 without training")
    parser.add_argument("--eval-only", action="store_true", help="Run holdout PQ evaluation only using best.pt")
    parser.add_argument("--max-train", type=int, default=None, help="Limit training samples for 1-epoch GPU smoke test")
    parser.add_argument("--batch", type=int, default=YOLOConfig.BATCH, help="Batch size (drop to 2 on T4 if OOM)")
    parser.add_argument("--epochs", type=int, default=YOLOConfig.EPOCHS, help="Number of training epochs")
    parser.add_argument("--weights", type=str, default=None, help="Path to existing weights for evaluation")
    args = parser.parse_args()

    # Step 1: Handle --dry-run
    if args.dry_run:
        print_frozen_config()
        print("Dry run completed successfully. Exiting 0 on CPU.\n")
        sys.exit(0)

    # Step 2: Enforce CUDA requirement
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available! YOLO11s-seg training refuses to run on CPU.")
        print("Run with --dry-run to inspect configuration on CPU.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' package is not installed.")
        print("Install via: pip install ultralytics")
        sys.exit(1)

    print_frozen_config()

    # Step 3: Handle --eval-only
    if args.eval_only:
        ckpt = Path(args.weights) if args.weights else YOLOConfig.WEIGHTS_PATH
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found at {ckpt}")
            sys.exit(1)
        model = YOLO(str(ckpt))
        evaluate_val_pq(model, data_dir=MAGFILO_DIR, yolo_data_dir=YOLO_DATA_DIR)
        return

    # Step 4: Launch Training
    print("[INFO] Launching YOLO11s-seg Training on GPU...")
    model = YOLO(YOLOConfig.MODEL)

    data_yaml = str(YOLOConfig.DATA_YAML)
    if not Path(data_yaml).exists():
        print(f"[ERROR] Dataset yaml not found at {data_yaml}. Run convert_coco_to_yolo.py first.")
        sys.exit(1)

    train_kwargs = {
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": YOLOConfig.IMGSZ,
        "batch": args.batch,
        "amp": YOLOConfig.AMP,
        "mosaic": YOLOConfig.MOSAIC,
        "close_mosaic": YOLOConfig.CLOSE_MOSAIC,
        "degrees": YOLOConfig.DEGREES,
        "flipud": YOLOConfig.FLIPUD,
        "fliplr": YOLOConfig.FLIPLR,
        "workers": YOLOConfig.WORKERS,
        "project": YOLOConfig.PROJECT,
        "name": YOLOConfig.NAME,
        "device": YOLOConfig.DEVICE,
        "overlap_mask": YOLOConfig.OVERLAP_MASK,
        "mask_ratio": YOLOConfig.MASK_RATIO,
        "exist_ok": True,
        "save": True,
        "plots": False,
    }

    if args.max_train:
        print(f"  [SMOKE] Limiting to {args.max_train} training samples for quick validation.")
        train_kwargs["fraction"] = min(1.0, args.max_train / 951.0)

    results = model.train(**train_kwargs)
    print("[OK] YOLO11s-seg training run completed!")

    # Memory cleanup & VRAM logging
    del model, results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for dev_idx in range(torch.cuda.device_count()):
            res_mb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
            alloc_mb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
            print(f"  [GPU {dev_idx}] Memory Reserved: {res_mb:.1f} MB | Allocated: {alloc_mb:.1f} MB")

    # Step 5: Post-train holdout PQ evaluation
    best_weights = YOLOConfig.resolve_weights(args.weights)
    if best_weights.exists():
        print(f"  [BEST WEIGHTS LOCATED]: {best_weights}")
        trained_model = YOLO(str(best_weights))
        evaluate_val_pq(trained_model, data_dir=MAGFILO_DIR, yolo_data_dir=YOLO_DATA_DIR)
        del trained_model
        torch.cuda.empty_cache()
    else:
        print(f"[WARN] Best weights not found at {best_weights}, skipping PQ evaluation.")


if __name__ == "__main__":
    main()
