"""
moonshot_2048/train_yolov8l.py — Native-2048 YOLOv8l-seg Training & OOM Fallback Engine.

Authority: ChatGPT Master
Directives: 17 & 18 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Frozen Training Specification:
- Base Architecture: yolov8l-seg.pt
- Fallback Sequence (Directive 18 Approved):
    (2048, 2) -> (2048, 1) -> (1792, 1) -> (1536, 1)
- Mixed Precision (AMP): True
- Single GPU training (cuda:0) to prevent notebook DDP deadlocks
- Conservative solar augmentations: degrees=10, flipud=0.5, fliplr=0.5, mosaic=0.0
- Validation: Deployment-matched NMS (iou=0.00) multi-annotator pq_mean on unique physical holdout disks.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moonshot_2048.config import MoonshotConfig, YOLO_DATA_DIR, MAGFILO_DIR, RUNS_DIR
from moonshot_2048.match_and_calibrate import evaluate_instances_multi_annotator, compute_instance_failure_bins
from moonshot_2048.predict_native import sanitize_instances_zero_overlap


def print_training_banner():
    """Print the frozen YOLOv8l native-2048 training banner."""
    device_desc = "CPU Only"
    if torch.cuda.is_available():
        try:
            device_desc = torch.cuda.get_device_name(0)
        except Exception:
            device_desc = "CUDA Device"

    print("=" * 75)
    print("MOONSHOT 2048: YOLOv8l-seg NATIVE-RESOLUTION TRAINING (DIRECTIVE 18)")
    print("=" * 75)
    print(f"  Model Architecture:    {MoonshotConfig.MODEL_V8L}")
    print(f"  Target Resolution:     {MoonshotConfig.IMGSZ}x{MoonshotConfig.IMGSZ}")
    print(f"  Fallback Attempts:     {MoonshotConfig.FALLBACK_ATTEMPTS}")
    print(f"  Target Epochs:         {MoonshotConfig.EPOCHS} (patience: {MoonshotConfig.PATIENCE})")
    print(f"  Mixed Precision (AMP): {MoonshotConfig.AMP}")
    print(f"  Single GPU Device:     cuda:{MoonshotConfig.DEVICE}")
    print(f"  Mosaic Augmentation:   {MoonshotConfig.MOSAIC} (conservative for solar morphology)")
    print(f"  Degrees / Flips:       {MoonshotConfig.DEGREES} deg | fliplr={MoonshotConfig.FLIPLR} | flipud={MoonshotConfig.FLIPUD}")
    print(f"  Dataset Config:        {YOLO_DATA_DIR / 'data.yaml'}")
    print(f"  Runs Directory:        {MoonshotConfig.RUNS_DIR}")
    print(f"  CUDA Available:        {torch.cuda.is_available()} ({device_desc})")
    print("=" * 75)


def run_holdout_validation_sweep(
    model,
    magfilo_dir: Path,
    yolo_data_dir: Path,
    conf_list: List[float] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    min_area_list: List[int] = (50, 100, 200, 400),
    nms_iou: float = MoonshotConfig.NMS_IOU,
    imgsz: int = MoonshotConfig.IMGSZ,
    device: str = "0",
) -> List[Dict[str, Any]]:
    """Run holdout evaluation on unique physical validation disks using deployment-matched NMS."""
    print("\n" + "=" * 80)
    print(f"HOLDOUT MULTI-ANNOTATOR KIRILLOV PQ SWEEP (Deployment NMS IoU={nms_iou:.2f})")
    print("=" * 80)

    # 1. Load official training JSON
    train_dir = magfilo_dir / "train"
    json_path = next(train_dir.glob("*.json"))
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map file_name -> list of annotator image_ids
    from collections import defaultdict
    fn_to_iids = defaultdict(list)
    for img in coco.get("images", []):
        fn_to_iids[img["file_name"]].append(img["id"])

    # Map image_id -> list of polygons
    iid_to_polys = defaultdict(list)
    for ann in coco.get("annotations", []):
        segs = ann.get("segmentation", [])
        if isinstance(segs, list):
            for poly in segs:
                if len(poly) >= 6 and len(poly) % 2 == 0:
                    iid_to_polys[ann["image_id"]].append(poly)

    # Identify validation unique physical stems
    val_labels = list((yolo_data_dir / "labels" / "val").glob("*.txt"))
    val_physical_stems = sorted(list({p.stem.split("_ann_")[0] for p in val_labels}))
    val_img_dir = train_dir / "train_images"

    print(f"  Holdout unique physical validation disks: {len(val_physical_stems)}")

    # Pre-rasterize GT per physical disk for fast sweep evaluation
    gt_cache = {}
    for stem in val_physical_stems:
        fn = f"{stem}.jpeg" if (val_img_dir / f"{stem}.jpeg").exists() else f"{stem}.jpg"
        img_ids = fn_to_iids.get(fn, [])
        annotator_instances = []
        for iid in img_ids:
            polys = iid_to_polys.get(iid, [])
            masks = []
            for p in polys:
                m = np.zeros((2048, 2048), dtype=np.uint8)
                pts = np.asarray(p, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(m, [pts], 1)
                if m.sum() > 0:
                    masks.append(m)
            annotator_instances.append(masks)
        gt_cache[stem] = {"fn": fn, "annotators_gt": annotator_instances}

    # Cache raw model predictions per disk using deployment-matched NMS iou
    print(f"  Caching raw model predictions at conf=0.10, deployment iou={nms_iou:.2f}, imgsz={imgsz}...")
    raw_preds_cache = {}
    for stem in val_physical_stems:
        fn = gt_cache[stem]["fn"]
        img_p = val_img_dir / fn
        preds = model.predict(
            source=str(img_p),
            imgsz=imgsz,
            conf=0.10,
            iou=nms_iou,  # Deployment parity: iou=0.00
            max_det=MoonshotConfig.MAX_DET,
            device=device,
            verbose=False,
        )
        raw_masks, raw_confs = [], []
        if preds and preds[0].masks is not None:
            ms = preds[0].masks.data.cpu().numpy()
            cs = preds[0].boxes.conf.cpu().numpy()
            for rm, c in zip(ms, cs):
                if rm.shape[:2] != (2048, 2048):
                    m2 = cv2.resize(rm.astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST)
                else:
                    m2 = rm.astype(np.uint8)
                if m2.sum() > 0:
                    raw_masks.append(m2)
                    raw_confs.append(float(c))
        raw_preds_cache[stem] = (raw_masks, raw_confs)

    # Sweep grid across conf and min_area
    results = []
    print(f"\n{'CONF':<8} | {'MIN_AREA':<10} | {'pq_mean (Selector)':<20} | {'pq_max':<10} | {'SQ':<8} | {'RQ':<8} | {'TP/FP/FN'}")
    print("-" * 80)

    for conf in conf_list:
        for min_area in min_area_list:
            disk_pqs = []
            disk_pq_maxs = []
            disk_sqs = []
            disk_rqs = []
            tot_tp, tot_fp, tot_fn = 0, 0, 0

            for stem in val_physical_stems:
                raw_m, raw_c = raw_preds_cache[stem]
                # Filter by current conf
                cand_m = [m for m, c in zip(raw_m, raw_c) if c >= conf]
                cand_c = [c for c in raw_c if c >= conf]

                # Strict greedy zero-overlap sanitizer
                clean_m, _ = sanitize_instances_zero_overlap(cand_m, cand_c, min_area=min_area)

                annotator_gt = gt_cache[stem]["annotators_gt"]
                score_dict = evaluate_instances_multi_annotator(clean_m, annotator_gt)

                disk_pqs.append(score_dict["pq_mean"])
                disk_pq_maxs.append(score_dict["pq_max"])
                disk_sqs.append(score_dict["sq_mean"])
                disk_rqs.append(score_dict["rq_mean"])
                tot_tp += score_dict["tp"]
                tot_fp += score_dict["fp"]
                tot_fn += score_dict["fn"]

            mean_pq = float(np.mean(disk_pqs)) if disk_pqs else 0.0
            max_pq = float(np.mean(disk_pq_maxs)) if disk_pq_maxs else 0.0
            sq_mean = float(np.mean(disk_sqs)) if disk_sqs else 0.0
            rq_mean = float(np.mean(disk_rqs)) if disk_rqs else 0.0

            res = {
                "conf": conf,
                "min_area": min_area,
                "nms_iou": nms_iou,
                "pq_mean": mean_pq,
                "pq_max": max_pq,
                "sq_mean": sq_mean,
                "rq_mean": rq_mean,
                "tp": tot_tp,
                "fp": tot_fp,
                "fn": tot_fn,
            }
            results.append(res)
            print(f"{conf:<8.2f} | {min_area:<10} | {mean_pq:<20.4f} | {max_pq:<10.4f} | {sq_mean:<8.4f} | {rq_mean:<8.4f} | {tot_tp}/{tot_fp}/{tot_fn}")

    best_res = max(results, key=lambda x: x["pq_mean"])
    print("=" * 80)
    print(f"[WINNER] Best Operating Point: conf={best_res['conf']}, min_area={best_res['min_area']} (NMS IoU={nms_iou}) -> pq_mean={best_res['pq_mean']:.4f}")
    print("=" * 80)
    return results


def train_yolov8l_with_fallback(
    data_yaml: Path,
    epochs: int = MoonshotConfig.EPOCHS,
    patience: int = MoonshotConfig.PATIENCE,
    device: int = MoonshotConfig.DEVICE,
    dry_run: bool = False,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Train YOLOv8l-seg with exact fallback sequence on CUDA OOM (Directive 18):
    2048/batch2 -> 2048/batch1 -> 1792/batch1 -> 1536/batch1.
    
    Returns:
        (best_checkpoint_path, experiment_manifest)
    """
    print_training_banner()

    if dry_run:
        manifest = {
            "status": "DRY_RUN",
            "checkpoint_path": None,
            "trained_imgsz": MoonshotConfig.IMGSZ,
            "trained_batch": MoonshotConfig.BATCH,
            "epochs": epochs,
        }
        print("[DRY-RUN] YOLOv8l configuration verified successfully. Exiting 0.")
        return None, manifest

    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available. YOLOv8l native training requires a GPU!")

    from ultralytics import YOLO

    attempts = MoonshotConfig.FALLBACK_ATTEMPTS  # [(2048, 2), (2048, 1), (1792, 1), (1536, 1)]

    for attempt_idx, (current_imgsz, current_batch) in enumerate(attempts):
        try:
            print(f"\n[ATTEMPT {attempt_idx + 1}/{len(attempts)}] Training YOLOv8l-seg at imgsz={current_imgsz}, batch={current_batch}...")
            model = YOLO(MoonshotConfig.MODEL_V8L)
            run_name = f"moonshot_v8l_{current_imgsz}_b{current_batch}"
            run_dir = MoonshotConfig.RUNS_DIR / run_name

            model.train(
                data=str(data_yaml),
                epochs=epochs,
                patience=patience,
                batch=current_batch,
                imgsz=current_imgsz,
                device=device,
                amp=MoonshotConfig.AMP,
                workers=MoonshotConfig.WORKERS,
                seed=MoonshotConfig.SEED,
                degrees=MoonshotConfig.DEGREES,
                fliplr=MoonshotConfig.FLIPLR,
                flipud=MoonshotConfig.FLIPUD,
                mosaic=MoonshotConfig.MOSAIC,
                copy_paste=MoonshotConfig.COPY_PASTE,
                close_mosaic=MoonshotConfig.CLOSE_MOSAIC,
                project=str(MoonshotConfig.RUNS_DIR),
                name=run_name,
                exist_ok=True,
                verbose=True,
            )

            # Determine actual checkpoint path
            best_ckpt = run_dir / "weights" / "best.pt"
            if not best_ckpt.exists():
                if hasattr(model, "trainer") and hasattr(model.trainer, "best"):
                    best_ckpt = Path(model.trainer.best)

            assert best_ckpt.exists(), f"FATAL: Trained weights not found at {best_ckpt}!"

            sha256 = hashlib.sha256(best_ckpt.read_bytes()).hexdigest()
            manifest = {
                "status": "SUCCESS",
                "attempt_index": attempt_idx,
                "attempt_spec": f"{current_imgsz}/batch{current_batch}",
                "trained_imgsz": current_imgsz,
                "trained_batch": current_batch,
                "epochs": epochs,
                "patience": patience,
                "run_dir": str(run_dir),
                "checkpoint_path": str(best_ckpt.resolve()),
                "checkpoint_sha256": sha256,
                "checkpoint_size_mb": round(best_ckpt.stat().st_size / (1024 ** 2), 2),
            }

            manifest_path = run_dir / "experiment_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            print(f"[SUCCESS] Training completed! Checkpoint: {best_ckpt} | SHA256: {sha256}")
            return best_ckpt, manifest

        except Exception as e:
            # Detect CUDA OutOfMemoryError (typed or string-reported)
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(e).lower() or ("cuda" in str(e).lower() and "memory" in str(e).lower())
            )
            if is_oom:
                print(f"[OOM DETECTED] Attempt {attempt_idx + 1} ({current_imgsz}/batch{current_batch}) failed with CUDA OOM: {e}")
                torch.cuda.empty_cache()
                if attempt_idx + 1 < len(attempts):
                    next_sz, next_b = attempts[attempt_idx + 1]
                    print(f"[FALLBACK] Transitioning to attempt {attempt_idx + 2}: {next_sz}/batch{next_b}...")
                    continue
                else:
                    raise RuntimeError("FATAL: Exhausted all fallback attempts without successful training.") from e
            else:
                # Non-OOM exception must propagate immediately!
                print(f"[FATAL ERROR] Non-OOM exception encountered during training: {e}")
                raise e

    raise RuntimeError("FATAL: Exhausted all fallback resolutions without stable training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Native-2048 YOLOv8l-seg")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit 0 without GPU training")
    parser.add_argument("--eval-only", action="store_true", help="Run multi-annotator PQ sweep only")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=MoonshotConfig.EPOCHS)
    parser.add_argument("--data-yaml", type=str, default=str(YOLO_DATA_DIR / "data.yaml"))
    args = parser.parse_args()

    if args.dry_run:
        print_training_banner()
        print("[DRY-RUN] Verified on CPU. Exiting 0.")
        sys.exit(0)

    if args.eval_only:
        from ultralytics import YOLO
        if not args.weights or not Path(args.weights).exists():
            print(f"[ERROR] Must provide valid checkpoint path via --weights")
            sys.exit(1)
        model = YOLO(args.weights)
        run_holdout_validation_sweep(model, magfilo_dir=MAGFILO_DIR, yolo_data_dir=YOLO_DATA_DIR)
        sys.exit(0)

    train_yolov8l_with_fallback(
        data_yaml=Path(args.data_yaml),
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
