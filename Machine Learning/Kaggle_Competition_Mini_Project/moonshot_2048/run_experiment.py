"""
moonshot_2048/run_experiment.py — Unified CLI Orchestrator for Moonshot 2048.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moonshot_2048.config import MoonshotConfig, MAGFILO_DIR, YOLO_DATA_DIR, SUBMISSIONS_DIR
from moonshot_2048.data import convert_dataset
from moonshot_2048.train_yolov8l import train_yolov8l_with_fallback, run_holdout_validation_sweep
from moonshot_2048.predict_native import generate_submission
from moonshot_2048.audit_submission import audit_submission_csv


def main():
    parser = argparse.ArgumentParser(description="Moonshot 2048 Pipeline Orchestrator")
    parser.add_argument("--stage", choices=["data", "train", "sweep", "infer", "audit", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Execute CPU dry-run without long training")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=MoonshotConfig.EPOCHS)
    parser.add_argument("--batch", type=int, default=MoonshotConfig.BATCH)
    parser.add_argument("--imgsz", type=int, default=MoonshotConfig.IMGSZ)
    args = parser.parse_args()

    print("=" * 75)
    print(f"MOONSHOT 2048 PIPELINE — STAGE: {args.stage.upper()} (dry-run: {args.dry_run})")
    print("=" * 75)

    # 1. Dataset Stage
    if args.stage in ["data", "all"]:
        print("\n>>> STAGE 1: DATASET CONVERSION & GROUPED SPLIT")
        if not args.dry_run:
            convert_dataset(
                data_dir=MAGFILO_DIR,
                out_dir=YOLO_DATA_DIR,
                val_fold=args.val_fold,
            )
        else:
            print("  [DRY-RUN] Dataset conversion skipped; configuration verified.")

    # 2. Train Stage
    best_ckpt = None
    if args.stage in ["train", "all"]:
        print("\n>>> STAGE 2: YOLOv8l-seg NATIVE-2048 TRAINING")
        yaml_p = YOLO_DATA_DIR / "data.yaml"
        best_ckpt, exp_manifest = train_yolov8l_with_fallback(
            data_yaml=yaml_p,
            epochs=args.epochs,
            dry_run=args.dry_run,
        )

    # 3. Sweep Stage
    if args.stage in ["sweep", "all"]:
        print("\n>>> STAGE 3: HOLDOUT VALIDATION PQ SWEEP")
        if args.dry_run:
            print("  [DRY-RUN] Validation sweep skipped; configuration verified.")
        else:
            from ultralytics import YOLO
            target_weights = best_ckpt if (best_ckpt and Path(best_ckpt).exists()) else (Path(args.weights) if args.weights else None)
            if target_weights and target_weights.exists():
                model = YOLO(str(target_weights))
                run_holdout_validation_sweep(model, magfilo_dir=MAGFILO_DIR, yolo_data_dir=YOLO_DATA_DIR, imgsz=args.imgsz)
            else:
                print(f"  [SKIP] Checkpoint not found at {target_weights}")

    # 4. Inference Stage
    if args.stage in ["infer", "all"]:
        print("\n>>> STAGE 4: NATIVE TEST INFERENCE & SUBMISSION BUILD")
        if args.dry_run:
            print("  [DRY-RUN] Test inference skipped; configuration verified.")
        else:
            from ultralytics import YOLO
            w_path = MoonshotConfig.resolve_v8l_weights(args.weights)
            test_img_dir = MAGFILO_DIR / "test" / "test_images"
            if w_path.exists() and test_img_dir.exists():
                model = YOLO(str(w_path))
                generate_submission(
                    model=model,
                    test_dir=test_img_dir,
                    out_csv=MoonshotConfig.SUBMISSION_PATH,
                    imgsz=args.imgsz,
                )
            else:
                print(f"  [SKIP] Checkpoint or test images directory not found.")

    # 5. Audit Stage
    if args.stage in ["audit", "all"]:
        print("\n>>> STAGE 5: SUBMISSION CONTRACT AUDIT")
        sub_p = MoonshotConfig.SUBMISSION_PATH
        if sub_p.exists():
            audit_submission_csv(sub_p)
        else:
            print(f"  [INFO] Submission file not present at {sub_p} (run inference stage first).")

    print("\n" + "=" * 75)
    print("MOONSHOT 2048 PIPELINE STAGE COMPLETE.")
    print("=" * 75)


if __name__ == "__main__":
    main()
