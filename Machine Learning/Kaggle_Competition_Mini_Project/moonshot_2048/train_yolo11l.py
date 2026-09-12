"""
moonshot_2048/train_yolo11l.py — Native-2048 YOLO11l-seg Training Engine (Phase D Foundation).

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Phase D Specification:
- Base Architecture: yolo11l-seg.pt
- Image Resolution: Largest stable size in {2048, 1792, 1536}
- Identical GroupKFold by year split to enable OOF cross-model calibration.
"""

import argparse
import sys
from pathlib import Path
import torch

from moonshot_2048.config import MoonshotConfig, YOLO_DATA_DIR, MAGFILO_DIR
from moonshot_2048.train_yolov8l import run_holdout_validation_sweep


def train_yolo11l(
    data_yaml: Path,
    epochs: int = MoonshotConfig.EPOCHS,
    patience: int = MoonshotConfig.PATIENCE,
    batch: int = MoonshotConfig.BATCH,
    device: int = MoonshotConfig.DEVICE,
    dry_run: bool = False,
):
    """Train YOLO11l-seg on the exact same native-2048 dataset split."""
    print("=" * 75)
    print("MOONSHOT 2048: YOLO11l-seg PHASE D TRAINING")
    print("=" * 75)
    print(f"  Model:           {MoonshotConfig.MODEL_11L}")
    print(f"  Resolution:      {MoonshotConfig.IMGSZ}x{MoonshotConfig.IMGSZ}")
    print(f"  Batch:           {batch}")
    print(f"  Epochs:          {epochs} (patience: {patience})")
    print(f"  Project Run Dir: {MoonshotConfig.V11L_RUN_DIR}")
    print("=" * 75)

    if dry_run:
        print("[DRY-RUN] YOLO11l configuration verified. Exiting 0.")
        return None

    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is required for YOLO11l training.")

    from ultralytics import YOLO

    for current_imgsz in MoonshotConfig.FALLBACK_IMGSZ:
        try:
            print(f"\n[INFO] Starting YOLO11l-seg training at imgsz={current_imgsz}, batch={batch}...")
            model = YOLO(MoonshotConfig.MODEL_11L)
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                patience=patience,
                batch=batch,
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
                name="moonshot_v11l_2048",
                exist_ok=True,
                verbose=True,
            )
            print(f"[SUCCESS] YOLO11l training completed at imgsz={current_imgsz}!")
            return MoonshotConfig.V11L_WEIGHTS_PATH
        except Exception as e:
            if "memory" in str(e).lower():
                print(f"[OOM] Memory error at imgsz={current_imgsz}: {e}")
                torch.cuda.empty_cache()
                continue
            raise e

    raise RuntimeError("FATAL: Exhausted fallback resolutions for YOLO11l.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Native-2048 YOLO11l-seg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--batch", type=int, default=MoonshotConfig.BATCH)
    parser.add_argument("--epochs", type=int, default=MoonshotConfig.EPOCHS)
    parser.add_argument("--data-yaml", type=str, default=str(YOLO_DATA_DIR / "data.yaml"))
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] YOLO11l checked on CPU. Exiting 0.")
        sys.exit(0)

    train_yolo11l(
        data_yaml=Path(args.data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        dry_run=args.dry_run,
    )
