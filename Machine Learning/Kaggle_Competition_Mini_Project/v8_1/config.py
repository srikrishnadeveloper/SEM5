"""
v8_1 Configuration — Unified Frozen Hyperparameters for Seed -> Refine Cascade.

Master: ChatGPT | Builder: Antigravity
Target: Kaggle Solar Filament Segmentation Challenge 2026
"""

from pathlib import Path
import os
import torch

# Base directories & environment detection
KAGGLE = Path("/kaggle/input").exists()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = Path("/kaggle/working") if KAGGLE else PROJECT_ROOT

CANDIDATE_DATA_DIRS = [
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026"),
    PROJECT_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026",
]

MAGFILO_DIR = next((p for p in CANDIDATE_DATA_DIRS if p.exists()), PROJECT_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026")
YOLO_DATA_DIR = OUT / "data" / "yolo_seg"
RUNS_DIR = OUT / "runs"
MODELS_DIR = OUT / "models" / "v8_1"
SUBMISSIONS_DIR = OUT if KAGGLE else (PROJECT_ROOT / "submissions")

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


class YOLOConfig:
    """Frozen YOLO11s-seg Stage 1 Proposer Specification."""
    MODEL = "yolo11s-seg.pt"
    IMGSZ = 1024
    EPOCHS = 20
    BATCH = 2  # Safe batch size on T4 at 1024x1024
    DEVICE = 0  # Single GPU for training to prevent notebook subprocess DDP hang
    AMP = True
    MOSAIC = 0.5
    CLOSE_MOSAIC = 2
    DEGREES = 10.0
    FLIPUD = 0.5
    FLIPLR = 0.5
    OVERLAP_MASK = True  # Pinned Ultralytics default for segmentation mask loss
    MASK_RATIO = 4       # Pinned Ultralytics default downsample ratio
    WORKERS = 2 if os.name != "nt" else 0  # 2 on Linux/Kaggle, 0 on Windows
    PROJECT = str(RUNS_DIR)
    NAME = "v8_1_yolo_s1024"
    DATA_YAML = str(YOLO_DATA_DIR / "data.yaml")
    WEIGHTS_PATH = RUNS_DIR / "v8_1_yolo_s1024" / "weights" / "best.pt"

    @classmethod
    def resolve_weights(cls, explicit_path=None) -> Path:
        """Dynamically resolve best.pt path across input datasets and runs."""
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path)
        if cls.WEIGHTS_PATH.exists():
            return cls.WEIGHTS_PATH
        # Search /kaggle/input for any best.pt
        if Path("/kaggle/input").exists():
            input_candidates = sorted(list(Path("/kaggle/input").glob("**/best.pt")), key=lambda p: p.stat().st_mtime)
            if input_candidates:
                return input_candidates[-1]
        # Search RUNS_DIR for any best.pt
        candidates = sorted(list(RUNS_DIR.glob("**/weights/best.pt")), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        return cls.WEIGHTS_PATH


class CropRefinerConfig:
    """Frozen Stage 2 Crop U-Net Specification."""
    ENCODER = "resnet34"
    IN_CHANNELS = 3  # [Raw, CLAHE, High-Pass Unsharp]
    CLASSES = 1
    CROP_MIN = 256
    CROP_MAX = 512
    CROP_PADDING = 1.2
    BATCH_SIZE = 8
    TARGET_SIZE = 384  # Square resized input size for refiner batch
    EPOCHS = 20
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    AMP = True
    CKPT_PATH = MODELS_DIR / "crop_refiner_r34.pth"

    @classmethod
    def resolve_ckpt(cls, explicit_path=None) -> Path:
        """Dynamically resolve crop_refiner_r34.pth across input datasets and models."""
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path)
        if cls.CKPT_PATH.exists():
            return cls.CKPT_PATH
        if Path("/kaggle/input").exists():
            input_candidates = sorted(list(Path("/kaggle/input").glob("**/crop_refiner_r34.pth")), key=lambda p: p.stat().st_mtime)
            if input_candidates:
                return input_candidates[-1]
        candidates = sorted(list(MODELS_DIR.glob("**/crop_refiner_r34.pth")), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        return cls.CKPT_PATH


class CascadeConfig:
    """Frozen Stage 3 Instance Cascade Inference Specification (Directive R3)."""
    RESIDUAL_DEFAULT = 0  # Off by default for Baseline 1
    RESIDUAL_IOU_THRESH = 0.30  # Keep residual only if IoU < 0.30 vs all YOLO instances
    DEFAULT_CONF = 0.20
    DEFAULT_MIN_AREA = 400
    DEFAULT_OVERLAP_MODE = "trim"
    DEFAULT_FALLBACK = 0
    SOLAR_DISK_R_FRAC = 0.93  # Geometric radius clipping
    TTA_4FLIP = True  # 4-flip TTA on crops
    SUBMISSION_PATH = (OUT / "submission.csv") if KAGGLE else (SUBMISSIONS_DIR / "submission.csv")
