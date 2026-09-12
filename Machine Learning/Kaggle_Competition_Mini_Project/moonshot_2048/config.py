"""
moonshot_2048/config.py — Frozen Configurations for Native-2048 Moonshot Pipeline.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity
"""

from pathlib import Path
import os
import torch

# Base paths & environment detection
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
YOLO_DATA_DIR = OUT / "data" / "yolo_native2048"
RUNS_DIR = OUT / "runs"
MODELS_DIR = OUT / "models" / "moonshot_2048"
SUBMISSIONS_DIR = OUT if KAGGLE else (PROJECT_ROOT / "submissions")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


class MoonshotConfig:
    """Core Native-2048 Moonshot Hyperparameters (Directive 17 / Directive 18)."""
    PROJECT_ROOT = PROJECT_ROOT
    OUT = OUT
    RUNS_DIR = RUNS_DIR
    MAGFILO_DIR = MAGFILO_DIR
    YOLO_DATA_DIR = YOLO_DATA_DIR

    # Architecture & Model weights
    MODEL_V8L = "yolov8l-seg.pt"
    MODEL_11L = "yolo11l-seg.pt"
    
    # Image resolution: native 2048 with approved fallback attempt chain (Directive 18)
    IMGSZ = 2048
    FALLBACK_IMGSZ = [2048, 1792, 1536]
    FALLBACK_ATTEMPTS = [
        (2048, 2),
        (2048, 1),
        (1792, 1),
        (1536, 1),
    ]
    
    # Training hyperparameters
    EPOCHS = 60
    PATIENCE = 15
    BATCH = 2  # Starting batch for 16 GB Tesla T4
    DEVICE = 0  # Single GPU for training to avoid notebook DDP hang
    AMP = True
    WORKERS = 2 if os.name != "nt" else 0
    SEED = 42
    MAX_DET = 100
    
    # Conservative augmentations tailored for solar chromosphere morphology
    DEGREES = 10.0
    FLIPUD = 0.5
    FLIPLR = 0.5
    MOSAIC = 0.0      # Disabled initially per Directive 17 to preserve native solar geometry
    COPY_PASTE = 0.0  # Disabled initially
    CLOSE_MOSAIC = 0
    
    # Inference defaults (Directive 17 & 0.55 anchor analysis)
    CONF = 0.30
    NMS_IOU = 0.00
    MIN_AREA = 200
    OVERLAP_MODE = "trim"  # Strictly enforced zero-overlap greedy carve
    
    # Geometric disk clipping
    SOLAR_DISK_R_FRAC = 0.93  # Clipping radius fraction (~952 px radius from disk center)
    
    # Output paths
    SUBMISSION_PATH = (OUT / "submission.csv") if KAGGLE else (SUBMISSIONS_DIR / "moonshot_submission.csv")
    V8L_RUN_DIR = RUNS_DIR / "moonshot_v8l_2048"
    V8L_WEIGHTS_PATH = V8L_RUN_DIR / "weights" / "best.pt"
    V11L_RUN_DIR = RUNS_DIR / "moonshot_v11l_2048"
    V11L_WEIGHTS_PATH = V11L_RUN_DIR / "weights" / "best.pt"

    @classmethod
    def resolve_v8l_weights(cls, explicit_path=None) -> Path:
        """Resolve YOLOv8l-seg weights across /kaggle/input and run directories."""
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path)
        if cls.V8L_WEIGHTS_PATH.exists():
            return cls.V8L_WEIGHTS_PATH
        # Scan /kaggle/input for any trained moonshot or yolov8l weights
        if Path("/kaggle/input").exists():
            candidates = sorted(list(Path("/kaggle/input").glob("**/best.pt")), key=lambda p: p.stat().st_mtime)
            if candidates:
                return candidates[-1]
            # Also check for external hdjojo weights if attached
            hd_weights = list(Path("/kaggle/input").glob("**/*yolov8l*.pt"))
            if hd_weights:
                return hd_weights[0]
        return cls.V8L_WEIGHTS_PATH
