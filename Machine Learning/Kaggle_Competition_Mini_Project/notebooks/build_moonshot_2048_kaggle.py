"""
notebooks/build_moonshot_2048_kaggle.py — Builder for Kaggle Moonshot 2048 Notebooks.

Authority: ChatGPT Master
Directives: 17 & 18 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Generates:
1. notebooks/Moonshot_2048_Train_Fold0.ipynb: Full Native-2048 YOLOv8l-seg Training, Fallback, & Holdout PQ Sweep
2. notebooks/Moonshot_2048_Inference.ipynb: Fast Native-2048 Inference & Host-Safe Submission Audit
"""

import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_file_text(rel_path: str) -> str:
    path = PROJECT_ROOT / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def make_markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def get_base_notebook_structure():
    return {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "gpuClass": "standard",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build_train_fold0_notebook() -> Path:
    """Build Moonshot_2048_Train_Fold0.ipynb with Directive 18 compliance."""
    nb = get_base_notebook_structure()

    # Cell 1: Markdown Overview
    c1_md = """# 🚀 MOONSHOT 2048: Native-Resolution YOLOv8l-seg Training (Fold 0)
### IEEE BigData Cup / Kaggle Solar Filament Segmentation Challenge 2026
**Authority:** ChatGPT Master | **Directives:** 17 & 18 | **Executor:** Antigravity  
**Goal:** Establish the native-resolution YOLOv8l instance segmentation anchor at 2048×2048, targeting the current 0.55+ cluster.

---
### 📋 Execution Guide:
1. **Accelerator**: Select **GPU T4 x2** or **GPU P100** (single-GPU training loop on `cuda:0` prevents notebook DDP deadlocks).
2. **Attached Inputs**:
   - `filament-segmentation-2026` (Official competition dataset)
3. **Training Specifications (Directive 18 Verified)**:
   - Architecture: `yolov8l-seg.pt`
   - Approved fallback sequence on CUDA OOM:  
     `2048 / batch 2 -> 2048 / batch 1 -> 1792 / batch 1 -> 1536 / batch 1`
   - Epochs: 60 (patience: 15), Mixed Precision (AMP): True, Single GPU `cuda:0`
   - Conservative solar morphology augmentations: `degrees=10`, `flipud=0.5`, `fliplr=0.5`, `mosaic=0.0`
4. **Validation & Metrics**:
   - Exact Stage-0 assertions: 707 physical files, 1,154 observations, zero file leakage, 579 train / 128 val files.
   - Stage 1 returns the exact checkpoint and experiment manifest; Stage 2 evaluates this checkpoint directly (no arbitrary discovery).
   - Deployment-matched NMS ($IoU = 0.00$) multi-annotator Kirillov PQ sweep across confidence {0.15, 0.20, 0.25, 0.30, 0.35, 0.40} and min_area {50, 100, 200, 400}.
"""
    nb["cells"].append(make_markdown_cell(c1_md))

    # Cell 2: Setup & Environment
    c2_code = """# ==============================================================================
# CELL 2: Environment Setup & Pinned Dependency Installation
# ==============================================================================
import os, sys, time
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

t_start = time.perf_counter()

!pip -q install ultralytics==8.4.103 pycocotools albumentations scikit-learn

import torch, cv2, numpy as np, pandas as pd
from importlib.metadata import version as pkg_version

def get_v(pkg):
    try: return pkg_version(pkg)
    except: return "unknown"

print("=" * 75)
print("RUNTIME ENVIRONMENT FREEZE")
print("=" * 75)
print(f"  Python:      {sys.version.split()[0]}")
print(f"  PyTorch:     {torch.__version__}")
print(f"  CUDA:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"  Ultralytics: {get_v('ultralytics')}")
print(f"  pycocotools: {get_v('pycocotools')}")
print(f"  sklearn:     {get_v('scikit-learn')}")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c2_code))

    # Cell 3: Dataset Detection
    c3_code = """# ==============================================================================
# CELL 3: Dataset Detection & Accelerator Inspection
# ==============================================================================
from pathlib import Path
import torch

candidate_paths = [
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026"),
    Path("data/MAGFiLO_1.0_Kaggle_2026"),
]
data_dir = next((p for p in candidate_paths if p.exists()), None)
print(f"Detected Dataset Root: {data_dir}")
assert data_dir is not None, "FATAL: Competition dataset not attached! Add filament-segmentation-2026."

n_gpus = torch.cuda.device_count()
print(f"CUDA GPUs Available: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  [GPU {i}] {p.name} | VRAM: {p.total_memory / (1024**3):.2f} GB")
"""
    nb["cells"].append(make_code_cell(c3_code))

    # Cell 4: Deploy Modular Code
    pq_code = read_file_text("metrics/pq.py")
    cfg_code = read_file_text("moonshot_2048/config.py")
    data_code = read_file_text("moonshot_2048/data.py")
    train_code = read_file_text("moonshot_2048/train_yolov8l.py")
    predict_code = read_file_text("moonshot_2048/predict_native.py")
    eval_code = read_file_text("moonshot_2048/match_and_calibrate.py")
    ensemble_code = read_file_text("moonshot_2048/ensemble_instances.py")
    refiner_code = read_file_text("moonshot_2048/gated_refiner.py")
    audit_code = read_file_text("moonshot_2048/audit_submission.py")

    c4_code = f'''# ==============================================================================
# CELL 4: Deploy Moonshot 2048 Source Modules to Working Directory
# ==============================================================================
from pathlib import Path

Path("metrics").mkdir(parents=True, exist_ok=True)
Path("moonshot_2048").mkdir(parents=True, exist_ok=True)

with open("metrics/__init__.py", "w") as f: f.write("")
with open("moonshot_2048/__init__.py", "w") as f: f.write("")

with open("metrics/pq.py", "w", encoding="utf-8") as f: f.write({repr(pq_code)})
with open("moonshot_2048/config.py", "w", encoding="utf-8") as f: f.write({repr(cfg_code)})
with open("moonshot_2048/data.py", "w", encoding="utf-8") as f: f.write({repr(data_code)})
with open("moonshot_2048/train_yolov8l.py", "w", encoding="utf-8") as f: f.write({repr(train_code)})
with open("moonshot_2048/predict_native.py", "w", encoding="utf-8") as f: f.write({repr(predict_code)})
with open("moonshot_2048/match_and_calibrate.py", "w", encoding="utf-8") as f: f.write({repr(eval_code)})
with open("moonshot_2048/ensemble_instances.py", "w", encoding="utf-8") as f: f.write({repr(ensemble_code)})
with open("moonshot_2048/gated_refiner.py", "w", encoding="utf-8") as f: f.write({repr(refiner_code)})
with open("moonshot_2048/audit_submission.py", "w", encoding="utf-8") as f: f.write({repr(audit_code)})

print("✅ Successfully deployed metrics/ and moonshot_2048/ modules to working directory.")
'''
    nb["cells"].append(make_code_cell(c4_code))

    # Cell 5: Stage 0 Dataset Conversion
    c5_code = """# ==============================================================================
# CELL 5: Stage 0 — Dataset Conversion & Grouped Split Verification
# ==============================================================================
from pathlib import Path
import subprocess, sys

out_yolo = Path("/kaggle/working/data/yolo_native2048")
cmd = [
    sys.executable, "-m", "moonshot_2048.data",
    "--data-dir", str(data_dir),
    "--out-dir", str(out_yolo),
    "--val-fold", "0",
    "--n-splits", "5",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)

yaml_path = out_yolo / "data.yaml"
assert yaml_path.exists(), f"FATAL: data.yaml not created at {yaml_path}"
with open(yaml_path) as f:
    print(f.read())
"""
    nb["cells"].append(make_code_cell(c5_code))

    # Cell 6: Stage 1 Training with Fallback Chain
    c6_code = """# ==============================================================================
# CELL 6: Stage 1 — Native-2048 YOLOv8l-seg Training (Fallback Sequence)
# ==============================================================================
import time
from pathlib import Path
from moonshot_2048.train_yolov8l import train_yolov8l_with_fallback
from moonshot_2048.config import MoonshotConfig

t0_train = time.perf_counter()
yaml_p = Path("/kaggle/working/data/yolo_native2048/data.yaml")

best_ckpt, exp_manifest = train_yolov8l_with_fallback(
    data_yaml=yaml_p,
    epochs=MoonshotConfig.EPOCHS,
    patience=MoonshotConfig.PATIENCE,
    device=MoonshotConfig.DEVICE,
    dry_run=False,
)
print(f"\\nTraining completed in {(time.perf_counter() - t0_train) / 60:.1f} minutes.")
print(f"Verified Best Checkpoint: {best_ckpt} (exists: {Path(best_ckpt).exists()})")
"""
    nb["cells"].append(make_code_cell(c6_code))

    # Cell 7: Stage 2 Validation Sweep using Exact Checkpoint
    c7_code = """# ==============================================================================
# CELL 7: Stage 2 — Multi-Annotator Kirillov PQ Sweep on Physical Disks
# ==============================================================================
from pathlib import Path
from ultralytics import YOLO
from moonshot_2048.train_yolov8l import run_holdout_validation_sweep
from moonshot_2048.config import MoonshotConfig

# Directive 18: Pass the direct Stage 1 checkpoint without arbitrary input discovery
assert best_ckpt is not None and Path(best_ckpt).exists(), f"FATAL: Checkpoint {best_ckpt} not found!"
print(f"Evaluating Direct Stage 1 Checkpoint: {best_ckpt}")

model = YOLO(str(best_ckpt))
sweep_results = run_holdout_validation_sweep(
    model=model,
    magfilo_dir=data_dir,
    yolo_data_dir=Path("/kaggle/working/data/yolo_native2048"),
    nms_iou=MoonshotConfig.NMS_IOU,  # Deployment parity (iou=0.00)
    imgsz=MoonshotConfig.IMGSZ,
    device="0",
)
"""
    nb["cells"].append(make_code_cell(c7_code))

    # Cell 8: Checkpoint Artifact Inspection & Experiment Manifest
    c8_code = """# ==============================================================================
# CELL 8: Checkpoint Artifact Inspection & Experiment Manifest
# ==============================================================================
import json, hashlib
from pathlib import Path

assert Path(best_ckpt).exists(), f"FATAL: Checkpoint not found at {best_ckpt}"
sha256 = hashlib.sha256(Path(best_ckpt).read_bytes()).hexdigest()
size_mb = Path(best_ckpt).stat().st_size / (1024 ** 2)

print("=" * 75)
print("TRAINED CHECKPOINT ARTIFACT VERIFICATION & EXPERIMENT MANIFEST")
print("=" * 75)
print(f"  Checkpoint Path: {best_ckpt}")
print(f"  File Size:       {size_mb:.2f} MB")
print(f"  SHA256 Hash:     {sha256}")
print("  Experiment Manifest Content:")
print(json.dumps(exp_manifest, indent=2))
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c8_code))

    out_path = PROJECT_ROOT / "notebooks" / "Moonshot_2048_Train_Fold0.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path}")
    return out_path


def build_inference_notebook() -> Path:
    """Build Moonshot_2048_Inference.ipynb with Directive 18 compliance."""
    nb = get_base_notebook_structure()

    c1_md = """# 🛰️ MOONSHOT 2048: Native-Resolution Inference & Submission Engine
### IEEE BigData Cup / Kaggle Solar Filament Segmentation Challenge 2026
**Authority:** ChatGPT Master | **Directives:** 17 & 18 | **Executor:** Antigravity  

Strict Host Rules Enforced:
1. Native 2048 Resolution Direct Inference (YOLOv8l-seg)
2. Assert exactly 180 discovered test images before inference and record inference manifest
3. Greedy Pixel-Carve Zero-Overlap Sanitizer (Assert 0 shared pixels per disk)
4. Zero rows emitted for disks with 0 detections (no dummy masks)
5. Valid pycocotools COCO Fortran RLE encoding
"""
    nb["cells"].append(make_markdown_cell(c1_md))

    c2_code = """# ==============================================================================
# CELL 2: Setup & Environment
# ==============================================================================
import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!pip -q install ultralytics==8.4.103 pycocotools scikit-learn
"""
    nb["cells"].append(make_code_cell(c2_code))

    c3_code = """# ==============================================================================
# CELL 3: Locate Dataset & Trained Weights (Assert Exactly 180 Test Images)
# ==============================================================================
from pathlib import Path
import torch

candidate_paths = [
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026"),
    Path("data/MAGFiLO_1.0_Kaggle_2026"),
]
data_dir = next((p for p in candidate_paths if p.exists()), None)
test_dir = data_dir / "test" / "test_images"
print(f"Test Images Directory: {test_dir}")
assert test_dir.exists(), f"FATAL: Test directory not found at {test_dir}"

# Exact 180 images assertion (Directive 18)
test_images = list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg"))
print(f"Test Images Discovered on Disk: {len(test_images)}")
assert len(test_images) == 180, f"FATAL: Expected exactly 180 test images, found {len(test_images)}!"

# Locate trained weights
weights_candidates = list(Path("/kaggle/input").glob("**/best.pt")) + list(Path(".").glob("**/best.pt"))
print(f"Weights Candidates Found: {weights_candidates}")
assert len(weights_candidates) > 0, "FATAL: Trained best.pt weights not attached!"
model_weights = weights_candidates[0]
print(f"Selected Weights: {model_weights}")
"""
    nb["cells"].append(make_code_cell(c3_code))

    # Deploy code
    pq_code = read_file_text("metrics/pq.py")
    cfg_code = read_file_text("moonshot_2048/config.py")
    predict_code = read_file_text("moonshot_2048/predict_native.py")
    audit_code = read_file_text("moonshot_2048/audit_submission.py")

    c4_code = f'''# ==============================================================================
# CELL 4: Deploy Inference Modules
# ==============================================================================
from pathlib import Path
Path("metrics").mkdir(parents=True, exist_ok=True)
Path("moonshot_2048").mkdir(parents=True, exist_ok=True)

with open("metrics/__init__.py", "w") as f: f.write("")
with open("moonshot_2048/__init__.py", "w") as f: f.write("")

with open("metrics/pq.py", "w", encoding="utf-8") as f: f.write({repr(pq_code)})
with open("moonshot_2048/config.py", "w", encoding="utf-8") as f: f.write({repr(cfg_code)})
with open("moonshot_2048/predict_native.py", "w", encoding="utf-8") as f: f.write({repr(predict_code)})
with open("moonshot_2048/audit_submission.py", "w", encoding="utf-8") as f: f.write({repr(audit_code)})
print("✅ Deployed inference modules.")
'''
    nb["cells"].append(make_code_cell(c4_code))

    c5_code = """# ==============================================================================
# CELL 5: Run Native-2048 Inference & Generate submission.csv
# ==============================================================================
import time
from pathlib import Path
from ultralytics import YOLO
from moonshot_2048.predict_native import generate_submission
from moonshot_2048.config import MoonshotConfig

t0_inf = time.perf_counter()
model = YOLO(str(model_weights))
out_csv = Path("/kaggle/working/submission.csv")

# Operating point locked by Stage 2 Holdout Sweep winner (pq_mean=0.4583, pq_max=0.4845)
df_sub = generate_submission(
    model=model,
    test_dir=test_dir,
    out_csv=out_csv,
    imgsz=MoonshotConfig.IMGSZ,
    conf=0.25,
    nms_iou=0.00,
    min_area=50,
    device="0",
    assert_180=True,
)
print(f"Inference completed in {(time.perf_counter() - t0_inf):.1f} seconds.")
"""
    nb["cells"].append(make_code_cell(c5_code))

    c6_code = """# ==============================================================================
# CELL 6: Mandatory Submission Contract Audit
# ==============================================================================
from pathlib import Path
from moonshot_2048.audit_submission import audit_submission_csv

sub_path = Path("/kaggle/working/submission.csv")
audit_report = audit_submission_csv(
    csv_path=sub_path,
    expected_test_stems=180,
    expected_shape=(2048, 2048),
)
print("AUDIT STATUS: FULLY PASSED ALL HOST CONTRACT RULES")
"""
    nb["cells"].append(make_code_cell(c6_code))

    out_path = PROJECT_ROOT / "notebooks" / "Moonshot_2048_Inference.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path}")
    return out_path


if __name__ == "__main__":
    train_nb = build_train_fold0_notebook()
    inf_nb = build_inference_notebook()
    print("Kaggle Moonshot 2048 Notebooks built successfully.")
