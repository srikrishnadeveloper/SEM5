"""
Builder script for notebooks/V8_1_Kaggle_Production.ipynb.

Produces a fully self-contained, valid Kaggle Dual-T4 notebook implementing
the frozen V8.1 True Instance-Segmentation Cascade.
"""

import json
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


def build_notebook():
    nb = {
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

    # Cell 1: Markdown Overview & Execution Guide
    c1_md = """# 🚀 V8.1 Directive R3: Host-Safe Trim Submission (Dual T4)
### Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup)
**Authority**: Grok Master | **Executor**: Antigravity  
**Baseline Anchor**: 0.350 Public LB (1,231 rows across 178 disks)  
**Cascade**: YOLO11s-seg @ 1024 -> Adaptive Square Crop -> Crop U-Net (ResNet-34 @ 384) with 4-flip TTA -> Greedy Pixel-Carve Sanitizer -> COCO RLE

---
### 📋 Execution Guide (Fast Path: ~4–8 minutes):
1. **Inputs Attached**:
   - Competition data: `filament-segmentation-2026`
   - Previous run: `notebooka464bcfe13` (via **+ Add Input -> Your Work**)
2. **Frozen R3 Submit Knobs**:
   - `conf = 0.20`
   - `min_area = 400`
   - `yolo_fallback = 0`
   - `overlap_mode = trim` (host-safe: zero shared pixels guaranteed)
   - `tta = True` (4-flip TTA on crops)
   - Same weights as `notebooka464bcfe13` (`best.pt` & `crop_refiner_r34.pth`)
3. **No Retrain / No 80-Point Sweep**: Stage C test inference only (~3–5 min).
4. **Mandatory Host Sanitizer & Audit Assertion**:
   - Every disk strictly asserted for `sum(mask_i & mask_j) == 0` for all $i \ne j$.
   - Fails the notebook if even a single shared pixel exists.
"""
    nb["cells"].append(make_markdown_cell(c1_md))

    # Cell 2: Install Block & Safe Version Logging
    c2_code = """# ==============================================================================
# CELL 2: Environment Setup & Pinned Dependency Freeze
# Purpose: Configure PyTorch memory allocator, install frozen production packages,
#          and log runtime package versions for complete reproducibility.
# Runtime: ~30 seconds
# ==============================================================================
import os, sys, time
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

t_notebook_start = time.perf_counter()

# Install pinned production dependencies (Internet ON)
!pip -q install ultralytics==8.4.103 segmentation-models-pytorch pycocotools albumentations timm

# Safe version resolution using importlib.metadata (cannot crash if package lacks __version__)
from importlib.metadata import version as pkg_version
import sys, torch, cv2, numpy as np, pandas as pd

def safe_version(pkg_name):
    try:
        return pkg_version(pkg_name)
    except Exception:
        return "unknown"

print("=" * 75)
print("DEPENDENCY REPRODUCIBILITY FREEZE")
print("=" * 75)
print(f"  Python:                       {sys.version.split()[0]}")
print(f"  PyTorch:                      {torch.__version__}")
print(f"  Ultralytics:                  {safe_version('ultralytics')}")
print(f"  segmentation_models_pytorch:  {safe_version('segmentation-models-pytorch')}")
print(f"  pycocotools:                  {safe_version('pycocotools')}")
print(f"  albumentations:               {safe_version('albumentations')}")
print(f"  timm:                         {safe_version('timm')}")
print(f"  OpenCV (cv2):                 {cv2.__version__}")
print(f"  NumPy:                        {np.__version__}")
print(f"  Pandas:                       {pd.__version__}")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c2_code))

    # Cell 3: Diagnostic & GPU Verification
    c3_code = """# ==============================================================================
# CELL 3: Hardware Diagnostics & Competition Dataset Detection
# Purpose: Verify Dual T4 GPUs, inspect available VRAM, and locate MAGFiLO dataset.
# Runtime: ~2 seconds
# ==============================================================================
import torch
from pathlib import Path

print("=" * 75)
print("SYSTEM & ACCELERATOR DIAGNOSTIC")
print("=" * 75)
n_gpus = torch.cuda.device_count()
print(f"CUDA Available: {torch.cuda.is_available()} | GPU Count: {n_gpus}")

for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"  [GPU {i}] {props.name} | VRAM: {vram_gb:.2f} GB")

assert torch.cuda.is_available(), "FATAL: GPU is required! Switch runtime to GPU T4 x2."

# Detect competition dataset paths in Kaggle input
candidate_paths = [
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026"),
    Path("data/MAGFiLO_1.0_Kaggle_2026"),
]
data_dir = next((p for p in candidate_paths if p.exists()), None)
print(f"Dataset Root: {data_dir}")
assert data_dir is not None, "FATAL: Dataset not found in /kaggle/input! Attach filament-segmentation-2026."
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c3_code))

    # Cell 4: Embed Source Code Modules to /kaggle/working
    pq_src = read_file_text("metrics/pq.py")
    convert_src = read_file_text("scripts/convert_coco_to_yolo.py")
    config_src = read_file_text("v8_1/config.py")
    geometry_src = read_file_text("v8_1/geometry.py")
    yolo_src = read_file_text("v8_1/1_train_yolo.py")
    crop_src = read_file_text("v8_1/2_train_crop_refiner.py")
    infer_src = read_file_text("v8_1/3_infer_cascade.py")
    sweep_src = read_file_text("v8_1/4_sweep_inference.py")

    c4_code = f'''# ==============================================================================
# CELL 4: Modular Code Deployment
# Purpose: Unpack modular production scripts into /kaggle/working directory:
#          - metrics/pq.py: Official Kirillov Panoptic Quality implementation
#          - scripts/convert_coco_to_yolo.py: COCO to YOLO-seg converter with GroupKFold
#          - v8_1/config.py: Frozen hyperparameters (YOLO batch 2, Refiner batch 8)
#          - v8_1/geometry.py: Boundary-safe square crop and resize math
#          - v8_1/1_train_yolo.py: YOLO11s-seg training & multi-annotator PQ sweep
#          - v8_1/2_train_crop_refiner.py: ResNet-34 Crop U-Net refiner training
#          - v8_1/3_infer_cascade.py: Dual-T4 model-parallel cascade inference engine
#          - v8_1/4_sweep_inference.py: Grok Directive R2 diagnostic & grid sweep engine
# Runtime: ~1 second
# ==============================================================================
from pathlib import Path

# Create module directories in /kaggle/working
Path("metrics").mkdir(parents=True, exist_ok=True)
Path("scripts").mkdir(parents=True, exist_ok=True)
Path("v8_1").mkdir(parents=True, exist_ok=True)

with open("metrics/__init__.py", "w") as f: f.write("")
with open("v8_1/__init__.py", "w") as f: f.write("")

with open("metrics/pq.py", "w", encoding="utf-8") as f:
    f.write({repr(pq_src)})

with open("scripts/convert_coco_to_yolo.py", "w", encoding="utf-8") as f:
    f.write({repr(convert_src)})

with open("v8_1/config.py", "w", encoding="utf-8") as f:
    f.write({repr(config_src)})

with open("v8_1/geometry.py", "w", encoding="utf-8") as f:
    f.write({repr(geometry_src)})

with open("v8_1/1_train_yolo.py", "w", encoding="utf-8") as f:
    f.write({repr(yolo_src)})

with open("v8_1/2_train_crop_refiner.py", "w", encoding="utf-8") as f:
    f.write({repr(crop_src)})

with open("v8_1/3_infer_cascade.py", "w", encoding="utf-8") as f:
    f.write({repr(infer_src)})

with open("v8_1/4_sweep_inference.py", "w", encoding="utf-8") as f:
    f.write({repr(sweep_src)})

print("✅ Successfully deployed metrics/, scripts/, and v8_1/ modules into working directory.")
'''
    nb["cells"].append(make_code_cell(c4_code))

    # Cell 5: Stage 0 Dataset Conversion (Python subprocess, no bash interpolation)
    c5_code = """# ==============================================================================
# CELL 5: Stage 0 — Dataset Conversion & Split Verification
# Purpose: Convert MAGFiLO COCO annotations to YOLO-seg polygon format.
#          - Enforces GroupKFold by year prefix (0 physical filename leakage)
#          - Produces 579 train images and 128 validation images
#          - Verifies data.yaml and generates conversion_report.json
# Runtime: ~1-2 minutes
# ==============================================================================
import time
t0_stage0 = time.perf_counter()
from pathlib import Path
import subprocess, sys

out = Path("/kaggle/working/data/yolo_seg")
cmd = [sys.executable, "scripts/convert_coco_to_yolo.py",
       "--data-dir", str(data_dir),
       "--out-dir", str(out)]
print("CONVERT:", cmd, flush=True)
subprocess.check_call(cmd)

# Verify conversion results
yaml_path = out / "data.yaml"
assert yaml_path.exists(), f"FATAL: data.yaml not found at {yaml_path}!"
print("\\n--- data.yaml content ---")
with open(yaml_path) as f:
    print(f.read())

n_trn = len(list((out / "images" / "train").glob("*.jpeg")))
n_val = len(list((out / "images" / "val").glob("*.jpeg")))
print(f"Verified YOLO dataset images: {n_trn} train images, {n_val} val images")
assert n_trn > 0 and n_val > 0, f"FATAL: Empty dataset! train={n_trn}, val={n_val}"

# Report annotator-level samples vs unique physical JPEGs
import json
report_path = out / "conversion_report.json"
if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as rf:
        rep = json.load(rf)
    print(f"Annotator-level samples: {rep.get('train_samples_count')} train, {rep.get('val_samples_count')} val")
    print(f"Unique physical JPEGs:   {rep.get('train_unique_jpegs')} train, {rep.get('val_unique_jpegs')} val")

t_stage0 = time.perf_counter() - t0_stage0
print(f"\\n[TIMING] Stage 0 (Dataset Conversion): {t_stage0:.2f}s ({t_stage0 / 60:.2f} min)")
"""
    nb["cells"].append(make_code_cell(c5_code))

    # Cell 6: Stage 1 YOLO11s-seg Training & Weights Resolution
    c6_code = """# ==============================================================================
# CELL 6: Stage 1 — YOLO11s-seg Proposer Training & Weights Resolution
# Purpose: Check if pretrained weights exist. If present, skip training!
#          If missing, train YOLO11s-seg (batch 2, 20 epochs, cuda:0).
# Runtime: ~10 seconds if weights present, ~20-25 minutes if training
# ==============================================================================
import time
t0_stage1 = time.perf_counter()
from pathlib import Path
from v8_1.config import YOLOConfig

existing_yolo = YOLOConfig.resolve_weights()
if existing_yolo.exists() and "weights/best.pt" in str(existing_yolo):
    print(f"✅ Existing Stage 1 YOLO weights detected at: {existing_yolo}")
    print("   Skipping Stage 1 training to preserve compute for R2 inference sweep.")
else:
    print("[INFO] Stage 1 YOLO weights not found. Launching training from scratch...")
    !python v8_1/1_train_yolo.py --batch 2 --epochs 20

t_stage1 = time.perf_counter() - t0_stage1
print(f"\\n[TIMING] Stage 1: {t_stage1:.2f}s ({t_stage1 / 60:.2f} min)")
"""
    nb["cells"].append(make_code_cell(c6_code))

    # Cell 7: Stage 2 Crop U-Net Refiner Training & Weights Resolution
    c7_code = """# ==============================================================================
# CELL 7: Stage 2 — Crop U-Net Refiner Training & Weights Resolution
# Purpose: Check if pretrained refiner weights exist. If present, skip training!
#          If missing, train Crop Refiner U-Net (batch 8, 20 epochs, cuda:0).
# Runtime: ~10 seconds if weights present, ~15-20 minutes if training
# ==============================================================================
import time
t0_stage2 = time.perf_counter()
from pathlib import Path
from v8_1.config import CropRefinerConfig

existing_refiner = CropRefinerConfig.resolve_ckpt()
if existing_refiner.exists() and "crop_refiner_r34.pth" in str(existing_refiner):
    print(f"✅ Existing Stage 2 Refiner checkpoint detected at: {existing_refiner}")
    print("   Skipping Stage 2 training to preserve compute for R2 inference sweep.")
else:
    print("[INFO] Stage 2 Refiner weights not found. Launching training from scratch...")
    !python v8_1/2_train_crop_refiner.py --boxes-from-gt

t_stage2 = time.perf_counter() - t0_stage2
print(f"\\n[TIMING] Stage 2: {t_stage2:.2f}s ({t_stage2 / 60:.2f} min)")
"""
    nb["cells"].append(make_code_cell(c7_code))

    # Cell 8: Stage 3 Grok Directive R3 Host-Safe Trim Test Submission
    c8_code = """# ==============================================================================
# CELL 8: Stage 3 — Grok Directive R3 Host-Safe Trim Test Submission
# Purpose: Generate test submission using frozen R3 knobs:
#          - conf: 0.20
#          - min_area: 400
#          - yolo_fallback: 0
#          - overlap_mode: trim (host-safe, zero shared pixels)
#          - tta: True (4-flip TTA on crops)
#          - Weights: best.pt and crop_refiner_r34.pth from notebooka464bcfe13
# Hardware: Dual T4 Model Parallelism (YOLO on cuda:0, Refiner on cuda:1)
# Runtime: ~3-5 minutes (Stage C only — no retraining, no sweep)
# Output Submission: /kaggle/working/submission.csv
# ==============================================================================
import time
t0_stage3 = time.perf_counter()

from pathlib import Path
from v8_1.config import YOLOConfig, CropRefinerConfig

best_yolo = YOLOConfig.resolve_weights()
best_refiner = CropRefinerConfig.resolve_ckpt()

print(f"R3 Proposer Weights: {best_yolo}")
print(f"R3 Refiner Weights:  {best_refiner}")
assert best_yolo.exists(), f"FATAL: YOLO weights missing at {best_yolo}!"
assert best_refiner.exists(), f"FATAL: Refiner weights missing at {best_refiner}!"

!python v8_1/3_infer_cascade.py \\
    --conf 0.20 \\
    --min-area 400 \\
    --yolo-fallback 0 \\
    --overlap-mode trim \\
    --tta \\
    --weights "{best_yolo}" \\
    --refiner-weights "{best_refiner}" \\
    --out /kaggle/working/submission.csv

t_stage3 = time.perf_counter() - t0_stage3
print(f"\\n[TIMING] Stage 3 R3 Host-Safe Trim Inference: {t_stage3:.2f}s ({t_stage3 / 60:.2f} min)")
"""
    nb["cells"].append(make_code_cell(c8_code))

    # Cell 9: Stage 4 Submission Audit & Host Contract Verification
    c9_code = """# ==============================================================================
# CELL 9: Stage 4 — Final Kaggle Submission Audit & Host Contract Verification
# Purpose: Rigorously audit the generated submission against the official host contract:
#          - Verifies columns: filament_id, segmentation_rle
#          - Verifies all 180 official test image stems are accounted for
#          - Confirms no duplicate filament_id values
#          - Decodes every RLE to ensure shape == (2048, 2048) and sum > 0 (strictly positive area)
#          - MANDATORY HOST CHECK: Asserts ZERO pairwise mask overlap per disk:
#            sum(mask_i & mask_j) == 0 for all i != j
#          - Confirms disks with 0 detections emit 0 rows (no penalized dummy masks)
#          - Prints summary statistics and total notebook wall-clock time
# Runtime: ~30 seconds
# ==============================================================================
import time
t0_stage4 = time.perf_counter()

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from metrics.pq import decode_rle

sub_path = Path("/kaggle/working/submission.csv")
assert sub_path.exists(), "FATAL: submission.csv was not generated!"

df = pd.read_csv(sub_path)
print("=" * 75)
print("FINAL KAGGLE SUBMISSION AUDIT & HOST CONTRACT VERIFICATION")
print("=" * 75)
print(f"File Path:        {sub_path}")
print(f"Columns:          {list(df.columns)}")
assert list(df.columns) == ["filament_id", "segmentation_rle"], "FATAL: Invalid columns!"
assert len(df) > 0, "FATAL: Submission file has 0 rows!"
assert df["filament_id"].is_unique, "FATAL: Duplicate filament_id found!"

# Verify all stems belong to official test set (180 images)
test_dir = data_dir / "test" / "test_images"
official_test_stems = {p.stem for p in (list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg")))}
print(f"Official Test JPEGs on disk: {len(official_test_stems)}")
assert len(official_test_stems) == 180, f"Expected 180 official test JPEGs, found {len(official_test_stems)}"

stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
unknown_stems = set(stems) - official_test_stems
assert len(unknown_stems) == 0, f"FATAL: Unknown test stems in submission: {unknown_stems}"

covered_stems = set(stems)
missing_prediction_stems = official_test_stems - covered_stems

print(f"Total Predicted Rows:     {len(df)}")
print(f"Covered Test Disks:       {len(covered_stems)} / 180 ({len(covered_stems) / 180 * 100:.1f}%)")
print(f"Zero-Prediction Disks:    {len(missing_prediction_stems)} / 180 ({len(missing_prediction_stems) / 180 * 100:.1f}%)")

# Group predictions by disk stem to decode and audit zero-overlap
stem_to_rles = defaultdict(list)
for idx, row in df.iterrows():
    s = row["filament_id"].rsplit("_", 1)[0]
    stem_to_rles[s].append((row["filament_id"], row["segmentation_rle"]))

# Verify every submitted RLE decodes to (2048, 2048) with strictly positive area (> 0)
# and MANDATORY PAIRWISE OVERLAP ASSERTION per disk
print("\\nVerifying RLE decode, area bounds, and ZERO pairwise overlap per disk...")
total_pairwise_checks = 0
for stem, items in stem_to_rles.items():
    masks = []
    for fid, rle in items:
        m = decode_rle(rle, 2048, 2048)
        assert m.shape == (2048, 2048), f"Row {fid} invalid decoded shape: {m.shape}"
        assert m.sum() > 0, f"FATAL: Row {fid} has 0 active pixels! No dummy zero masks permitted."
        masks.append(m)
    
    # Check pairwise overlap on this disk
    n_m = len(masks)
    if n_m > 1:
        occupied = np.zeros((2048, 2048), dtype=np.uint8)
        for k in range(n_m):
            occupied = np.maximum(occupied, masks[k])
        
        for i in range(n_m):
            for j in range(i + 1, n_m):
                overlap = int(np.logical_and(masks[i], masks[j]).sum())
                assert overlap == 0, (
                    f"FATAL: Host overlap violation on disk {stem}! "
                    f"Instances {items[i][0]} and {items[j][0]} share {overlap} pixels! "
                    f"Submissions may not contain overlapping masks."
                )
                total_pairwise_checks += 1
        
        assert sum(int(m.sum()) for m in masks) == int(occupied.sum()), (
            f"FATAL: Disjoint sum check failed on {stem}!"
        )

print(f"✅ OVERLAP ASSERTION PASS: Verified zero shared pixels across all test disks ({total_pairwise_checks} pairwise checks passed)!")

counts_per_disk = stems.value_counts()
print(f"Instances / Covered Disk: Mean={counts_per_disk.mean():.2f} | p50={counts_per_disk.median():.1f} | Min={counts_per_disk.min()} | Max={counts_per_disk.max()}")

print("\\nFirst 10 Rows:")
print(df.head(10))

t_stage4 = time.perf_counter() - t0_stage4
t_total_notebook = time.perf_counter() - t_notebook_start

print("=" * 75)
print(f"[TIMING] Stage 4 Audit:          {t_stage4:.2f}s ({t_stage4 / 60:.2f} min)")
print(f"[TIMING] Total Notebook Time:    {t_total_notebook:.2f}s ({t_total_notebook / 60:.2f} min)")
print("=" * 75)
print("[OK] ALL HOST SUBMISSION CONTRACT CHECKS PASSED (ZERO OVERLAP VERIFIED)!")
"""
    nb["cells"].append(make_code_cell(c9_code))

    out_path = PROJECT_ROOT / "notebooks" / "V8_1_Kaggle_Production.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated valid Kaggle notebook at: {out_path} ({len(nb['cells'])} cells)")

    # Mirror copies to Desktop locations for human upload convenience
    import hashlib, shutil
    h_orig = hashlib.sha256(out_path.read_bytes()).hexdigest()

    targets = [
        Path("C:/Users/srik2/Desktop/Filament_Colab_Run/V8_1_Kaggle_Production.ipynb"),
        Path("C:/Users/srik2/Desktop/V8_1_Kaggle_Production.ipynb"),
    ]
    for tgt in targets:
        if tgt.parent.exists():
            shutil.copy2(out_path, tgt)
            h_tgt = hashlib.sha256(tgt.read_bytes()).hexdigest()
            assert h_tgt == h_orig, f"FATAL: Hash mismatch between {out_path} and {tgt}!"
            print(f"  Mirrored to: {tgt} (SHA256: {h_tgt[:12]}...)")

    return out_path


if __name__ == "__main__":
    build_notebook()
