"""
notebooks/build_p0_dual_gpu_nb.py
Builder for the High-Speed Dual-GPU P0 Champion Kaggle Notebook.
Uses isolated multi-process GPU execution to eliminate cuDNN thread conflicts.

Authority: ChatGPT Master
Executor: Antigravity
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
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


def build_p0_notebook():
    nb = get_base_notebook_structure()

    # Cell 1: Pure Python Header (safe whether run as Code or Markdown)
    c1_code = """# ==============================================================================
# P0 MOONSHOT CHAMPION: ISOLATED DUAL-GPU SOFT-TTA & SOFT-OWNERSHIP ENGINE
# IEEE BigData Cup / Kaggle Solar Filament Segmentation Challenge 2026
# Authority: ChatGPT Master | Executor: Antigravity
# Base Model: Moonshot 2048 YOLOv8l-seg Champion (best.pt - 0.360 Verified LB)
#
# Core Upgrades Implemented:
# 1. Resampling Leak Eliminated: Continuous 2048x2048 bilinear prototype logits.
# 2. Soft-Overlap Ownership: Argmax continuous probability resolution (0 shared pixels).
# 3. Soft-TTA: 3-view continuous probability fusion (Original + FlipLR + FlipUD).
# 4. Isolated Dual-GPU Parallelism: Separate OS worker processes per GPU (no cuDNN thread locks).
# 5. Topological Cleanup: Retains primary contiguous filament body per MAGFiLO spec.
# 6. Host V6 Compliance: Strict Fortran RLE, 0 dummy rows on empty disks.
# ==============================================================================
print("🚀 Initializing P0 Moonshot Champion Pipeline...")
"""
    nb["cells"].append(make_code_cell(c1_code))

    # Cell 2: Setup & Environment
    c2_code = """# ==============================================================================
# CELL 2: Environment Setup & Dual-GPU Hardware Inspection
# ==============================================================================
import os, sys, time, gc
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Install pinned dependencies
!pip -q install ultralytics==8.4.103 pycocotools scikit-learn

import torch
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

print("=" * 75)
print("🚀 DUAL-GPU HARDWARE TELEMETRY")
print("=" * 75)
print(f"  PyTorch Version: {torch.__version__}")
print(f"  CUDA Available:  {torch.cuda.is_available()}")
n_gpus = torch.cuda.device_count()
print(f"  GPU Count:       {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"    [GPU {i}] {p.name} | VRAM: {p.total_memory / (1024**3):.2f} GB")
print("=" * 75)
assert torch.cuda.is_available(), "FATAL: CUDA accelerator not enabled! Select GPU T4 x2 in Notebook Settings."
"""
    nb["cells"].append(make_code_cell(c2_code))

    # Cell 3: Dataset & Checkpoint Discovery
    c3_code = """# ==============================================================================
# CELL 3: Dataset Detection & Champion Checkpoint Discovery
# ==============================================================================
# 1. Locate competition test images
candidate_data_paths = [
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026"),
    Path("data/MAGFiLO_1.0_Kaggle_2026"),
]
data_dir = next((p for p in candidate_data_paths if p.exists()), None)
assert data_dir is not None, "FATAL: Competition dataset not attached! Attach filament-segmentation-2026."

test_dir = data_dir / "test" / "test_images"
if not test_dir.exists():
    test_dir = data_dir / "test_images"
assert test_dir.exists(), f"FATAL: Test images directory not found at {test_dir}"

test_images = sorted(list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg")))
print(f"Discovered {len(test_images)} test images on disk.")
assert len(test_images) == 180, f"FATAL: Expected exactly 180 test images, found {len(test_images)}!"

# 2. Locate champion weights (best.pt)
weights_candidates = (
    list(Path("/kaggle/input").glob("**/best.pt")) +
    list(Path("/kaggle/working").glob("**/best.pt")) +
    list(Path(".").glob("**/best.pt")) +
    [Path("C:/Users/srik2/Desktop/Filament_Colab_Run/best.pt")]
)
valid_weights = [p for p in weights_candidates if p.exists()]
print(f"Found {len(valid_weights)} candidate weights files.")
assert len(valid_weights) > 0, "FATAL: Trained best.pt weights not attached! Please add the dataset containing best.pt."
model_weights_path = valid_weights[0]
print(f"✅ Loaded Champion Weights: {model_weights_path} ({model_weights_path.stat().st_size / (1024**2):.1f} MB)")
"""
    nb["cells"].append(make_code_cell(c3_code))

    # Cell 4: Deploy infer_worker.py (Self-contained process-safe worker)
    c4_code = """# ==============================================================================
# CELL 4: Deploy Isolated Multi-Process GPU Worker Script (infer_worker.py)
# ==============================================================================
worker_script_path = Path("/kaggle/working/infer_worker.py")

worker_script_content = '''
import os, sys, time, gc, argparse, pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True)
parser.add_argument("--weights", type=str, required=True)
parser.add_argument("--img-list", type=str, required=True)
parser.add_argument("--out-pkl", type=str, required=True)
args = parser.parse_args()

# Pin this OS process exclusively to its assigned GPU
device_str = f"cuda:{args.gpu_id}"
torch.cuda.set_device(args.gpu_id)
print(f"[Worker GPU {args.gpu_id}] Initialized on {device_str}. Loading model...", flush=True)

model = YOLO(args.weights)
model.model.to(device_str).eval()
print(f"[Worker GPU {args.gpu_id}] Model loaded in VRAM: {torch.cuda.memory_allocated(args.gpu_id)/(1024**2):.1f} MB", flush=True)

with open(args.img_list, "r") as f:
    image_paths = [Path(line.strip()) for line in f if line.strip()]

def apply_solar_limb_mask(mask: np.ndarray, r_frac: float = 0.98) -> np.ndarray:
    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    max_r = int((min(w, h) / 2.0) * r_frac)
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    mask[dist_sq > max_r ** 2] = 0
    return mask

def clean_connected_components(mask: np.ndarray, min_fragment_ratio: float = 0.15) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = np.max(areas)
    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    for i, area in enumerate(areas, start=1):
        if area == max_area or area >= (min_fragment_ratio * max_area):
            clean_mask[labels == i] = 1
    return clean_mask

def resolve_soft_overlaps(candidate_probs: List[np.ndarray], confidences: List[float], prob_threshold: float = 0.45, min_area: int = 50):
    if not candidate_probs:
        return [], []
    valid = []
    for p, conf in zip(candidate_probs, confidences):
        p_clipped = apply_solar_limb_mask(p.copy())
        bin_m = (p_clipped > prob_threshold).astype(np.uint8)
        area = int(bin_m.sum())
        if area >= min_area:
            valid.append({"prob": p_clipped, "conf": float(conf), "area": area})
    if not valid:
        return [], []
    valid.sort(key=lambda x: x["conf"], reverse=True)
    n_cands = len(valid)
    if n_cands == 1:
        m = (valid[0]["prob"] > prob_threshold).astype(np.uint8)
        m = clean_connected_components(m)
        if m.sum() >= min_area:
            return [m], [valid[0]["conf"]]
        return [], []
    prob_stack = np.stack([c["prob"] for c in valid], axis=0)
    active_mask = (prob_stack > prob_threshold)
    winner_idx = np.argmax(prob_stack, axis=0)
    final_masks = []
    final_confs = []
    for k in range(n_cands):
        cand_m = np.logical_and(winner_idx == k, active_mask[k]).astype(np.uint8)
        cand_m = clean_connected_components(cand_m)
        if cand_m.sum() >= min_area:
            final_masks.append(cand_m)
            final_confs.append(valid[k]["conf"])
    return final_masks, final_confs

def predict_single_view_soft(img_rgb: np.ndarray, conf_thresh: float = 0.20):
    h_orig, w_orig = img_rgb.shape[:2]
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img_t = img_t.to(device_str)
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            preds = model.model(img_t)
            protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
            det = non_max_suppression(preds[0][0], conf_thres=conf_thresh, iou_thres=0.45, classes=[0], max_det=300, nc=1)[0]
    if det is None or len(det) == 0:
        del img_t, preds
        torch.cuda.empty_cache()
        return [], []
    bboxes = det[:, :4]
    scores = det[:, 4].cpu().numpy().tolist()
    coeffs = det[:, 6:]
    c, mh, mw = protos[0].shape
    logits_512 = (coeffs @ protos[0].float().view(c, -1)).view(-1, mh, mw)
    logits_2048 = F.interpolate(logits_512.unsqueeze(1), size=(h_orig, w_orig), mode="bilinear", align_corners=False).squeeze(1)
    probs_2048 = torch.sigmoid(logits_2048)
    probs_2048 = ops.crop_mask(probs_2048, bboxes)
    prob_maps = [probs_2048[i].cpu().numpy() for i in range(len(probs_2048))]
    del img_t, preds, protos, det, bboxes, coeffs, logits_512, logits_2048, probs_2048
    torch.cuda.empty_cache()
    return prob_maps, scores

def run_soft_tta_disk(img_path: Path, conf_thresh: float = 0.20, prob_thresh: float = 0.45, min_area: int = 50):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return [], []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    p0, s0 = predict_single_view_soft(img_rgb, conf_thresh=conf_thresh)
    img_hflip = cv2.flip(img_rgb, 1)
    p1, s1 = predict_single_view_soft(img_hflip, conf_thresh=conf_thresh)
    p1_unflipped = [cv2.flip(p, 1) for p in p1]
    img_vflip = cv2.flip(img_rgb, 0)
    p2, s2 = predict_single_view_soft(img_vflip, conf_thresh=conf_thresh)
    p2_unflipped = [cv2.flip(p, 0) for p in p2]
    fused_probs = []
    fused_confs = []
    used_v1, used_v2 = set(), set()
    for i, (m0, c0) in enumerate(zip(p0, s0)):
        bin_m0 = (m0 > prob_thresh)
        if bin_m0.sum() < min_area: continue
        matched_v1 = None
        best_iou1 = 0.30
        for j, m1 in enumerate(p1_unflipped):
            if j in used_v1: continue
            bin_m1 = (m1 > prob_thresh)
            inter = np.logical_and(bin_m0, bin_m1).sum()
            union = np.logical_or(bin_m0, bin_m1).sum()
            iou = inter / union if union > 0 else 0
            if iou > best_iou1:
                best_iou1 = iou
                matched_v1 = j
        matched_v2 = None
        best_iou2 = 0.30
        for k, m2 in enumerate(p2_unflipped):
            if k in used_v2: continue
            bin_m2 = (m2 > prob_thresh)
            inter = np.logical_and(bin_m0, bin_m2).sum()
            union = np.logical_or(bin_m0, bin_m2).sum()
            iou = inter / union if union > 0 else 0
            if iou > best_iou2:
                best_iou2 = iou
                matched_v2 = k
        maps = [m0]
        confs = [c0]
        if matched_v1 is not None:
            maps.append(p1_unflipped[matched_v1])
            confs.append(s1[matched_v1])
            used_v1.add(matched_v1)
        if matched_v2 is not None:
            maps.append(p2_unflipped[matched_v2])
            confs.append(s2[matched_v2])
            used_v2.add(matched_v2)
        fused_probs.append(np.mean(maps, axis=0))
        fused_confs.append(float(np.mean(confs)))
    for j, (m1, c1) in enumerate(zip(p1_unflipped, s1)):
        if j not in used_v1 and c1 >= 0.38 and (m1 > prob_thresh).sum() >= min_area:
            fused_probs.append(m1)
            fused_confs.append(c1)
    for k, (m2, c2) in enumerate(zip(p2_unflipped, s2)):
        if k not in used_v2 and c2 >= 0.38 and (m2 > prob_thresh).sum() >= min_area:
            fused_probs.append(m2)
            fused_confs.append(c2)
    clean_masks, clean_confs = resolve_soft_overlaps(fused_probs, fused_confs, prob_threshold=prob_thresh, min_area=min_area)
    return clean_masks, clean_confs

results = []
t_start = time.time()
for idx, img_p in enumerate(image_paths):
    try:
        masks, confs = run_soft_tta_disk(img_p)
        results.append({"stem": img_p.stem, "masks": masks, "confs": confs})
    except Exception as e:
        print(f"[Worker GPU {args.gpu_id}] ❌ Error processing {img_p.name}: {e}", flush=True)
        results.append({"stem": img_p.stem, "masks": [], "confs": []})
    if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
        dt = time.time() - t_start
        print(f"  [GPU {args.gpu_id}] Progress: {idx+1}/{len(image_paths)} disks ({dt:.1f}s | {(idx+1)/dt:.2f} disks/s)", flush=True)

with open(args.out_pkl, "wb") as f:
    pickle.dump(results, f)
print(f"✅ [Worker GPU {args.gpu_id}] Finished {len(results)} disks. Saved to {args.out_pkl}", flush=True)
'''

with open(worker_script_path, "w", encoding="utf-8") as f:
    f.write(worker_script_content)

print(f"✅ Deployed isolated multi-process worker to: {worker_script_path}")
"""
    nb["cells"].append(make_code_cell(c4_code))

    # Cell 5: Launch Parallel Processes (100% cuDNN thread-safe)
    c5_code = """# ==============================================================================
# CELL 5: Launch Isolated Parallel Worker Processes Across Both GPUs
# ==============================================================================
import subprocess

n_gpus = torch.cuda.device_count()
print("=" * 75)
print(f"🚀 LAUNCHING ISOLATED MULTI-PROCESS WORKERS (GPU Count: {n_gpus})")
print("=" * 75)

# Partition image lists
if n_gpus >= 2:
    mid = len(test_images) // 2
    partitions = [
        (0, test_images[:mid], Path("/kaggle/working/imgs_gpu0.txt"), Path("/kaggle/working/chunk_gpu0.pkl")),
        (1, test_images[mid:], Path("/kaggle/working/imgs_gpu1.txt"), Path("/kaggle/working/chunk_gpu1.pkl")),
    ]
else:
    partitions = [
        (0, test_images, Path("/kaggle/working/imgs_gpu0.txt"), Path("/kaggle/working/chunk_gpu0.pkl")),
    ]

# Write image paths to text files
for gpu_id, imgs, txt_p, pkl_p in partitions:
    with open(txt_p, "w") as f:
        for p in imgs:
            f.write(str(p) + "\\n")
    print(f"  GPU {gpu_id} assigned: {len(imgs)} images -> {txt_p.name}")

t_start = time.time()
processes = []

# Launch independent OS processes pinned to each GPU
for gpu_id, imgs, txt_p, pkl_p in partitions:
    cmd = [
        sys.executable, "/kaggle/working/infer_worker.py",
        "--gpu-id", str(gpu_id),
        "--weights", str(model_weights_path),
        "--img-list", str(txt_p),
        "--out-pkl", str(pkl_p),
    ]
    print(f"Launching Worker Process on GPU {gpu_id}...")
    p = subprocess.Popen(cmd)
    processes.append((gpu_id, p))

# Wait for both processes to complete
for gpu_id, p in processes:
    p.wait()
    assert p.returncode == 0, f"FATAL: Worker GPU {gpu_id} exited with error code {p.returncode}!"

total_time = time.time() - t_start
print("=" * 75)
print(f"✅ ALL 180 TEST DISKS COMPLETED IN {total_time:.1f}s ({total_time/60:.2f} min)!")
print(f"   Throughput: {len(test_images) / total_time:.2f} disks/sec (True Dual-GPU Parallel)")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c5_code))

    # Cell 6: Merge Chunks & Export Submissions
    c6_code = """# ==============================================================================
# CELL 6: Merge Worker Predictions & Export Calibration Submissions
# ==============================================================================
import pickle

def encode_coco_rle(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")

# Aggregate results from worker pkl files
all_results = []
for gpu_id, imgs, txt_p, pkl_p in partitions:
    assert pkl_p.exists(), f"FATAL: Missing output file {pkl_p}"
    with open(pkl_p, "rb") as f:
        chunk_data = pickle.load(f)
        all_results.extend(chunk_data)

print(f"Successfully aggregated {len(all_results)} test disk predictions.")
assert len(all_results) == 180, f"Expected 180 results, got {len(all_results)}!"

thresholds = [0.20, 0.25, 0.28, 0.30]
sub_records = {th: [] for th in thresholds}

for res in all_results:
    stem = res["stem"]
    masks = res["masks"]
    confs = res["confs"]

    for th in thresholds:
        k = 1
        for m, c in zip(masks, confs):
            if c >= th and m.sum() >= 50:
                sub_records[th].append({
                    "filament_id": f"{stem}_{k}",
                    "segmentation_rle": encode_coco_rle(m),
                    "conf": c,
                    "area": int(m.sum()),
                })
                k += 1

print("=" * 75)
print("📊 SUBMISSION CALIBRATION SWEEP SUMMARY")
print("=" * 75)

for th in thresholds:
    rows = sub_records[th]
    df = pd.DataFrame(rows)
    if not df.empty:
        df_export = df[["filament_id", "segmentation_rle"]]
        active_disks = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique()
        mean_fil = len(df) / 180.0
        mean_area = df["area"].mean()
    else:
        df_export = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        active_disks, mean_fil, mean_area = 0, 0.0, 0.0

    csv_name = f"submission_conf{th:.2f}.csv"
    csv_path = Path(f"/kaggle/working/{csv_name}")
    df_export.to_csv(csv_path, index=False)
    print(f"  [conf={th:.2f}] Rows: {len(df_export):4d} | Active Disks: {active_disks}/180 | Fil/Disk: {mean_fil:.2f} | Mean Area: {mean_area:.0f} px -> {csv_name}")

primary_df = pd.DataFrame(sub_records[0.25])[["filament_id", "segmentation_rle"]]
primary_path = Path("/kaggle/working/submission.csv")
primary_df.to_csv(primary_path, index=False)
print("=" * 75)
print(f"🏆 PRIMARY SUBMISSION LOCKED: {primary_path} ({len(primary_df)} rows)")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c6_code))

    # Cell 7: Forensic Contract Audit
    c7_code = """# ==============================================================================
# CELL 7: Mandatory Host V6 Forensic Contract Audit
# ==============================================================================
sub_path = Path("/kaggle/working/submission.csv")
assert sub_path.exists(), f"FATAL: {sub_path} does not exist!"

df_check = pd.read_csv(sub_path)
print(f"Checking Submission: {sub_path.name} ({len(df_check)} rows)")

assert list(df_check.columns) == ["filament_id", "segmentation_rle"], "FATAL: Columns mismatch!"

disks_seen = set()
for idx, row in df_check.iterrows():
    fid = row["filament_id"]
    stem = fid.rsplit("_", 1)[0]
    disks_seen.add(stem)
    rle_str = row["segmentation_rle"]
    rle_dict = {"size": [2048, 2048], "counts": rle_str.encode("utf-8")}
    m = mask_utils.decode(rle_dict)
    assert m.shape == (2048, 2048), f"FATAL: Decoded mask shape {m.shape} != (2048, 2048)"
    assert m.sum() > 0, f"FATAL: Empty mask found for {fid}!"

print("✅ RLE Fortran Format: 100% Valid (All decode to 2048x2048 with area > 0)")
print(f"✅ Active Test Disks: {len(disks_seen)}/180")
print(f"✅ Disks with Zero Predictions: {180 - len(disks_seen)} (Emitted zero rows per Host Rule)")
print(f"✅ Mean Filaments per Disk: {len(df_check) / 180.0:.2f}")
print("=" * 75)
print("🎉 ALL HOST CONTRACT RULES FULLY PASSED — READY FOR LEADERBOARD SUBMISSION!")
print("=" * 75)
"""
    nb["cells"].append(make_code_cell(c7_code))

    # Save to notebooks directory and Desktop folder
    target_paths = [
        PROJECT_ROOT / "notebooks" / "P0_Moonshot_Champion_DualGPU.ipynb",
        PROJECT_ROOT / "notebooks" / "kaggle" / "P0_Moonshot_Champion_DualGPU.ipynb",
        Path("C:/Users/srik2/Desktop/Filament_Colab_Run/P0_Moonshot_Champion_DualGPU.ipynb"),
    ]

    for tp in target_paths:
        tp.parent.mkdir(parents=True, exist_ok=True)
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Generated: {tp}")

    return target_paths[0]


if __name__ == "__main__":
    out = build_p0_notebook()
    print("Notebook build completed successfully.")
