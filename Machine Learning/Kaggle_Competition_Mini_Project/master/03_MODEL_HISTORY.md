# 03_MODEL_HISTORY.md — Chronological Architecture History & Forensics
**Project:** Solar Filament Segmentation Challenge 2026  
**Status Date:** September 11, 2026  

---

## 1. Chronological Model Generations

```
[V1 Monolithic U-Net] (512×512) ──> LB 0.220
         │
         ▼
[V3/V4 Semantic Ensembles] (512–1024) ──> LB 0.300
         │
         ▼
[V8.1 True Instance Cascade] (YOLOv8 1024 + Crop Refiner 384) ──> LB 0.350 (Anchor)
         │
         ▼
[Moonshot Native 2048 YOLOv8l-seg] (Fold 0, 60 epochs) ──> LB 0.360 (Highest Verified)
         │
         ├── [Ensemble: Moonshot + V8.1] ──> LB 0.350 (Regressed: 1024 dilation + 146 FPs)
         │
         ▼
[Moonshot Full-Data Native 2048] (100% Data, Mosaic=1.0, 50 epochs) ──> LB 0.340 (Regressed: Mosaic Seams)
```


---

## 2. Detailed Generation Specifications

### Generation 1: V1 Monolithic Semantic U-Net (512×512)
- **Architecture:** Standard U-Net (`smp`) with ResNet-34 encoder.
- **Input Resolution:** 512×512 (bilinear downscaled from 2048).
- **Target:** Binary semantic segmentation (foreground vs background).
- **Metric Score:** **0.220 Public LB**.
- **Fatal Flaws:** Heavy resolution downsampling destroyed 80% of thin filaments; semantic thresholding merged adjacent instances into singular giant blobs, failing instance-level IoU matching.

### Generation 2: V3 / V4 Semantic Multi-Backbone Ensembles (1024×1024)
- **Architectures:** DeepLabV3+ (ResNet-50d), SegFormer (MiT-B3), U-Net++ (EfficientNet-B4).
- **Input Resolution:** 1024×1024 with watershed post-processing.
- **Metric Score:** **0.300 Public LB**.
- **Limitations:** Watershed separation proved brittle on curvilinear geometry, causing over-segmentation and fragmented instances that severely penalized Recognition Quality (RQ).

### Generation 3: V8.1 True Instance Cascade (Verified Baseline Anchor)
- **Architecture:** Two-stage cascade:
  1. *Stage 1 (Detector):* YOLOv8l bounding box detector at 1024×1024.
  2. *Stage 2 (Refiner):* U-Net++ (EfficientNet-B4) running on adaptive $384\times 384$ local square crops around each detected filament.
- **Metric Score:** **0.350 Public LB**.
- **Strengths:** Fully compliant with Host Self-Evaluation V6; zero overlap; native crop resolution recovered local details.
- **Limitations:** Crop bounding boxes from Stage 1 truncated long, curving filaments; high computational complexity during inference.

### Generation 4: Moonshot Native 2048 YOLOv8l-seg (Fold-0)
- **Architecture:** Native full-resolution YOLOv8l-seg (`ultralytics`).
- **Input Resolution:** **2048×2048 Native** (no downsampling, no tiling).
- **Training Scope:** Fold 0 (82% data, 951 observations, 128 physical images held out).
- **Hyperparameters:** `epochs=60`, `batch=2`, `mosaic=0.0` (disabled), `lr0=0.01`, `conf=0.25`.
- **Metric Score:** **0.360 Public LB** (**Current Project Record**).

### Generation 5: Moonshot Full-Data Native 2048 (Evaluated Sept 11, 2026)
- **Architecture:** Native full-resolution YOLOv8l-seg (`ultralytics`).
- **Input Resolution:** **2048×2048 Native** (uncompressed full disk).
- **Training Scope:** **100% of dataset (all 1,154 observations / 707 physical images)**, zero holdout.
- **Initialization:** Fine-tuned from 60-epoch Fold-0 checkpoint `best.pt`.
- **Hyperparameters:** `epochs=50`, `batch=1` (pinned with AMP to prevent OOM on T4), `mosaic=1.0`, `close_mosaic=10`, `cache="ram"`, `patience=50`.
- **Inference & Calibration:** Pure PyTorch GPU inference (~35s for 180 disks), limb mask $r=0.93$, greedy zero-overlap sanitizer, calibration sweep across `conf in [0.15, 0.20, 0.25, 0.30, 0.35]`.
- **Official Public LB Results:**
  - `conf=0.15`: **`0.340`**
  - `conf=0.20`: **`0.340`**
  - `conf=0.25`: **`0.330`**
  - `conf=0.30`: **`0.330`** (Primary frozen)
  - `conf=0.35`: **`0.330`**
- **Status:** **Regressed vs 0.360 Baseline (Score plateau at 0.330–0.340)**.

---

## 3. Forensic Analysis of Experiments

### A. Why the Cross-Architecture Ensemble Regressed (0.360 ➔ 0.350)
On September 7, 2026, Antigravity built an ensemble combining Moonshot 2048 masks with V8.1 cascade masks. The result was a drop from 0.360 to 0.350:
1. **Segmentation Quality (SQ) Degradation:** V8.1 masks (upscaled from 1024 global coordinates) had soft, dilated boundaries. Averaging them with crisp 2048px Moonshot masks degraded boundary alignment, lowering SQ.
2. **False Positive Inflation:** V8.1 generated 146 extra low-confidence detections that were not corroborated by Moonshot (1,323 total vs 1,182). In the Kirillov PQ denominator:
   $$\text{Denominator} = |\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|$$
   Each uncorroborated detection added $+0.5 \cdot \text{FP}$, penalizing RQ and dropping the score.
3. **Master Rule Established:** **Strictly prohibit ensembling 2048 native models with 1024 downsampled models.**

### B. Why Full-Data with Mosaic Regressed (0.360 ➔ 0.330–0.340)
Empirical audit between the 0.360 model and the full-data model revealed:
1. **Mosaic Augmentation Harm (`mosaic=1.0`):** In the 0.360 model, `mosaic=0.0` was strictly disabled. Circular solar disks and continuous chromospheric absorption channels were intact. In the full-data model, 40 epochs of mosaic 4-quadrant stitching cut filaments across artificial crosshairs, training the prototype masks to produce artificial straight-line boundaries.
2. **Cumulative Prototype Over-Smoothing (110 Total Epochs):** Fine-tuning for 50 additional epochs on top of the already-converged 60-epoch `best.pt` caused representation drift in the mask prototype coefficients. Filaments experienced an average +8.7% area dilation (median area 1,451 px ➔ 1,577.5 px), causing borderline IoUs to slip from 0.52 to 0.48.
3. **Public SOTA Reality Check:** Forensic inspection of public notebooks claiming "0.55+" proved they relied on static precomputed test payloads (`CHAMPION_PAYLOAD` with 1,342 precomputed masks) or pre-August 12 broken metric evaluation. On the official, re-scored Panoptic Quality leaderboard, **0.360 is the project's genuine verified record**.

