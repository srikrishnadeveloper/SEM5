# 00_MASTER_CONTEXT_CHATGPT.md — Authoritative Master Context
**Project:** Solar Filament Segmentation Challenge 2026 (Kaggle & IEEE BigData Cup)  
**Current Master / Red-Team Reviewer:** ChatGPT  
**Executor / Builder:** Antigravity  
**Grok Status:** INACTIVE until human explicitly says "Grok back"  
**Human Role:** Non-technical copy/paste bridge; no technical decisions required  
**Date:** September 8, 2026  
**Primary Master Bundle:** See `master/` directory for ready-to-attach ChatGPT files.

---

## 1. Authoritative Competition Truths

1. **Primary Leaderboard Metric:**
   - Since August 7, 2026, the official competition evaluation metric is **Kirillov Panoptic Quality (PQ)** ($IoU > 0.5$).
   - The competition leaderboard was officially re-scored on August 12, 2026.
   - Host evaluation logic is codified in **Self Evaluation V6**: pure Fortran RLE counts string, 1-indexed filament IDs (`{stem}_{i+1}`).

2. **Submission Format & Semantics:**
   - Each CSV row corresponds to **one actual predicted filament**.
   - **No Dummy Zero-Mask Rows:** If zero filaments are detected on a test image, **zero rows are emitted** for that image.
   - Strictly **zero shared pixels** between any two masks on the same disk (enforced via greedy pixel-carve sanitizer).
   - Every submitted RLE must decode to a 2048×2048 binary mask with positive active area ($\text{sum} > 0$).

3. **Classification & Chirality:**
   - Single-class binary filament segmentation: all categories 1, 2, 3, 4 map strictly to Class 0 (`filament`).

4. **Dataset Truth:**
   - 707 unique physical training JPEGs, 1,154 COCO annotator-level observations, 8,199 polygon annotations, 180 test JPEGs.
   - Multiple annotators are never merged with OR operations; each annotator observation remains an independent training sample.
   - Harvard Dataverse external training data is strictly prohibited (data leakage).

5. **Current Verified Benchmarks & Active Baselines:**
   - V8.1 True Cascade: **0.350 Public Leaderboard Score** (1,231 rows across 178 disks).
   - **Moonshot 2048 (Native YOLOv8l-seg @ 2048): 0.360 Public Leaderboard Score** (1,182 rows, Fold 0, 82% data).
   - Cross-Architecture Ensemble: **0.350 Public Leaderboard Score** (regressed due to 1024px mask dilation & 146 extra false positives).
   - **Moonshot Full-Data (Ready to Run):** Built in `notebooks/kaggle/moonshot_fulldata.ipynb` with 100% data (1,154 observations), `mosaic=1.0`, `cache="ram"`, and pure PyTorch GPU inference engine. Target: **0.50+ Public Leaderboard**.

---

## 2. Frozen Production Architecture (V8.1 Baseline 1)

```
GONG 2048x2048 Full Disk JPEG
            │
            ▼
Stage 1: YOLO11s-seg Proposer @ 1024 (Single GPU cuda:0, 20 epochs, batch 2, AMP)
            │ (Boxes in native 2048 pixel coordinates)
            ▼
Stage 2: Strictly Square Adaptive Cropping (256..512 px window centered on proposal)
            │ (Square resized to 384x384 without aspect-ratio distortion)
            ▼
Stage 2: ResNet-34 U-Net Crop Refiner @ 384 (batch 8, 20 epochs, 4-flip crop TTA)
            │
            ▼
Stage 3: Paste-back to 2048x2048 canvas + Solar Disk Limb Mask (r = 0.93)
            │
            ▼
Stage 3: Greedy Confidence-Ordered Non-Overlap Arbitration (Zero-OR)
            │
            ▼
Stage 3: pycocotools Fortran-order RLE encoding (Only positive-area detections)
            │
            ▼
`/kaggle/working/submission.csv` (filament_id, segmentation_rle)
```

- **Hardware Placement:**
  - Training: Single GPU (`cuda:0`) for Stage 1 YOLO (avoids Kaggle notebook DDP subprocess hangs).
  - Inference: Model parallelism across Dual T4 (`cuda:0` for YOLO, `cuda:1` for Crop Refiner).
- **Residual Discovery Branch:** Explicitly DISABLED for Baseline 1.
- **Crop Geometry:** Shared helper `v8_1/geometry.py` guarantees strict square bounds `(x2 - x1) == (y2 - y1) == side` across image borders without aspect-ratio stretching.
- **Production Artifact SHA256 (Current):** `331ce02dc7c0defd73fc6a037b093e0c65a628588f27cd8af0c13118fba15d01`

---

## 3. Active Rulebook & Roles

- **ChatGPT:** Master Reviewer & Independent Red-Team Validator. All technical reviews, gates, and approvals come from ChatGPT.
- **Antigravity:** Execution and builder agent. Implements verified fixes, runs unit/smoke tests, and produces evidence.
- **Human User:** Copy/paste bridge. Not expected to make technical decisions.
- **Grok:** Inactive until human explicitly enters "Grok back".
- **Rule of Evidence:** Never claim correctness or safety without verifiable code execution, unit tests, and programmatic inspection.
