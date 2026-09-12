# 04_CURRENT_DIRECTIVE.md — Authoritative Architecture Directives
**Authority:** ChatGPT Master (Chief Architect)  
**Execution Date:** September 11, 2026  
**Status:** 🏁 **COMPLETED & EVALUATED (EMPIRICAL LB: 0.330 – 0.340 | REGRESSED VS 0.360 BASELINE)**  


---

## 1. Executive Ruling

ChatGPT Master evaluated the project state, verified 0.360 LB results, and forensic audit, issuing the following official decisions:

1. **Option A Approval:**  
   *"My decision: ✅ APPROVE launching `moonshot_fulldata.ipynb` on Kaggle GPU T4×2. The remaining gap to the 0.55 cluster is not architecture discovery anymore; it is data scale + convergence + inference calibration."*
2. **Checkpoint Strategy Ruling:**  
   *"❌ Do not start completely from scratch. ✅ Use the 60-epoch `best.pt` as initialization and fine-tune on full data."*  
   - *Rationale:* Preserves representation learning of solar disk morphology, filament texture, and instance separation behavior acquired over 60 epochs.
3. **Ensemble Prohibition:**  
   *"Important: Do NOT ensemble with V8.1 again. The previous experiment already proved Moonshot 2048 + V8.1 dropped 0.360 → 0.350 because the 1024 masks damaged SQ and added false positives."*

---

## 2. Approved Training Hyperparameters

| Hyperparameter | Value | Directive Notes |
| :--- | :--- | :--- |
| **Model** | `YOLOv8l-seg` | Approved. Do NOT move to YOLO11l yet (Directive 17). |
| **Resolution** | `imgsz=2048` | **Mandatory.** Do not reduce. |
| **Initial Weights** | `best.pt` | 60-epoch Fold-0 weights (`C:\Users\srik2\Desktop\Filament_Colab_Run\best.pt`). |
| **Training Data** | **100% (1,154 obs)** | Zero holdout loss. All 707 physical images. |
| **Epoch Budget** | **50 Epochs** | Safe ~8.5–9.5h budget. If epoch 45 shows rapid improvement, extend up to 70. |
| **Batch Size** | `batch=1` | Pinned per Decision 09 with AMP; eliminates OOM on T4 (peak 8.71 GB). |
| **Mosaic** | `mosaic=1.0` | **Approved.** Restores multi-scale invariance. |
| **Close Mosaic** | `close_mosaic=10` | Disables mosaic in last 10 epochs for crisp boundary tuning. |
| **Symmetry Flips** | `fliplr=0.5, flipud=0.5` | Planar solar rotational symmetry. |
| **Rotations / Scale** | `degrees=10.0, scale=0.5` | Mild geometry scaling. |
| **Prohibited Augs** | `copy_paste`, heavy perspective, high HSV | Avoided. Physically distorts solar absorption morphology. |
| **RAM Cache** | `cache="ram"` | Stores tensors in Kaggle 30 GB RAM; cuts epoch time from 13 min to ~9 min. |

---

## 3. Calibration Sweep & Submission Contract

1. **Inference Threshold Sweep:**  
   Do NOT lock to 0.25. Scan `conf in [0.15, 0.20, 0.25, 0.30, 0.35]`, `min_area=50`, and evaluate on all 180 test disks.
2. **Primary Candidate:**  
   Freeze primary `submission.csv` at **`conf = 0.30`** (matching HDJoJo SOTA recipe).
3. **Zero-Overlap Guarantee:**  
   Enforce greedy confidence-ordered pixel carving on GPU (`occupied |= m`).
4. **Post-Training Return Criteria:**  
   Return epoch 50 metrics, `best.pt` path, confidence sweep table, row counts, and mean instances/disk before final Kaggle submission.

---

## 4. Empirical Evaluation Outcome & Next Directives
- **Empirical Leaderboard Results:**
  - `conf=0.15`: 0.340
  - `conf=0.20`: 0.340
  - `conf=0.25`: 0.330
  - `conf=0.30`: 0.330 (Frozen primary)
  - `conf=0.35`: 0.330
- **Forensic Diagnosis:** The score regressed from 0.360 ➔ 0.330–0.340 due to (1) 40 epochs of `mosaic=1.0` cutting circular chromosphere disks with crosshair seams, and (2) 110 total fine-tuning epochs causing prototype mask logit dilation (+8.7% area), dropping borderline IoUs below 0.50.
- **Immediate Mandate:**
  1. Revert primary production benchmark to the 60-epoch Fold-0 `mosaic=0.0` champion (`best.pt`, **0.360 LB**).
  2. Permanently ban mosaic on solar disks (`mosaic=0.0`).
  3. Submit next strategic options to ChatGPT Master and the User.

