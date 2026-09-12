# 07_KAGGLE_RUNBOOK.md — Production Kaggle Execution Runbook
**Project:** Solar Filament Segmentation Challenge 2026  
**Target:** Kaggle GPU Dual T4 / P100  
**Status:** 🚀 **LIVE RUN IN PROGRESS (`notebooka7da5ba0b9` — EPOCH 49/50)**  

---

## 1. Overview & Pre-Flight Checklist

This runbook provides the exact, unambiguous instructions and verified telemetry for the full-data native 2048 YOLOv8l-seg fine-tuning run on Kaggle.

### Pre-Flight Requirements & Audit Verifications:
- [x] Staged Notebook: `moonshot_fulldata.ipynb` (SHA256: `fab5dc00a7180671809c8bbfa2e4f164f26ee483894e030043b6bd9ed1150c9c`).
- [x] Attached Weights: `best · best · V1` (`best.pt`, 92.8 MB, SHA256: `f444e87b39433881ae39608e9740c44c9bab0161590212047e224b6730d6b6f9`).
- [x] Memory Hardening: Pinned `batch=1` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (guarantees safe 7.75–8.71 GB VRAM, preventing CUDA OOM on T4).
- [x] Early Stopping Hardening: Pinned `patience=50` to guarantee all 50 full epochs execute.

---

## 2. Step-by-Step Execution Protocol

### Step 1: Initialize Kaggle Notebook
1. Navigate to [kaggle.com/code](https://www.kaggle.com/code) and select **New Notebook**.
2. Click **File** ➔ **Upload Notebook**.
3. Select `C:\Users\srik2\Desktop\Filament_Colab_Run\moonshot_fulldata.ipynb`.

### Step 2: Configure Hardware & Networking
1. In the notebook settings panel (right sidebar):
   - **Accelerator:** Select **GPU T4 x 2** (or **GPU P100**).
   - **Internet:** Toggle to **ON** (required for `pip install ultralytics==8.3.145 pycocotools`).
   - **Environment:** Use default Kaggle Python 3.10/3.11 environment.

### Step 3: Attach Input Datasets
1. **Competition Dataset:**
   - Click **+ Add Input** ➔ Search `filament-segmentation-2026` ➔ Click **Add**.
   - Verified path: `/kaggle/input/competitions/filament-segmentation-2026` or `/kaggle/input/filament-segmentation-2026`.
2. **Prior Checkpoint Dataset (`best.pt`):**
   - Click **+ Add Input** ➔ **Upload Dataset**.
   - Select file: `C:\Users\srik2\Desktop\Filament_Colab_Run\best.pt`.
   - Name the dataset (e.g. `solar-filament-yolo-best`) and select **Private**.
   - Attach the uploaded dataset as an input to the notebook.
   - *Verification in Cell 5:* The notebook will print:  
     `Found existing weights: /kaggle/input/.../best.pt -> Fine-tuning on full data!`

### Step 4: Execution
- Click **Save Version** ➔ Select **Save & Run All (Commit)**.
- *Why Commit Mode:* Runs in the background independently of browser connectivity and safely terminates before Kaggle's 12-hour session limit.
- **Estimated Runtime:** **~8.5 to 9.5 hours** on Dual T4 with `cache="ram"`.

---

## 3. Post-Training Artifacts & Deliverables

When the notebook completes, `/kaggle/working/` will contain:

| Artifact | File Name | Description |
| :--- | :--- | :--- |
| **Primary Submission** | `submission.csv` | Frozen at **`conf = 0.30`** (strictly zero overlap, Host V6 verified). |
| **Sweep Summary** | `sweep_summary.csv` | Metrics table across `conf = 0.15, 0.20, 0.25, 0.30, 0.35`. |
| **Alternative Candidates**| `submission_conf*.csv` | Individual CSVs for each confidence threshold. |
| **Trained Checkpoint** | `best_fulldata.pt` | ~93 MB fine-tuned weights for local archival. |
| **Training Log** | `results.csv` | Epoch-by-epoch loss, precision, recall, and mAP trajectory. |

---

## 4. Post-Execution Reporting Protocol

Before submitting `submission.csv` to the competition leaderboard, the user must report the following back to ChatGPT Master:
1. Final epoch metrics (Box mAP50, Mask mAP50, `seg_loss`).
2. Contents of `sweep_summary.csv` (total rows, mean instances/disk for each confidence threshold).
3. Result of the Cell 7 Host Evaluation V6 audit check.
