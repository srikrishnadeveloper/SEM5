# 🚀 Solar Filament Segmentation — 5-Minute Viva Quick Cheat Sheet
**For:** Srikrishna & Project Team  
**Full Guide:** [`team/VIVA_STUDY_GUIDE_SOLAR_FILAMENTS.md`](file:///c:/Users/srik2/Desktop/College/Machine%20Learning/Kaggle_Competition_Mini_Project/team/VIVA_STUDY_GUIDE_SOLAR_FILAMENTS.md)

---

## ⚡ 1. The 30-Second Elevator Pitch
> *"Our project tackles real-time space weather forecasting by performing instance segmentation of solar filaments from 2048×2048 full-disk H-alpha imagery in the MAGFiLO dataset. While previous approaches downsampled images to 1024 and suffered a 75% pixel loss and crop stitching seams (reaching only 0.350 on Kaggle), our architecture trains a native-resolution YOLOv8l-seg model directly on 2048×2048 pixels. We prevent multi-annotator data leakage using GroupKFold by physical disk, enforce a strict zero-overlap physical constraint via greedy pixel carving, and optimize directly for Kirillov Panoptic Quality at IoU > 0.50."*

---

## 📁 2. File & Folder Names You Must Know

| Folder / File Path | What It Is / Why It Matters |
| :--- | :--- |
| **`team/`** | Team study materials, viva guide, and quick reference sheets. |
| **`moonshot_2048/`** | Isolated native-2048 Python modules (`data.py`, `train_yolov8l.py`, `predict_native.py`, `match_and_calibrate.py`). |
| **`notebooks/Moonshot_2048_Train_Fold0.ipynb`** | **The active notebook training on Kaggle** (YOLOv8l-seg @ 2048 on dual Tesla T4 GPUs). |
| **`notebooks/Moonshot_2048_Inference.ipynb`** | Generates test predictions and formats the final `submission.csv`. |
| **`notebooks/build_moonshot_2048_kaggle.py`** | Builder script that compiles our modular code into self-contained Kaggle notebooks. |
| **`v8_1/`** | Baseline 1 cascade anchor (scored **`0.350`** on Kaggle public leaderboard). |
| **`tests/`** | 20 unit tests verifying zero data leakage, metric math, and submission contracts. |
| **`docs/`** | Research directives, master control, and technical build reports. |

---

## 🧠 3. Top 5 Viva Questions & Exact Answers

### Q1: What is a solar filament and why segment it?
* **Answer:** A filament is dense, cool plasma ($10^4\text{ K}$) suspended in the corona ($10^6\text{ K}$) along magnetic neutral lines. It absorbs H-alpha light ($656.3\text{ nm}$), appearing dark. When its magnetic field destabilizes, it erupts into a **Coronal Mass Ejection (CME)**, causing geomagnetic storms on Earth that knock out power grids and satellites. Automated segmentation provides early CME warnings.

### Q2: What is "Multi-Annotator Leakage"?
* **Answer:** 707 physical images have 1,154 observations because multiple solar physicists annotated the same images. If you do random splitting, Observer 1's annotation of Image A goes to Train while Observer 2's annotation of the **same physical image** goes to Val. The model memorizes the disk, giving fake high scores. We solved this by using **GroupKFold grouped by physical filename and year prefix** (579 train, 128 val, exactly 0 shared files).

### Q3: Why did V8.1 score 0.350 and why is Native-2048 better?
* **Answer:** V8.1 downsampled images from $2048 \to 1024$, losing **75% of pixels** and blurring thin filaments. It also used square crops that sliced curved filaments in half. Moonshot trains directly on **native 2048×2048** (zero pixel loss, 46M parameters, single-pass with no crop seams).

### Q4: Explain the Panoptic Quality (PQ) metric and the "0.50 cliff":
* **Formula:** $\text{PQ} = \text{Segmentation Quality (mean IoU of TPs)} \times \text{Recognition Quality (F1 score)}$.
* **The 0.50 Cliff:** Matches require **$\text{IoU} > 0.50$** (which mathematically guarantees unique 1-to-1 matching). If IoU is **0.51**, it counts as a **True Positive**. If it drops to **0.49**, it is penalized as **both a False Positive and a False Negative** (double penalty).

### Q5: What is the Zero-Overlap Rule?
* **Answer:** Two filaments cannot share a pixel on the solar disk. We enforce this via a **greedy pixel-carve algorithm**: sort predictions by confidence and area, carve out any already-claimed pixels, discard fragments smaller than `min_area`, and verify zero shared pixels.

---

## 📊 4. Numbers to Memorize
* **Image Size:** $2048 \times 2048$
* **Dataset:** 707 physical train JPEGs, 1,154 observations, 180 hidden test JPEGs
* **Train / Val Split:** 579 train files / 128 validation files
* **Baseline Score:** **`0.350`** on Kaggle public leaderboard
* **Moonshot Validation Gate:** **$\ge 0.4800$** local PQ
* **Moonshot Target Score:** **$> 0.550$** public leaderboard
