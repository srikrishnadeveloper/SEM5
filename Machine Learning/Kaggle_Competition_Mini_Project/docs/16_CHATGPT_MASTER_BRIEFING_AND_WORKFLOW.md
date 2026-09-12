# 16_CHATGPT_MASTER_BRIEFING_AND_WORKFLOW.md — Comprehensive Master Context & Operating Briefing

> **Target Audience:** Incoming ChatGPT Master Session (Zero Prior Context)  
> **Project:** Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup & Kaggle)  
> **Repository Root:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project`  
> **Date:** September 6, 2026  
> **Active Baseline Anchor:** **0.350 Public Leaderboard Score** (V8.1 True Instance Cascade)  
> **Current Operating Directive:** **Directive R3 (Host-Safe Zero-Overlap Trim Submission)**  

---

## 1. Executive Governance & Operating Protocol

### Roles
1. **ChatGPT (You): Master Architect & Red-Team Validator.**
   - Formulates experiment directives, verifies mathematical integrity, gates changes, and ensures 100% adherence to competition rules.
2. **Antigravity (IDE AI Assistant): Lead Execution & Builder Agent.**
   - Modifies local code, maintains unit tests, builds production notebooks, runs local sanity suites, and reports execution logs.
3. **Human User: Bridge & Cloud Operator.**
   - Transmits prompts between ChatGPT and Antigravity, imports generated notebooks into Kaggle, selects hardware, and submits CSVs.

### Ground Rules
- **No Hallucinated Benchmarks:** The only verified score above baseline is **0.350**. Do not cite unverified public notebook titles (e.g. "0.55/0.60") without forensic verification.
- **Strict Host Contract Compliance:** The host strictly rejects overlapping masks and penalizes empty images if dummy rows are emitted.
- **Never Train on Public MAGFiLO Dataverse:** The competition test images are part of the public Harvard Dataverse release. Training on the full Dataverse dataset is **fatal data leakage**.

---

## 2. Competition Specifications & Mathematical Truths

### 2.1 The Task
Predict pixel-precise binary segmentation masks for **every individual solar filament** in 2048×2048 full-disk H-alpha solar chromosphere images from the GONG telescope network (MAGFiLO v1.0 dataset).

### 2.2 Official Evaluation Metric: Kirillov Panoptic Quality (PQ)
- The metric was officially changed to **Panoptic Quality (PQ)** on August 7, 2026, and the leaderboard was re-scored on August 12, 2026.
- Panoptic Quality is the product of Segmentation Quality (SQ) and Recognition Quality (RQ):
  $$\text{PQ} = \text{SQ} \times \text{RQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}|} \times \frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$
- **Matching Contract:** Greedy 1-to-1 matching at **$\text{IoU} > 0.50$**.
  - Since IoU threshold is $> 0.50$, at most one predicted instance can match a ground truth instance.
  - Unmatched predictions are False Positives ($\text{FP}$).
  - Unmatched ground truth filaments are False Negatives ($\text{FN}$).

### 2.3 Single-Class Segmentation Target
- The competition train JSON contains annotations with category IDs:
  - `1`: Left chirality
  - `2`: Right chirality
  - `3`: Unidentifiable chirality
  - `4`: Ambiguous (0 annotations in train JSON, but must map to class 0 if it appears)
- **All categories (1, 2, 3, 4) represent solar filaments.** The task is strictly **single-class binary segmentation** (Class 0: `filament`). Chirality is metadata and is **not evaluated**.

### 2.4 Dataset Anatomy & Multi-Annotator Structure
- **Train Set:** 707 unique physical 2048×2048 JPEGs, but **1,154 annotator observations** and 8,199 polygon annotations.
  - Up to 3 independent human annotators annotated the exact same physical disk.
  - **Leakage Prohibition:** Splits must be grouped by **physical filename / year prefix**, NEVER by COCO `image_id`.
  - **No OR Merging:** Multi-annotator copies must NOT be merged with logical OR (which creates bloated, physically incorrect masks). Each annotator observation remains an independent training sample.
- **Test Set:** 180 physical 2048×2048 JPEGs.

### 2.5 Strict Host Submission Contract
1. **File Format:** Single CSV with exactly two columns: `filament_id,segmentation_rle`.
2. **Row Semantics:** Exactly **one row per actual detected filament** (`{image_stem}_{instance_idx}`).
3. **Empty Disks (Zero Detections):**
   - If a disk has no detected filaments, **ZERO rows must be emitted** for that disk.
   - Emitting dummy rows (like all-zero masks or `PPP2`) creates an unmatchable mask, incurring a False Positive ($|\text{FP}| + 1$) and degrading PQ.
4. **Mandatory ZERO-OVERLAP Rule:**
   - On September 6, 2026, the Kaggle submission engine rejected overlapping masks:  
     > *"Invalid Submission! Submissions may not contain overlapping masks. That is, predictions of the same filament may not overlap."*
   - Submissions must have **strictly zero shared pixels** between any two predicted masks on the same disk:
     $$\forall i \ne j: \quad \sum (\text{mask}_i \land \text{mask}_j) = 0$$
5. **RLE Format:** COCO Fortran-order run-length encoding generated via `pycocotools.mask.encode`.

---

## 3. Architecture & Pipeline History (From V1 to V8.1)

### 3.1 Historical Evolution & The Semantic Plateau
- **V1–V4 Legacy Semantic Pipelines (Score: ~0.22 → 0.30):**
  - Attempted full-disk semantic segmentation (UNet, UNet++, DeepLabV3+, SegFormer) at downsampled resolutions (512, 1024) followed by watershed/connected components.
  - **Why it failed:** Semantic segmentation models merge nearby parallel solar filaments into monolithic blobs. Splitting them post-hoc with watershed or distance transforms causes over-fragmentation or misses thin filaments, triggering heavy PQ penalties.

### 3.2 V8.1: True Instance-Segmentation Cascade (The 0.350 Breakthrough)
To solve the instance segmentation problem natively, we designed a two-stage cascade:

```
2048x2048 Full-Disk H-alpha JPEG
             │
             ▼
[Stage 1: YOLO11s-seg Proposer] @ 1024x1024
   - Trained on 951 train observations (Fold-0 GroupKFold by year)
   - Outputs candidate bounding boxes and rough instance proposals
   - Boxes mapped directly back to native 2048 coordinate frame
             │
             ▼
[Stage 2: Adaptive Square Cropping Engine] (v8_1/geometry.py)
   - Formula: side = min(512, max(256, int(1.2 * max(w, h))))
   - Boundary-safe square bounding window (prevents aspect ratio distortion)
   - 3-Channel Preprocessing: [Raw, CLAHE (clip=3.0), High-Pass Unsharp Gaussian]
             │
             ▼
[Stage 2: Crop Refiner U-Net] @ 384x384 (ResNet-34 Backbone)
   - Specialized in refining thin chromosphere absorption fibril boundaries
   - 4-Flip Crop TTA: [Original, Horizontal, Vertical, Both] averaged
   - Bilinear upsample back to native crop size, pasted into 2048 canvas
             │
             ▼
[Stage 3: Solar Limb Masking + Zero-Overlap Sanitizer] (v8_1/3_infer_cascade.py)
   - Solar Disk clipping at radius r = 0.93 * 1024 (~952 px)
   - Greedy Pixel-Carve Sanitizer: Sorted by confidence desc, area desc
   - mask[occupied > 0] = 0; drop if area < min_area
   - Enforce: sum(mask_i & mask_j) == 0 for all i != j
             │
             ▼
[Stage 4: pycocotools COCO Fortran RLE Encoding] -> submission.csv
```

### 3.3 The Milestone 0.350 Run
- **Kaggle Kernel:** `notebooka464bcfe13`, run `347500668`.
- **Result:** **0.350 Public Leaderboard Score** (+0.05 / +16.7% over the historical 0.30 baseline).
- **Submission Health:** 1,231 rows across 178 disks (2 empty disks emitted 0 rows). Mean 6.92 filaments/disk. 0 errors, 0 penalties.

---

## 4. Recent Directives: R2 & R3 Chronicles

### 4.1 Directive R2: 80-Point Diagnostic Sweep
Grok Master ordered an 80-point grid search on the 128 validation disks across:
- `conf ∈ {0.10, 0.15, 0.20, 0.25, 0.35}`
- `min_area ∈ {50, 100, 200, 400}`
- `yolo_fallback ∈ {0, 1}`
- `overlap_mode ∈ {'trim', 'allow'}`

**Findings:**
1. **`overlap_mode=allow`:** Achieved a validation PQ of **0.4596** (at `conf=0.20, min_area=50`). It retained overlapping candidate masks with IoU $\le 0.5$. **However, the Kaggle host rejected the resulting submission** because touching masks shared pixels.
2. **`overlap_mode=trim` (Legal, Non-Overlapping):** Achieved validation PQ of **0.4389** (at `conf=0.20, min_area=400`), outperforming the 0.350 baseline operating point (**0.4357** at `conf=0.25, min_area=100`).
3. **`yolo_fallback=1`:** Matched `fallback=0` almost identically. YOLO mask fallback was declared ineffective.

### 4.2 Directive R3: Host-Safe Trim Submission
To permanently close the host rejection path:
1. **Mandatory Post-Arbitration Sanitizer (`sanitize_instances_zero_overlap`):**
   - Sorts instances by `confidence` descending, then `area` descending.
   - Greedy pixel carve: `mask[occupied > 0] = 0`.
   - Discards any fragment where `area < min_area` (400 px) after carving.
   - Updates `occupied = np.maximum(occupied, mask)`.
   - **Strict Pairwise Assertion:** Asserts `sum(mask_i & mask_j) == 0` for all $i \ne j$.
   - **Disjoint Union Assertion:** Asserts $\sum \text{area}(m_i) = \text{area}(\bigcup m_i)$.
   - Raises fatal error if even 1 pixel of overlap remains.
2. **Frozen R3 Submit Knobs:**
   - `conf = 0.20`
   - `min_area = 400`
   - `yolo_fallback = 0`
   - `overlap_mode = trim`
   - `tta = True`
   - Weights: Exact 0.350 weights (`best.pt` and `crop_refiner_r34.pth` from `notebooka464bcfe13`).
3. **Fast Execution Pipeline (~4–8 min total):**
   - The 80-point sweep was completely removed from the notebook.
   - Stage C test inference runs directly on the 180 test disks.
   - Audit Cell (Cell 9) re-decodes every test mask and verifies zero pairwise overlap per disk before emitting `[OK]`.
4. **Notebook Artifact:**
   - Rebuilt as `notebooks/V8_1_Kaggle_Production.ipynb` (and mirrored to `C:\Users\srik2\Desktop\V8_1_Kaggle_Production.ipynb`).
   - **Verified SHA256:** `29a7120d69e83bd2b7bf678e3236c23f6aa1725bc0e64575804d152a15ea61ed`.
   - Currently launched on Kaggle Dual T4.

---

## 5. Local Codebase Structure & Fast-Path Architecture

```
Kaggle_Competition_Mini_Project/
├── v8_1/
│   ├── config.py                 # Frozen hyperparameters & dynamic weight resolution
│   ├── geometry.py               # Boundary-safe adaptive square cropping math
│   ├── 1_train_yolo.py           # Stage 1 YOLO11s-seg proposer training
│   ├── 2_train_crop_refiner.py   # Stage 2 ResNet-34 Crop U-Net refiner training
│   ├── 3_infer_cascade.py        # Stage 3 Dual-T4 cascade inference & zero-overlap sanitizer
│   └── 4_sweep_inference.py      # Diagnostic evaluation & validation grid search
├── metrics/
│   └── pq.py                     # Kirillov Panoptic Quality & pycocotools RLE utilities
├── scripts/
│   └── convert_coco_to_yolo.py   # GroupKFold COCO -> YOLO-seg polygon converter
├── notebooks/
│   ├── build_v8_1_kaggle_nb.py   # Modular compiler that packages code into production .ipynb
│   └── V8_1_Kaggle_Production.ipynb # Standalone self-contained Kaggle production notebook
├── tests/                        # 100% passing unit & contract regression suites
│   ├── test_r3_sanitizer.py      # Verifies greedy carve, sub-400px drop, and zero-overlap assertion
│   ├── test_submission_contract.py # Verifies CSV column format, empty disk handling, RLE decode
│   ├── test_v8_1_geometry.py     # Verifies boundary-safe square crop coordinates
│   └── test_host_pq_compat.py    # Verifies greedy Kirillov matching against host logic
├── docs/                         # Directives and architectural records (00 through 16)
└── WORKLOG.md                    # Chronological engineering log
```

### The Fast-Path Cloud Training-Skip Pattern
Training YOLO (20 epochs) + Crop Refiner (20 epochs) takes ~45 minutes on Dual T4. To iterate quickly on inference, postprocessing, or ensembling:
1. When creating a new Kaggle notebook, attach the completed run (`notebooka464bcfe13`) via **+ Add Input → Your Work**.
2. Cells 6 and 7 call `YOLOConfig.resolve_weights()` and `CropRefinerConfig.resolve_ckpt()`.
3. They dynamically scan `/kaggle/input/**` for `best.pt` and `crop_refiner_r34.pth`.
4. If found, training is **skipped in under 4 seconds**, allowing the notebook to run in ~4–8 minutes total.

---

## 6. Current State & Immediate Strategic Roadmap for ChatGPT Master

### Current Live Status
- The user has launched **Directive R3** on Kaggle (`V8_1_Kaggle_Production.ipynb`, SHA `29a7120d...`).
- We are awaiting the test CSV audit and the official Public Leaderboard score.
- **Expected Leaderboard Range:** **~0.35 to 0.38 PQ** (the legal trim winner of the validation sweep).

### Key Next-Phase Exploration Levers for ChatGPT Master
Once R3's score lands, here are the highest-ROI directions to evaluate:

1. **Stage 1 Proposer Recall (Closing the False Negative Gap):**
   - Ground truth has ~8.24 filaments per observation; V8.1 currently detects ~6.92 per disk.
   - Explore training **YOLO11m-seg** (medium backbone) at 1024 or 1280 to capture faint, thin filaments that YOLO11s misses.
2. **Stage 2 Crop Refiner Architecture Upgrade:**
   - Replace ResNet-34 backbone with **ConvNeXt-T** or **EfficientNet-B3/B4** or a lightweight **SegFormer (MiT-B2)** head on crops.
   - Crops are small (384×384), so heavier backbones easily fit in VRAM with batch size 8.
3. **Solar-Specific Feature Engineering:**
   - Implement **Radial Limb Darkening Normalization** (polynomial or Minnaert profile) to equalize contrast between the bright solar disk center and the dark limb.
   - High-pass Butterworth / Bandpass spatial filtering to accentuate magnetic fibril absorption.
4. **Cascade Ensembling:**
   - Combine proposals from YOLO11s + YOLO11m, or ensemble two Crop Refiners (ResNet-34 + ConvNeXt-T) via probability map averaging under 4-flip TTA.
5. **Residual Full-Disk Discovery Branch:**
   - Ablate a lightweight full-disk semantic branch to discover filaments completely missed by bounding-box proposals (adding only instances with $\text{IoU} < 0.30$ vs YOLO proposals).

---

### Operating Hand-off Complete
You are now fully armed with complete context, mathematical constraints, historical lessons, and active codebase architecture. Welcome back, ChatGPT Master!
