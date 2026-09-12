# BUILD REPORT — Phase B: True Instance Cascade Plumbing & Verification
# Master Reviewer: Grok | Builder Agent: Antigravity
# Target: Kaggle Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup)
# Date: 2026-09-05

---

## 1. Submission Contract

A forensic audit of the official competition environment and competition submissions was performed to freeze the submission contract:

| Contract Item | Finding | Details |
| :--- | :--- | :--- |
| **Kaggle Competition Data Files** | **No `sample_submission.csv` provided** | Paged Kaggle API returned exactly 888 files: 707 train JPEGs + 180 test JPEGs + 1 train JSON. Host uploaded zero standalone `sample_submission.csv`. |
| **Verified Test Submissions** | **180 unique stems (Stand-in)** | Analyzed `C:\Users\srik2\Downloads\submission.csv` (180 rows, all-empty team-generated stand-in, not a host file) and `results (4)\submission.csv` (509 rows, multi-instance). |
| **Columns** | `filament_id,segmentation_rle` | Exactly 2 columns. `filament_id` formatted as `{stem}_{instance_index}` (1-indexed). |
| **Empty Disk Representation** | **Single row with zero-mask RLE** | When a test disk contains zero predicted filaments, it produces exactly **one row**: `{stem}_1` with the RLE of an all-zero Fortran 2048x2048 mask. Empty disks are **never omitted**. |
| **RLE Encoding Policy** | **Bitwise Fortran 3D via pycocotools** | Evaluated dynamically via `encode_mask(zero_mask)`. For 2048x2048 zero-mask, `pycocotools.mask.encode(np.asfortranarray(mask, dtype=np.uint8).reshape((h, w, 1)))[0]['counts']` yields `"PPPP4"`. Decoded sum is strictly 0. Never hardcoded as a magic string. |

---

## 2. Dataset Conversion Audit (`scripts/convert_coco_to_yolo.py`)

Annotator-aware COCO to YOLO-seg conversion completed successfully. Polygons are keyed strictly off JSON `image_id`, producing one sample per annotator observation with zero polygon OR-merging:

```
======================================================================
COCO -> YOLO-SEG CONVERTER (Annotator-Aware, Zero-OR)
======================================================================
  Annotation JSON: data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  Image directory: data/MAGFiLO_1.0_Kaggle_2026/train/train_images
  Output root:     data/yolo_seg
  Val fold:        0 (of 5)

PRE-FLIGHT DATASET AUDIT:
  1. JSON images count:          1154
  2. JPEGs on disk:              707
  3. file_name missing on disk:  0 (0.00%)
  4. JPEGs with zero annotations: 0
  5. Polygons audit:
     - Valid filaments (class 0): 8199
     - Skipped (category 4):      0
     - Skipped (<6 pts/coords):   0
     - Skipped (zero area):       0

SPLIT ASSIGNMENT (GroupKFold by year, val_fold=0):
  Train samples (annotator observations): 951
  Val samples (annotator observations):   203
  Train unique physical JPEGs:            579
  Val unique physical JPEGs:              128

WRITING SAMPLES TO DISK...
  File linking method: hardlink (NTFS zero-copy)
  Wrote 951 train files, 203 val files
  Written data.yaml: data/yolo_seg/data.yaml
  Written conversion_report.json: data/yolo_seg/conversion_report.json
======================================================================
```

- **Missing Files:** 0 missing files (100% of JSON filenames exist on disk).
- **Class ID:** Single class `filament` (id 0).
- **Ambiguous:** Category 4 (`Ambiguous`) has 0 annotations in the JSON and is strictly dropped.

---

## 3. Real-Image Smoke Cascade Audit (`scripts/smoke_cascade.py --n 8`)

The cascade plumbing was executed on 8 real MAGFiLO train JPEGs ($2048 \times 2048$) evaluated against real COCO ground truth polygons rasterized with `cv2.fillPoly`. Ground truth is strictly independent of predictions (no leakage):

```
===========================================================================
SMOKE CASCADE — Real-Image MAGFiLO Plumbing & Kirillov PQ Verifier
===========================================================================
  Target Disks:      8
  Data Directory:    data/MAGFiLO_1.0_Kaggle_2026
  Execution Mode:    CPU (Stub Proposer + Real COCO GT)
  Loaded 8 real MAGFiLO images + real COCO ground truth.

---------------------------------------------------------------------------
STEM                   | N_GT | N_PRED |     PQ |     SQ |     RQ |  TP  FP  FN
---------------------------------------------------------------------------
20110109104734Ch       |    6 |      5 | 0.0000 | 0.0000 | 0.0000 |   0   5   6
20110114105034Ch       |    5 |      3 | 0.0000 | 0.0000 | 0.0000 |   0   3   5
20110119082634Lh       |    5 |      5 | 0.0000 | 0.0000 | 0.0000 |   0   5   5
20110123174914Mh       |    3 |      4 | 0.0000 | 0.0000 | 0.0000 |   0   4   3
20110128174814Mh       |    3 |      3 | 0.0000 | 0.0000 | 0.0000 |   0   3   3
20110203082634Lh       |    1 |      5 | 0.0000 | 0.0000 | 0.0000 |   0   5   1
20110205082634Lh       |    4 |      6 | 0.0000 | 0.0000 | 0.0000 |   0   6   4
20110211084114Th       |    1 |      4 | 0.0000 | 0.0000 | 0.0000 |   0   4   1
---------------------------------------------------------------------------
TOTALS / MEAN          |   28 |     35 | 0.0000 | 0.0000 | 0.0000 |   0  35  28
===========================================================================
Audit Check: FN = 28 (Assert > 0 on stub weights: True)
Wall Clock Time: 9.22s

Written CSV: submissions/smoke_real8.csv
Total CSV Rows: 35
Rows per Image Audit:
  20110109104734Ch: 5 row(s)
  20110114105034Ch: 3 row(s)
  20110119082634Lh: 5 row(s)
  20110123174914Mh: 4 row(s)
  20110128174814Mh: 3 row(s)
  20110203082634Lh: 5 row(s)
  20110205082634Lh: 6 row(s)
  20110211084114Th: 4 row(s)

RLE DECODE VERIFICATION:
  Successfully decoded all 35 rows with shape (2048, 2048).
===========================================================================
```

---

## 4. Compliance with Constraints
- No YOLO training was launched.
- No long runs were scheduled.
- Synthetic PQ was eradicated; real Kirillov evaluation is in place with $FN=28 > 0$ on stub weights.
- All code runs on CPU and passes lint/compile checks.
