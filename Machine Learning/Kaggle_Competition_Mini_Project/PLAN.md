# PLAN.md — Phase A: True Instance-Segmentation Cascade (V8.1)
# Master Reviewer: Grok | Builder Agent: Antigravity
# Target: Kaggle Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup)
# Goal: Beat Public LB 0.40 PQ (Chase 0.55 Top Cluster)

---

## 1. Workspace & Environment Inspection Summary (Phase A Audit)

A complete inspection of the local workspace and runtime environment was performed without modifying any code files:

| Inspection Item | Status / Finding | Details |
| :--- | :--- | :--- |
| **V7/V8 Python Files** | **Present & Compiling** | `v7/preflight_health_check.py`, `v7/v7_mega_master_pipeline.py`, `v8/1_train_yolo_seed_detector.py`, `v8/2_train_crop_unet_refiner.py`, `v8/3_infer_seed_refine_cascade.py` all exist and pass `python -m py_compile`. All except `preflight_health_check.py` lack `if __name__ == '__main__':` guards (flat procedural). |
| **Dataset on Disk** | **Present & Verified** | `data/MAGFiLO_1.0_Kaggle_2026/`: 707 train JPEGs, 180 test JPEGs (all $2048 \times 2048$). Training JSON: 1,154 image IDs, 8,199 annotations, 4 categories (Left=1, Right=2, Unidentifiable=3, Ambiguous=4). |
| **Installed Packages** | **Partial** | `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`, `smp 0.5.0`, `pycocotools`, `timm 1.0.28`, `albumentations 2.0.8`, `opencv 5.0.0`, `pandas 3.0.3`, `numpy 2.4.6`. `ultralytics` is **missing locally** (will be pip installed in Phase B; readily available on Kaggle/Colab). |
| **`sample_submission.csv`** | **Not in Local Repo** | File is not present locally; `submissions/submission.csv` has a 2-row dummy template with `PPP2`. The recent submission file has 509 rows where empty masks are encoded as the exact Fortran 3D zero-mask string `"PPPP4"`. |
| **Local Compute / Hardware** | **CPU Only** | Local machine is CPU-only. Target production hardware is Kaggle Dual T4 (2x 15.6 GB VRAM) or Colab T4/L4. |

---

## 2. Chosen Architecture: Stage 1 Instance Proposer

Following the strict priority order in `01_ANTIGRAVITY_PROMPT.md`:
1. **Primary Choice:** **`YOLO11m-seg` @ 1280** (or `YOLO11-seg` / `YOLOv8x-seg` fallback).
   - High-recall instance boundary proposals.
   - Lettersize / disk-normalized coordinates.
   - Evaluated on Kaggle Dual T4 with mixed precision (`amp=True`).
2. **Crop Refiner:** Dedicated **U-Net (`resnet34` backbone)** operating on $256 \times 256$ object-centered crops with 20% spatial context padding, using 3-channel astronomical feature inputs (`[raw, CLAHE, unsharp]`).
3. **True Union Fusion Engine:**
   - Run YOLO proposals **AND** a single global semantic checkpoint (`UnetPlusPlus-effb4`) on every image.
   - Global semantic branch detects *residual candidates* missed by YOLO.
   - Candidates are merged, deduplicated, refined via Crop U-Net with 4-flip TTA, and assembled with greedy confidence-score non-overlap arbitration.
   - Optimization target: **Official Kirillov Panoptic Quality ($IoU > 0.5$)**, not Dice.

---

## 3. What We Will Reuse vs. Rewrite

### Reused Components (Proven & Correct)
- **3D Fortran COCO RLE Encoding:** `mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8).reshape((h, w, 1)))[0]['counts']`.
- **Solar Disk Optical Masking:** Geometric radius clipping ($R \approx 0.93 \times 1024 \approx 952.3\text{ px}$) to eliminate off-disk artifacts.
- **Astronomical 3-Channel Preprocessing:** `[Raw Grayscale, CLAHE (clip=3.0, grid=8x8), High-Pass Unsharp Ridge Filter]`.
- **GroupKFold by Year:** Grouping on `filename[:4]` (e.g. `2011`, `2012`) to prevent multi-annotator and temporal leakage.
- **Dual-T4 Model Parallelism Pattern:** Alternating device placement (`dev = f"cuda:{i % n_gpus}"`), strictly avoiding `nn.DataParallel`.
- **Existing Model Checkpoints:** Selected single UNet++ checkpoint retained strictly as a *residual instance discovery branch*, never as a 14-model logit-averaged LB engine.

### Rewritten / Newly Built Components
- **`metrics/pq.py`:** Clean Kirillov PQ evaluator implementing exact Kaggle host logic ($IoU > 0.5$ unique matching, returning PQ, SQ, RQ, TP, FP, FN).
- **Annotator-Aware COCO $\to$ YOLO Dataset Converter:** Separating multi-annotated observations into independent samples. **Zero OR-merging** of multi-annotator polygons (eliminates the 11.6 polygon/image label inflation bug).
- **Lazy-Loading Crop Dataset:** On-the-fly crop generation in `__getitem__` (eliminates the multi-gigabyte in-memory crop array crash).
- **True Union Cascade Fusion:** Eliminates the V8 "fake fusion" bug (`if len(candidates) == 0:`), ensuring global residual discovery runs on all disks.
- **Automated PQ Grid Search:** Calibration of confidence threshold, minimum area, and non-overlap arbitration directly against local holdout PQ.

---

## 4. File-Level Change List

```
├── metrics/
│   ├── __init__.py
│   └── pq.py                        # Kirillov PQ, SQ, RQ, TP, FP, FN scorer (IoU > 0.5)
├── tests/
│   ├── test_pq.py                   # Unit tests on 6 synthetic scenarios (perfect, shifted, merged, split, empty)
│   ├── test_rle_roundtrip.py        # 3D Fortran RLE round-trip identity test
│   └── test_v7_v8_preflight.py      # Updated preflight suite
├── scripts/
│   ├── convert_coco_to_yolo.py      # Annotator-aware dataset generator (no polygon unioning)
│   └── smoke_cascade.py             # 8-disk end-to-end smoke test printing PQ/SQ/RQ and sample CSV
├── v8_1/
│   ├── __init__.py
│   ├── config.py                    # Unified hyperparameter configuration
│   ├── dataset.py                   # Lazy-loaded crop dataset + YOLO format helpers
│   ├── 1_train_yolo.py              # YOLO11m-seg training script
│   ├── 2_train_crop_refiner.py      # Crop U-Net (ResNet-34) refiner training script
│   └── 3_infer_cascade.py           # True fusion inference cascade + PQ post-processor
├── notebooks/
│   └── V8_1_Kaggle_Production.ipynb # Valid, self-contained Kaggle submission notebook
├── BUILD_REPORT.md                  # Comprehensive engineering report for Grok
└── requirements.txt                 # Pinned package requirements including ultralytics & timm
```

---

## 5. Compute Budget (Kaggle Dual T4 / Single T4)

| Stage | Target Hardware | Configuration | Estimated Wall Time |
| :--- | :--- | :--- | :--- |
| **YOLO11m-seg Training** | Kaggle Dual T4 / Single T4 | 30 epochs, img 1280, batch 4, accum 2, AMP | ~75 – 90 minutes |
| **Crop U-Net Refiner Training** | Kaggle Single T4 | 20 epochs on ~8,000 crops, batch 32, AMP | ~18 – 22 minutes |
| **OOF PQ Threshold Sweep** | Kaggle Single T4 | Grid search over conf $\in [0.3, 0.6]$, area $\in [150, 400]$ | ~5 minutes |
| **Test Set Inference (180 Disks)** | Kaggle Dual T4 | True Union Cascade (YOLO + Residual + Crop Refiner + TTA) | ~3.5 minutes |
| **Total End-to-End Budget** | **Kaggle GPU Session** | **Full Pipeline Execution** | **~1.75 to 2.0 hours** (Well within Kaggle 12h limit) |

---

## 6. Verification Plan & Smoke Contract (Phase C)

Execution sequence to verify Phase B implementation before pushing to Kaggle:

```powershell
# 1. Compile all Python files across workspace
python -m py_compile $(Get-ChildItem -Recurse -Filter *.py | Select-Object -ExpandProperty FullName)

# 2. Kirillov PQ Scorer Unit Tests (Synthetic masks: perfect, shifted, merged, split, empty vs empty, empty vs blob)
python tests/test_pq.py

# 3. RLE Fortran 3D Round-Trip Identity Test
python tests/test_rle_roundtrip.py

# 4. Smoke Cascade on 8 Real MAGFiLO Disks (must print PQ/SQ/RQ and generate valid CSV)
python scripts/smoke_cascade.py --n 8
```

---

## 7. Risks & Concrete Mitigations

1. **Empty Disk Encoding Discrepancy (`PPP2` vs `"PPPP4"`):**
   - *Risk:* Incompatible empty string leads to Kaggle scoring failure.
   - *Mitigation:* In MAGFiLO 2048x2048, an all-zero Fortran mask encoded via `pycocotools.mask.encode` yields exactly `"PPPP4"`. Our `rle_empty()` function encodes a verified bitwise zero Fortran array and decodes it back to assert zero active pixels. We also verify against submission row structure.
2. **Multi-Annotator Label Corruption:**
   - *Risk:* Pooling 8,199 annotations over 707 files inflates ground truth from 7.1 to 11.6 filaments/image, causing duplicate bounding boxes.
   - *Mitigation:* `convert_coco_to_yolo.py` indexes annotations by `image_id` (annotator entry), generating distinct training targets per annotator observation with zero polygon unioning.
3. **RAM Memory Exhaustion on Crop Extraction:**
   - *Risk:* Storing 8k-10k crops in memory consumes 3-4 GB RAM.
   - *Mitigation:* Lazy-loaded `FilamentCropDataset` loads source JPEGs and slices $256 \times 256$ crops dynamically inside `__getitem__`.
4. **Notebook JSON Malformation:**
   - *Risk:* Top-level notebooks in `notebooks/` previously failed `json.load`.
   - *Mitigation:* Automated validation test `python -c "import json; json.load(open('...'))"` integrated into the build pipeline.
5. **Secret Leakage:**
   - *Risk:* Unencrypted API tokens committed in `.env`.
   - *Mitigation:* Remove plaintext keys from `.env`; use standard environment variables or Kaggle user secrets.
