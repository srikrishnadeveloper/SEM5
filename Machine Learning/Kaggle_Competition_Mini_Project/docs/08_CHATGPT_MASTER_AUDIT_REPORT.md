# V8_1 PRE-KAGGLE FORENSIC AUDIT & VERIFICATION REPORT
**Master Reviewer:** ChatGPT  
**Execution Agent:** Antigravity  
**Independent Validator:** Grok  
**Timestamp:** 2026-09-05T17:00:00+05:30  
**Target:** Kaggle Solar Filament Segmentation Challenge 2026 (Dual T4 GPU, Internet ON)  
**Notebook Artifact:** `notebooks/V8_1_Kaggle_Production.ipynb`  

---

## 1. CURRENT GIT / WORKTREE STATE

The local git repository state was audited via `git status -s`.
- The current working directory is clean of unwanted modifications.
- Active V8.1 production modules:
  - `v8_1/config.py`
  - `v8_1/1_train_yolo.py`
  - `v8_1/2_train_crop_refiner.py`
  - `v8_1/3_infer_cascade.py`
  - `scripts/convert_coco_to_yolo.py`
  - `metrics/pq.py`
  - `notebooks/build_v8_1_kaggle_nb.py`
  - `notebooks/V8_1_Kaggle_Production.ipynb`
- Verified that all historical files (`v1`–`v4`, `solar-filament-segmentation-boobathi_branch`) are untouched / unaffected.
- No Kaggle API keys or secrets are stored in any source or notebook files.

---

## 2. FILES INSPECTED

The following files were inspected line-by-line:
1. `docs/00_SHARED_CONTEXT.md`
2. `docs/01_ANTIGRAVITY_PROMPT.md`
3. `docs/02_GROK_VERDICT_PHASE_A.md`
4. `docs/03_GROK_VERDICT_PHASE_B.md`
5. `docs/04_GROK_VERDICT_PHASE_C.md`
6. `docs/05_KAGGLE_DUAL_T4_CONTRACT.md`
7. `docs/06_GROK_VERDICT_NOTEBOOK.md`
8. `docs/07_GO_KAGGLE.md`
9. `AGENTS.md`
10. `LESSONS_LEARNED.md`
11. `metrics/pq.py`
12. `scripts/convert_coco_to_yolo.py`
13. `scripts/smoke_cascade.py`
14. `v8_1/config.py`
15. `v8_1/1_train_yolo.py`
16. `v8_1/2_train_crop_refiner.py`
17. `v8_1/3_infer_cascade.py`
18. `notebooks/build_v8_1_kaggle_nb.py`
19. `notebooks/V8_1_Kaggle_Production.ipynb`
20. `C:\Users\srik2\Desktop\Filament_Colab_Run\V8_1_Kaggle_Production.ipynb`
21. `C:\Users\srik2\Desktop\V8_1_Kaggle_Production.ipynb`

---

## 3. BUGS ACTUALLY FOUND

1. **Bug A2 (Cross-Mount File Linking in Converter):**
   In `scripts/convert_coco_to_yolo.py`, the linking logic tested hardlink or symlink, but in auto mode defaulted to attempting hardlink first. Because `/kaggle/input` (read-only mount) and `/kaggle/working` (overlay filesystem) reside on distinct mounts, `os.link` raises `OSError: [Errno 18] Invalid cross-device link (EXDEV)`. Furthermore, per-file materialization had no post-creation existence or non-zero size assertions.

2. **Bug B1 (Full-Set 2048x2048 Mask Memory Retention in Validation):**
   In `v8_1/1_train_yolo.py`, `val_gt_by_file` pre-rasterized ground truth 2048x2048 masks for all annotators across all 128 validation images simultaneously in RAM (~5.7 GB). This posed a major memory spike and OOM risk on Kaggle's ~13 GB CPU memory during validation.

3. **Bug A10 (Residual Discovery Display Inconsistency):**
   In `v8_1/3_infer_cascade.py`, `print_infer_config` indicated that `--residual 1` would enable `IoU < 0.30` discovery, even though residual logic is intentionally omitted in `run_cascade_inference` for Baseline 1. This risked giving the false impression that residual proposals were running.

4. **Bug A3 (Notebook Reporting of Dataset Split Views):**
   Cell 5 of the generated notebook only reported total image file counts, without reporting both annotator-level sample counts (951 train / 203 val) and unique physical JPEG counts (579 train / 128 val).

---

## 4. CHANGES ACTUALLY MADE

1. **Robust Materialization Strategy (`scripts/convert_coco_to_yolo.py`):**
   - Implemented `materialize_file(src, dst, mode="auto")` with ordered fallback: `symlink` (works cross-mount on Linux/Kaggle without copying 700MB) -> `hardlink` (local NTFS zero-copy) -> `shutil.copy2` (universal fallback).
   - Added strict post-materialization assertions: `assert dst_img.exists() and dst_img.stat().st_size > 0`.
   - Recorded `train_unique_jpegs` and `val_unique_jpegs` in `conversion_report.json`.

2. **Streaming Memory-Safe Validation (`v8_1/1_train_yolo.py`):**
   - Refactored `evaluate_val_pq` to store only polygon coordinates (negligible memory).
   - Rasterization of 2048x2048 ground truth masks is performed on-demand for one physical image at a time, evaluated against predictions, and immediately deleted (`del pred_masks, annotator_instances`).
   - Added tracking and summary reporting of `SQ`, `RQ`, `TP`, `FP`, `FN` alongside `pq_mean` and `pq_max`.
   - Set `"plots": False` in Ultralytics `train_kwargs`.

3. **Explicit Residual Notice (`v8_1/3_infer_cascade.py`):**
   - Updated `print_infer_config` to report `Residual Discovery: DISABLED (Baseline 1: Pure YOLO -> Crop Refiner cascade)`.
   - Added explicit notification in `main()` if `--residual 1` is passed, ensuring no false impression of active residual branches.

4. **Enhanced Stage 0 Inspection in Notebook (`notebooks/build_v8_1_kaggle_nb.py`):**
   - Updated Cell 5 to read `conversion_report.json` and print both:
     - Annotator-level samples (`train_samples_count` / `val_samples_count`)
     - Unique physical JPEGs (`train_unique_jpegs` / `val_unique_jpegs`)

5. **Desktop Notebook Mirroring & SHA256 Sync:**
   - Updated `build_v8_1_kaggle_nb.py` to mirror the generated notebook to:
     - `C:/Users/srik2/Desktop/Filament_Colab_Run/V8_1_Kaggle_Production.ipynb`
     - `C:/Users/srik2/Desktop/V8_1_Kaggle_Production.ipynb`
   - Verified that all three notebooks have identical SHA256 hashes.

---

## 5. BUGS SUSPECTED BUT NOT REPRODUCED / ALREADY CLEAN IN CODE

1. **Shell Variable `$data_dir` in Cell 5:**
   The codebase already uses `subprocess.check_call([sys.executable, "scripts/convert_coco_to_yolo.py", "--data-dir", str(data_dir), "--out-dir", str(out)])`. No `$data_dir` shell variable exists in executable code.

2. **Ultralytics Coordinate Rescaling (`w / 1024`, `* scale_x`):**
   In `v8_1/3_infer_cascade.py`, native 2048 JPEGs are passed directly to `yolo_model.predict()`. Coordinates from `result.boxes.xyxy` are already in native 2048 space. No coordinate scaling is applied; coordinates are directly clipped to `[0, w]` and `[0, h]`. A first-image debug check asserts `max_c <= max(h, w) + 50`.

3. **Silent Stem Dropping on Missing/Corrupt Images:**
   In `v8_1/3_infer_cascade.py`, if `raw is None`, the code increments `empty_disk_count`, encodes a 2048x2048 zero-mask via `encode_mask(zero_mask)`, appends `{stem}_1`, and preserves the stem. Exactly 180 stems are emitted and asserted.

4. **Hardcoded `"PPP2"` or `"PPPP4"` Tokens:**
   Both empty and active masks are encoded dynamically via `pycocotools.mask.encode` on 3D Fortran-contiguous arrays `(H, W, 1)`. No hardcoded strings are returned.

5. **Ultralytics DDP Subprocess Hangs (`device=[0,1]`):**
   Training is frozen on single GPU `device=0` with `batch=2` and `epochs=20`. Dual-T4 is utilized during Stage 3 inference (YOLO on `cuda:0`, Crop Refiner on `cuda:1`).

---

## 6. STATIC STALE-PATTERN SCAN

Scanned all target source files (`v8_1/*.py`, `scripts/*.py`, `metrics/pq.py`, `notebooks/*.py`, `notebooks/*.ipynb`):

| Pattern | Found Occurrences | Classification | Status / Evidence |
|---|---|---|---|
| `"$data_dir"` | 0 | CLEAN / FIXED | No occurrences in source or notebook code |
| `"scale_x = w /"` | 0 | CLEAN / FIXED | No coordinate scaling applied |
| `"scale_y = h /"` | 0 | CLEAN / FIXED | No coordinate scaling applied |
| `"w / 1024"` | 0 | CLEAN / FIXED | Clean |
| `"h / 1024"` | 0 | CLEAN / FIXED | Clean |
| `"* scale_x"` | 0 | CLEAN / FIXED | Clean |
| `"* scale_y"` | 0 | CLEAN / FIXED | Clean |
| `"device=[0,1]"` | 0 | CLEAN / FIXED | Single GPU `device=0` for baseline training |
| `"device = [0,1]"` | 0 | CLEAN / FIXED | Single GPU `device=0` for baseline training |
| `"--batch 4"` | 0 | CLEAN / FIXED | Baseline frozen at `--batch 2` |
| `"PPP2"` | 0 | CLEAN / FIXED | Clean (eliminated) |
| `"PPPP4"` | 2 (docstring only) | DOCUMENTATION ONLY | Mentioned in docstring of `metrics/pq.py:27` as example output; dynamic pycocotools encoding is always executed in code |
| `"if raw is None: continue"` | 0 | CLEAN / FIXED | If raw is None, `{stem}_1` with zero mask is emitted; stem is never dropped |

---

## 7. TEST RESULTS

All local verification suites were executed and passed cleanly:

1. **Python Syntax Compilation Check:**
   `python -m py_compile v8_1\config.py v8_1\1_train_yolo.py v8_1\2_train_crop_refiner.py v8_1\3_infer_cascade.py scripts\convert_coco_to_yolo.py notebooks\build_v8_1_kaggle_nb.py`
   **Result:** Exit code 0 (All files compiled successfully).

2. **PQ Unit Tests (`tests/test_pq.py`):**
   - `test_perfect_copy`: PQ=1.0000 [PASS]
   - `test_shifted_copy`: PQ=0.5385, SQ=0.5385 [PASS]
   - `test_merged_pair`: PQ=0.0000, TP=0, FP=1, FN=2 [PASS]
   - `test_fragmented_pair`: PQ=0.0000, TP=0, FP=2, FN=1 [PASS]
   - `test_empty_vs_empty`: PQ=1.0000 [PASS]
   - `test_empty_vs_blob`: PQ=0.0000, FN=1 [PASS]
   - `test_blob_vs_empty`: PQ=0.0000, FP=1 [PASS]
   **Result:** ALL 7 PQ TESTS PASSED.

3. **COCO RLE Round-Trip Tests (`tests/test_rle_roundtrip.py`):**
   - `test_nonzero_roundtrip`: 50,000 active pixels [PASS]
   - `test_zero_2048_roundtrip`: Decoded sum = 0, shape (2048, 2048) [PASS]
   - `test_varied_sizes`: Resolutions 256, 512, 1024, 2048 all verified [PASS]
   - `test_no_hardcoded_strings`: All encoded dynamically [PASS]
   **Result:** ALL 4 RLE ROUND-TRIP TESTS PASSED.

4. **Dataset Split Isolation & Leakage Tests (`tests/test_dataset_split.py`):**
   - 579 train files, 128 val files, 0 overlap [PASS]
   - All 5 GroupKFold folds verified: 0 physical file_name leakage [PASS]
   **Result:** DATASET SPLIT ISOLATION TESTS PASSED.

5. **Full Pipeline CPU Smoke Cascade on Real MAGFiLO Disks (`scripts/smoke_cascade.py --n 2`):**
   - Processed 2 real 2048x2048 GONG images with 3-channel astronomical preprocessing.
   - Evaluated against real rasterized COCO ground truth.
   - Computed per-disk Kirillov PQ / SQ / RQ / TP / FP / FN.
   - Generated `submissions/smoke_real8.csv` (8 rows) and verified RLE decode round-trip on all rows.
   **Result:** PASSED in 1.42 seconds.

6. **YOLO Proposer Help & Dry-Run Checks:**
   - `python v8_1/1_train_yolo.py --help`: Exit code 0 [PASS]
   - `python v8_1/1_train_yolo.py --dry-run`: Exit code 0 [PASS]
     Confirmed frozen parameters: Architecture: `yolo11s-seg.pt`, Imgsz: 1024, Epochs: 20, Batch: 2, Device: `cuda:0` (Single GPU), AMP: True, Mosaic: 0.5.

---

## 8. NOTEBOOK COMMAND EXTRACTION

Programmatic extraction from `notebooks/V8_1_Kaggle_Production.ipynb`:

### Cell 5: Stage 0 Dataset Conversion
```python
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
print("\n--- data.yaml content ---")
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
```

### Cell 6: Stage 1 YOLO11s-seg Training
```python
# Stage 1: Train YOLO11s-seg Proposer on Single GPU 0 (batch 2, 20 epochs, AMP, mosaic 0.5)
# Single GPU 0 avoids notebook DDP subprocess hangs on Kaggle
!python v8_1/1_train_yolo.py --batch 2 --epochs 20
```

### Cell 7: Stage 2 Crop Refiner Training
```python
# Stage 2: Train Crop U-Net Refiner (ResNet-34) on Train-Fold Crops
# --boxes-from-gt extracts crops directly from GT polygons for clean pass 1 training
!python v8_1/2_train_crop_refiner.py --boxes-from-gt
```

### Cell 8: Stage 3 Inference Cascade
```python
from pathlib import Path
from v8_1.config import YOLOConfig

best_yolo = YOLOConfig.resolve_weights()
print(f"Inference using YOLO weights: {best_yolo}")
assert best_yolo.exists(), f"FATAL: YOLO weights missing at {best_yolo}!"

# Stage 3: True Union Cascade Inference on 180 Test Disks (Model-Parallel Dual T4)
# YOLO on cuda:0, Crop Refiner on cuda:1
!python v8_1/3_infer_cascade.py --conf 0.25 --residual 0 --weights "{best_yolo}" --out /kaggle/working/submission.csv
```

---

## 9. SOURCE-VS-NOTEBOOK SYNCHRONIZATION RESULT

Programmatic byte verification confirmed that Cell 4 of `V8_1_Kaggle_Production.ipynb` embeds the exact, unmodified contents of:
- `metrics/pq.py` -> EXACT MATCH
- `scripts/convert_coco_to_yolo.py` -> EXACT MATCH
- `v8_1/config.py` -> EXACT MATCH
- `v8_1/1_train_yolo.py` -> EXACT MATCH
- `v8_1/2_train_crop_refiner.py` -> EXACT MATCH
- `v8_1/3_infer_cascade.py` -> EXACT MATCH

---

## 10. SHA256 HASHES

| File Path | SHA256 Digest |
|---|---|
| `notebooks/V8_1_Kaggle_Production.ipynb` | `777dd86e858c4bd5cc1127dda80c4e077955612c233f68fde7cf47df6a3d969f` |
| `C:/Users/srik2/Desktop/Filament_Colab_Run/V8_1_Kaggle_Production.ipynb` | `777dd86e858c4bd5cc1127dda80c4e077955612c233f68fde7cf47df6a3d969f` |
| `C:/Users/srik2/Desktop/V8_1_Kaggle_Production.ipynb` | `777dd86e858c4bd5cc1127dda80c4e077955612c233f68fde7cf47df6a3d969f` |
| `v8_1/config.py` | `e8b5a328cc8ed5cebb58758b97a21a799e4ad69d607f81d277d8b1643a44c4dc` |
| `v8_1/1_train_yolo.py` | `af38c925090f021eebef61829f5d4d43e53b1eadc46a7a9cf80541c1ee55d4c9` |
| `v8_1/2_train_crop_refiner.py` | `1b8bb87c2271bdb0ae17433bf89939f9b31237dd44932592cceebbb7494dead9` |
| `v8_1/3_infer_cascade.py` | `3dbc0d572996d4ce799aba0d5c9a7623119b9ab07c6dcb6926f7d614c66c76c3` |
| `scripts/convert_coco_to_yolo.py` | `1f82eca38e960fac55cf04ae37d4d2222ea6bfefd03fa19270f00e03505fdb17` |
| `metrics/pq.py` | `a62cfc8a392f6055f79a5f07a47c68293967910b2a5ab1f53e835c50042f4b18` |

**Verification:** All three notebook copies (repository, `Filament_Colab_Run`, and Desktop) share the identical hash: `777dd86e858c4bd5cc1127dda80c4e077955612c233f68fde7cf47df6a3d969f`.

---

## 11. KNOWN UNRESOLVED RISKS (REPORTED PER DIRECTIVE)

1. **Crop Refiner Validation Selector:**
   `v8_1/2_train_crop_refiner.py` trains for 10 epochs on train-fold instances and saves the final checkpoint. There is currently no separate validation set or PQ-based checkpoint selector for the crop refiner; it trains to completion on BCE+Dice loss. This is reported honestly per the directive and should not be artificially fabricated without empirical validation.
2. **Distribution Shift (GT Box Training vs YOLO Proposal Inference):**
   The crop refiner is trained on clean ground truth bounding boxes (`--boxes-from-gt`), whereas during inference it receives bounding boxes proposed by YOLO11s-seg (which may have loose borders or false positive proposals). This distribution shift is known and documented for subsequent optimization.
3. **Kaggle Kernel Wall Time:**
   20 epochs of YOLO11s-seg @ 1024 with batch 2 on a single T4 takes ~25–35 minutes; 10 epochs of Crop U-Net takes ~15–20 minutes; inference on 180 test disks with 4-flip TTA takes ~4–6 minutes. Total runtime is ~45–60 minutes, comfortably within Kaggle's 9-hour limit.

---

## 12. GATE REQUEST

**Verdict:** All correctness checks, stale pattern scans, memory optimizations, and artifact integrity verifications have passed with zero fatal bugs remaining.

**Requested Gate:** `MASTER_GO_KAGGLE` (Pending final confirmation from Master Reviewer ChatGPT). No Kaggle execution has been started.
