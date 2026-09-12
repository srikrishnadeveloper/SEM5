# 06_TEST_RESULTS.md — Test Suite Audit & Metric Verification
**Project:** Solar Filament Segmentation Challenge 2026  
**Test Runner:** Pytest 9.1.1 on Python 3.13.3 (Windows 11)  
**Audit Date:** September 8, 2026  
**Summary:** **53 Passed, 1 Failed, 0 Skipped (98.1% Pass Rate)**  

---

## 1. Test Suite Summary Table

| Test Module | Tests | Result | Focus Area |
| :--- | :--- | :--- | :--- |
| `tests/test_rle_roundtrip.py` | 4 | **4 Passed** | COCO Fortran RLE encode/decode consistency |
| `tests/test_host_pq_compat.py` | 1 | **1 Passed** | Exact parity with Host Self-Evaluation V6 |
| `tests/test_r3_sanitizer.py` | 3 | **3 Passed** | Pairwise overlap detection & greedy carving |
| `tests/test_submission_contract.py`| 1 | **1 Passed** | Non-empty rows, correct columns, valid RLE |
| `tests/test_dataset_split.py` | 2 | **2 Passed** | GroupKFold physical filename split isolation |
| `tests/test_pq.py` | 7 | **7 Passed** | Kirillov PQ edge cases (empty vs empty, blobs) |
| `tests/test_moonshot_group_split.py`| 3 | **3 Passed** | Zero-leakage multi-annotator split verification |
| `tests/test_moonshot_matching.py` | 5 | **5 Passed** | IoU > 0.50 matching, candidate features |
| `tests/test_moonshot_submission.py`| 4 | **4 Passed** | CSV auditor asserts zero overlap & positive area |
| `tests/test_moonshot_ensemble.py` | 4 | **4 Passed** | Instance clustering and greedy resolution |
| `tests/test_r2_sweep.py` | 5 | **5 Passed** | Area filtering, overlap trimming, PQ contracts |
| `tests/test_v8_1_geometry.py` | 8 | **8 Passed** | Adaptive square bounds & crop sizing |
| `tests/test_moonshot_fallback.py` | 4 | **4 Passed** | OOM exception propagation & deployment NMS |
| `tests/test_r1_2_verification.py` | 3 | **2 Passed, 1 Failed** | Legacy notebook config inspection |

---

## 2. Detailed Verification of Core Contracts

### A. Host Self-Evaluation V6 Compatibility (`test_host_pq_compat.py`)
- **Status:** **PASSED (100% agreement)**.
- Local greedy Kirillov Panoptic Quality implementation computes identical values to the official competition host evaluator on synthetic ground-truth and prediction pairs.

### B. Fortran RLE Roundtrip Integrity (`test_rle_roundtrip.py`)
- **Status:** **PASSED**.
- Confirmed that masks decoded from `segmentation_rle` reproduce exact original binary masks with zero loss or shape distortion (`(2048, 2048)`).

### C. Zero-Overlap Pixel Sanitizer (`test_r3_sanitizer.py`)
- **Status:** **PASSED**.
- Greedy carving algorithm (`m = m_sub[k] & ~occupied`) strictly guarantees 0 shared pixels across all instances on a disk.
- Dropped fragments below `min_area` are purged to prevent low-area noise artifacts.

### D. GroupKFold Split Isolation (`test_moonshot_group_split.py`)
- **Status:** **PASSED**.
- Confirmed zero cross-contamination between training and validation sets when grouping by physical filename stem.

---

## 3. Forensic Analysis of the Single Test Failure

- **Failing Test:** `tests/test_r1_2_verification.py::test_notebook_json_and_embedded_config`
- **Error Trace:**
  ```python
  > assert CascadeConfig.DEFAULT_CONF == 0.25, f"Expected 0.25, got {CascadeConfig.DEFAULT_CONF}"
  E AssertionError: Expected 0.25, got 0.2
  ```
- **Root Cause:**  
  This legacy test specifically inspects Cell 4 of the deprecated `V8_1_Kaggle_Production.ipynb` notebook, where `DEFAULT_CONF` was set to `0.20` instead of `0.25`.
- **Impact Assessment:**  
  **Zero impact on production.** The production target is `moonshot_fulldata.ipynb`, which does not use `CascadeConfig` and instead sweeps confidences across `[0.15, 0.20, 0.25, 0.30, 0.35]` on GPU. Training code was left unmodified per project instructions.
