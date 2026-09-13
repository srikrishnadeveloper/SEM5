# 08_DECISION_LOG.md — Authoritative Architectural Decision Log
**Project:** Solar Filament Segmentation Challenge 2026  
**Status Date:** September 8, 2026  

---

## Decision 01: Official Metric Pivot to Kirillov Panoptic Quality (Aug 7–12, 2026)
- **Context:** Early code evaluated pixel-wise Dice/IoU. On August 7, the competition host officially updated the scoring script to Panoptic Quality (PQ) at $IoU > 0.50$ and re-scored the leaderboard on August 12.
- **Decision:** Fully align all local loss functions, validation evaluators, and post-processors to Kirillov PQ ($IoU > 0.50$).
- **Status:** **FROZEN & RATIFIED**.

---

## Decision 02: Abandonment of Semantic Segmentation & Watershed (Aug 2026)
- **Context:** V1–V4 used semantic U-Net/DeepLab models at 512–1024 with watershed to split blobs. Scores peaked at 0.300.
- **Forensics:** Watershed oversegmented curvilinear filaments into dozens of fragments, destroying Recognition Quality (RQ).
- **Decision:** Shift exclusively to native instance segmentation architectures where instances are natively proposed and bounded.
- **Status:** **PERMANENTLY RETIRED**.

---

## Decision 03: V8.1 True Instance Cascade (Aug 2026)
- **Context:** To combine detection with local detail, V8.1 used YOLOv8 1024 to detect filaments and U-Net++ to refine $384\times 384$ crops.
- **Outcome:** Reached **0.350 Public LB**, establishing the first robust, host-compliant baseline.
- **Status:** **RETAINED AS SECONDARY BASELINE ANCHOR**.

---

## Decision 04: Promotion of Native 2048 Resolution (Sept 2026)
- **Context:** V8.1 crop refiner struggled on filaments longer than 384 pixels, and downsampling to 1024 lost faint fibrils.
- **Decision:** Train YOLOv8l-seg directly on uncompressed 2048×2048 full-disk images.
- **Outcome:** Fold-0 model scored **0.360 Public LB**, outperforming V8.1 in its first run.
- **Status:** **ADOPTED AS PRIMARY PRODUCTION ARCHITECTURE**.

---

## Decision 05: Rejection of Cross-Architecture Ensembling (Sept 7, 2026)
- **Context:** An ensemble merging Moonshot 2048 masks with V8.1 1024 masks scored **0.350 Public LB** (a drop from 0.360).
- **Forensics:** 1024px mask dilation degraded boundary alignment (SQ), and 146 uncorroborated V8.1 false positives inflated the denominator.
- **Decision:** Strictly prohibit ensembling 2048 native models with 1024 downsampled models.
- **Status:** **ACTIVE ENSEMBLE PROHIBITION**.

---

## Decision 06: RAM Caching (`cache="ram"`) for 12-Hour Budget (Sept 8, 2026)
- **Context:** Training 2048px images with disk I/O took ~13 minutes per epoch, risking the Kaggle 12-hour timeout.
- **Decision:** Pre-cache all 1,154 observations as PyTorch tensors in Kaggle's 30 GB RAM.
- **Outcome:** Reduced epoch time from 13 min to ~9 min, safely fitting a 50-epoch budget into ~8.5–9.5 hours.
- **Status:** **ACTIVE PRODUCTION FEATURE**.

---

## Decision 07: Checkpoint Fine-Tuning Strategy (Sept 8, 2026)
- **Context:** Choosing between starting fresh from base `yolov8l-seg.pt` vs fine-tuning from 60-epoch Fold-0 `best.pt`.
- **Ruling by ChatGPT Master:** Fine-tune from `best.pt` on 100% data with `mosaic=1.0` and `close_mosaic=10` for 50 epochs.
- **Rationale:** Preserves learned solar chromospheric morphology and domain representations while expanding to all 1,154 observations.
- **Status:** **APPROVED FOR EXECUTION**.

---

## Decision 08: Pure PyTorch GPU Acceleration (Sept 8, 2026)
- **Context:** Test inference with OpenCV/NumPy on CPU took 3–4 minutes for 180 images.
- **Decision:** Implement GPU tensor operations for limb masking, confidence thresholding, and greedy zero-overlap carving.
- **Outcome:** Full 180-disk inference runs in ~25–35 seconds in GPU VRAM.
- **Status:** **VERIFIED & FROZEN**.

---

## Decision 09: Pinned `batch=1` & Memory De-fragmentation (Sept 11, 2026)
- **Context:** Initial Kaggle run (`notebooke9a0244f4e`) crashed with `CUDA out of memory` at step 63/577 of Epoch 1 because `batch=2` + `mosaic=1.0` at 2048px peaked at 14.8 GB on a 14.56 GB Tesla T4.
- **Decision:** Pin `batch=1` with PyTorch AMP and set `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` in Cell 1.
- **Outcome:** Dropped peak VRAM to 8.71 GB (and 7.75 GB post-mosaic), providing 5.8 GB of safety headroom. 10.5+ hours of training executed with zero OOM.
- **Status:** **RATIFIED PRODUCTION STANDARD**.

---

## Decision 10: Premature Early-Stopping Mitigation (`patience=50`) (Sept 11, 2026)
- **Context:** `patience=15` risked terminating training prematurely on the 5-image duplicate validation loop if validation loss fluctuated.
- **Decision:** Increase `patience=50` to guarantee all 50 full epochs execute across the 100% data regime.
- **Outcome:** Model successfully traversed epochs 1 through 49 without premature halt.
- **Status:** **ACTIVE PRODUCTION SETTING**.

---

## Decision 11: Fail-Safe Inference Object Pre-definition (Sept 11, 2026)
- **Context:** If training was interrupted, downstream cells crashed with `NameError: name 'WEIGHTS' is not defined` or `NameError: name 'rows' is not defined`.
- **Decision:** Pre-define fallback pointers for `WEIGHTS` and initialize `rows` to the primary frozen submission record.
- **Outcome:** Cells 6, 7, and 8 are completely fail-safe.
- **Status:** **PERMANENT STANDARD**.

---

## Decision 12: Permanent Ban on Mosaic Augmentation for Solar Chromosphere Disks (Sept 11, 2026)
- **Context:** Full-data YOLOv8l-seg fine-tuned with `mosaic=1.0` scored 0.330 – 0.340 on the official Kaggle leaderboard, regressing from our 0.360 Fold-0 baseline.
- **Forensics:**
  1. Mosaic 4-quadrant stitching physically bisects circular solar disks with artificial crosshair seams, training the segmentation prototype head to expect artificial vertical/horizontal filament terminations.
  2. Fine-tuning for 50 additional epochs on top of 60 epochs (110 total) caused prototype logit drift and an +8.7% mask dilation, dropping borderline IoUs below the strict 0.50 cutoff.
- **Decision:**
  1. **Strictly prohibit `mosaic > 0` on solar filament segmentation.** All future models must train exclusively on intact, natural circular disks (`mosaic=0.0`).
  2. **Re-affirm the 60-epoch Fold-0 model (`models/moonshot_2048/best.pt`, 0.360 LB)** as the project's primary verified champion.
- **Status:** **RATIFIED ARCHITECTURAL LAW**.

---

## Decision 13: Red-Team Forensic Audit — Empirical Barrier Documentation (Sept 13, 2026)
- **Context:** Antigravity deployed a 5-agent red-team swarm audit to empirically diagnose all structural barriers to PQ improvement.
- **Key Findings:**
  1. Human-vs-human PQ on MAGFiLO multi-annotator pairs = **0.3329** → our 0.360 exceeds human agreement.
  2. YOLO 1/4 prototype resolution creates a hard SQ ceiling at ~0.92 mean IoU.
  3. Bbox fill ratio of 24.3% means 75.7% of YOLO features are contaminated by background.
  4. Pixel-carve zero-overlap sanitizer fragments 6.9% of masks into up to 6 disconnected pieces.
  5. Binary mask ensembling injected +141 FPs (0.360 → 0.350 regression).
  6. Full-data fine-tuning with mosaic=1.0 missed 31% of the filaments detected by the 0.360 model.
- **Decision:**
  1. All 6 findings are documented in `master/10_RED_TEAM_FORENSIC_AUDIT.md` and referenced in `00_EXECUTIVE_SUMMARY.md`.
  2. Post-carve connected component cleanup is identified as a free PQ recovery lever (estimated +0.01–0.02 PQ).
  3. Comprehensive ChatGPT Master research prompt issued in `master/MASTER_PROMPT_SEPT13.md`.
- **Status:** **DOCUMENTED — AWAITING MASTER STRATEGIC RESPONSE**.

