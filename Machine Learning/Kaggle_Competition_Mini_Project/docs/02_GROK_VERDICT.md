# GROK VERDICT — PLAN.md Phase A
# Status: APPROVED WITH AMENDMENTS
# Paste this entire file into Antigravity as the next message.

You asked two questions. Answers first, then the changes you must make before writing training code.

---

## Answers to QUESTIONS_FOR_GROK

### 1. Empty-disk encoding: `PPP2` vs `PPPP4`
Do **not** hardcode either string.

Write `encode_mask(mask_hw_uint8) -> str` that always goes through pycocotools Fortran 3D encode. For an empty disk, encode a real `(2048, 2048)` zero array. If that yields `PPPP4`, good — that is the correct counts string for this resolution, not a magic token.

Rules:
- Never emit `PPP2`. That was a dummy template, not a host spec.
- Never type the literal `"PPPP4"` in inference code. Call the encoder.
- Download official `sample_submission.csv` from Kaggle **before** you freeze row policy.
- Default row policy until sample_submission says otherwise:
  - one row per predicted instance
  - if a test disk has zero instances, emit exactly one row `{stem}_1` + encoded zero mask
- Add a unit test: encode zeros → decode → `mask.sum() == 0` and counts is a non-empty string.

### 2. YOLO size
Start **`YOLO11s-seg` @ 1024** for the first trained checkpoint (fast feedback, less T4 OOM).
Keep the config flag to bump to **`YOLO11m-seg` @ 1280** only after:
- smoke_cascade passes
- s-model holdout PQ is measured
- VRAM headroom on one T4 is confirmed

Do not begin with `n`. Too weak on thin 2k filaments. Do not begin with `m` until s works.

---

## What is approved
- Pivot off the 14-model ensemble. Correct.
- Kirillov PQ scorer first. Correct.
- Annotator-separated COCO→YOLO, no OR-merge. Correct.
- Lazy crop dataset. Correct.
- Reuse Fortran RLE, disk mask, CLAHE+unsharp, year GroupKFold, no DataParallel. Correct.
- New tree under `v8_1/` instead of mutating broken V8 notebooks. Correct.
- Identify V8 bug `if len(candidates) == 0`. Correct that this muted the residual branch. How you replace it is **not** yet approved as written.

## What is rejected or constrained

### A. "True Union Fusion" is too sloppy. Rename and gate it.
You will **not** OR / union YOLO masks with a full-disk semantic mask.

Residual branch rules:
1. Run **one** existing Unet++ ckpt full-disk, threshold, connectedComponents.
2. A residual component becomes a candidate **only if** its IoU with every YOLO instance is `< 0.30`.
3. Residual candidates go through the **same crop refiner + 4-flip TTA**. They do not skip refinement.
4. Final assembly is **greedy non-overlap by confidence**. Higher score keeps pixels. Loser is dropped or trimmed. Never fused into one instance.
5. Mandatory ablation on the year-holdout:
   - YOLO + refiner only
   - YOLO + refiner + residual
   Ship whichever has **higher PQ**. If residual hurts RQ, it is off in the submission.

Change the name in code and docs to `residual_proposals`. "Union" will make you merge instances again.

### B. Crop 256 is too small for long filaments
Mean bbox area is ~15k–17k px. Many spines are hundreds of pixels long. A fixed 256 crop with 20% pad **clips** the thing PQ needs.

Change:
- `side = min(512, max(256, int(1.2 * max(box_w, box_h))))`
- letterbox the crop to that square, keep scale + offset so paste-back is exact
- default train crop **384**, infer adaptive 256–512
- assert paste-back roundtrip on 3 long synthetic filaments (aspect > 8:1)

### C. 707 JPEGs vs expected unique count
JSON has 1,154 image records / 8,199 anns / 707 JPEGs. Ratio ~1.63 copies/file matches MAGFiLO. Unique-train expectation from 958−180 is **~778**. You are **71 files short** or they live in another folder.

Before training:
- print how many JSON `file_name`s do not exist on disk
- print how many JPEGs have zero annotations
- do not silently drop missing files
- drop `category_id == 4` (Ambiguous) from YOLO labels

### D. Do not write the Kaggle notebook in the same breath as the scorer
Phase B order is frozen:

1. `metrics/pq.py` + `tests/test_pq.py` + `tests/test_rle_roundtrip.py`
2. Download / locate `sample_submission.csv`; document its empty-row policy in `BUILD_REPORT.md`
3. `scripts/convert_coco_to_yolo.py` + missing-file report
4. `scripts/smoke_cascade.py --n 8` using **untrained or random-init weights** so the plumbing is proven on CPU
5. Only then `v8_1/1_train_yolo.py` and `2_train_crop_refiner.py`
6. Notebook last, and `json.load` must succeed

Local is CPU-only. Training scripts must detect CUDA and **refuse to start a full train on CPU** with a clear message. Smoke on CPU is fine with 8 disks and `imgsz=256` stub weights.

### E. Compute budget is a hope, not a contract
75–90 min for YOLO11m @ 1280 batch 4 on one T4 is optimistic. First Kaggle run is s @ 1024, 20 epochs, early stop on mask mAP. If OOM: drop to 960 or batch 2 accum 4.

### F. Forbidden still stands
- No 14-model logit average
- No training on full Dataverse MAGFiLO
- No `nn.DataParallel`
- No hardcoded LB scores
- No claiming D8 TTA if you only do 4 flips
- No new features until the four verification commands pass

---

## Phase B prompt (do this now)

Implement amendments A–F. Do not re-litigate architecture.

First commit that I will accept:
- `metrics/pq.py`
- `tests/test_pq.py` (perfect, shifted, merged pair, fragmented pair, empty-empty, empty-vs-blob)
- `tests/test_rle_roundtrip.py` (nonzero mask + all-zero 2048 mask)
- `scripts/smoke_cascade.py` that runs on CPU with stub/random weights on 8 disks and writes a tiny CSV

Then stop and return the four-section footer plus the smoke stdout. Do not start YOLO training until I say so.

## PLAN_OR_DIFF_SUMMARY / FILES_TOUCHED / VERIFICATION / QUESTIONS_FOR_GROK
Use that footer. If you have no questions, write `QUESTIONS_FOR_GROK: none`.
