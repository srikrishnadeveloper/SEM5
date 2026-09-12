# PASTE THIS INTO ANTIGRAVITY
# Mode: Planning first. Do not write training code until the plan artifact is approved.
# Attach 00_SHARED_CONTEXT.md in the same workspace (or paste it under this prompt).
# Also attach / keep in the repo: AGENTS.md, LESSONS_LEARNED.md, V7/V8 .py files, sample_submission.csv if present.

---

You are the **builder agent**. Grok is the **master reviewer**. I will paste your plan and later your code/diffs back to Grok. Stay aligned with `00_SHARED_CONTEXT.md`. If anything in the repo contradicts that file, **trust the shared context** and flag the contradiction.

## Mission
Build a **working instance-segmentation pipeline** for Kaggle `filament-segmentation-2026` that can beat **public LB 0.40**. Do not chase a 14-model semantic ensemble. That path is proven stuck at ~0.30 PQ.

Success for this assignment is NOT "train 14 models". Success is:

1. A local Kirillov PQ scorer that reports PQ / SQ / RQ / TP / FP / FN.
2. A two-stage instance pipeline that runs end-to-end on a small slice of train data.
3. A Kaggle-ready inference notebook or script that writes a valid `submission.csv`.
4. A short BUILD_REPORT.md Grok can review.

## Hard constraints
- Metric is Panoptic Quality (IoU > 0.5 unique match). Dice is a debug number only.
- CSV: columns `filament_id,segmentation_rle`. One row per instance. Fortran COCO RLE. Confirm empty-disk encoding from `sample_submission.csv` (do not assume `PPP2`).
- Train only on Kaggle train IDs (8,199 annotations). Never the full 10,244 MAGFiLO release.
- Images are 2048×2048 grayscale GONG H-α.
- Target hardware: Kaggle Dual T4 or one Colab T4/L4. No `nn.DataParallel` in notebooks.
- Do not rewrite the whole repo. Reuse RLE, disk mask, CLAHE/unsharp, GroupKFold if they exist and are correct.
- Do not claim D8 TTA if you only implement 4 flips.
- Do not invent leaderboard scores.

## What to build (this is the architecture — do not substitute a full-disk Unet++ ensemble)

### Stage 0 — Truth and scorer (must exist before any new training)
- Read `sample_submission.csv` and one train annotation JSON. Document exact RLE encode/decode.
- Implement `pq_score(pred_instances, gt_instances) -> dict` with greedy unique matching IoU>0.5.
- Unit test on synthetic masks: perfect copy, shifted copy, merged pair, fragmented pair, empty vs empty, empty vs one blob.
- Decode our latest existing `submission.csv` if present. Print rows-per-image histogram. That tells us merge vs fragment vs semantic-union.

### Stage 1 — Instance proposer
Preferred, in this order (pick the first that fits installed packages + VRAM):
1. YOLO11-seg or YOLO11m-seg @ 1280 (letterbox, disk-cropped if helpful).
2. YOLO11m detect @ 1280 + axis-aligned boxes.
3. Mask R-CNN / torchvision only if ultralytics cannot be installed.

Convert MAGFiLO COCO polygons → YOLO-seg labels. One annotation = one instance. For multi-annotated disks, treat each annotator file as a separate sample. Do not OR masks.

### Stage 2 — Crop refiner
- For each box: pad ~20%, crop, resize to 256 or 384, 3-channel stack = raw + CLAHE + unsharp (reuse existing code).
- Small Unet / Unet++ (resnet34 or efficientnet-b3, not b4-14-ensemble).
- 4-flip TTA on the crop only.
- Paste refined mask back onto the 2048 canvas. Resolve overlaps with higher-score wins, do not union them into one instance.

### Stage 3 — PQ post
Sweep on a year-grouped holdout (not random images):
- score / conf threshold
- min_area in {150, 250, 400, 600}
- optional watershed split only inside a crop that contains two obvious lobes
- do **not** merge neighbors unless box IoU is high AND there is a thick neck; MAGFiLO often wants them separate

Pick the operating point that maximizes **PQ**, not Dice.

### Stage 4 — Inference + submission
- Script/notebook that reads official test images, writes `/kaggle/working/submission.csv`.
- Fallback if YOLO returns zero boxes on a disk: run one full-disk semantic model (ONE of the existing ckpts, not 14) + connectedComponents + min_area. Log how often fallback fires.
- Print: n_test_images, n_rows, rows-per-image mean/p50/p90, n_empty, elapsed time.

## Process (Antigravity rules)
Follow Google Antigravity practice: explore → plan artifact → wait → implement → verify.

### Phase A — Explore only. Do not edit files.
Inspect the workspace. List:
- which V7/V8 `.py` files exist and whether they import
- whether train images + COCO JSON are on disk
- whether ultralytics / smp / pycocotools are installed
- whether `sample_submission.csv` exists
- GPU / VRAM

Write `PLAN.md` with:
- chosen architecture (1, 2, or 3 from Stage 1)
- file-level change list
- what you will reuse vs rewrite
- compute budget (hours on T4)
- verification commands
- risks (empty RLE, leakage, VRAM, corrupt notebooks)

STOP and wait for me to paste PLAN.md to Grok. Do not implement yet.

### Phase B — After I say "approved"
Implement in this order, committing logically:
1. `metrics/pq.py` + tests
2. dataset conversion COCO → YOLO
3. train scripts (small smoke first: 20 images, 1 epoch)
4. infer cascade
5. `BUILD_REPORT.md`

After every step run the verification commands from PLAN.md. If a test fails, fix that test before adding features.

### Phase C — Smoke contract (must pass)
```
python -m py_compile $(git ls-files '*.py' || find . -name '*.py')
python tests/test_pq.py
python tests/test_rle_roundtrip.py
python scripts/smoke_cascade.py --n 8
```
`smoke_cascade.py` must print PQ/SQ/RQ on 8 real or synthetic disks and write a tiny CSV Grok can inspect.

## Forbidden
- Another 14-model Unet++ / DeepLab logit ensemble as the scoring path
- Training on Dataverse full MAGFiLO
- `nn.DataParallel`
- Fake scores, fake checkpoints, placeholder RLE like `"1 1"`
- Editing files during Phase A
- Huge refactors of unused v1–v4 notebooks
- Claiming "SOTA" or "0.55 guaranteed"

## What to return to me (so I can paste to Grok)
Always end with these four sections, nothing else at the bottom:

```
## PLAN_OR_DIFF_SUMMARY
(bullets)

## FILES_TOUCHED
(path — purpose)

## VERIFICATION
(commands + pass/fail)

## QUESTIONS_FOR_GROK
(only blockers)
```

If you are in Phase A, PLAN.md is the artifact. If Phase B, include the diff stat and the smoke output.

Start now with Phase A.
