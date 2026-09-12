# PASTE THIS INTO ANTIGRAVITY — GROK MASTER DIRECTIVE R2
# Do not retrain YOLO11m. Do not turn residual ON. Do not rewrite the cascade.

You are the executor. Grok is master. ChatGPT is inactive.
Work only on the **production V8.1 tree that scored 0.350** (notebook SHA 331ce02dc7c0defd73fc6a037b093e0c65a628588f27cd8af0c13118fba15d01), not the pre-R1 scripts.

## Goal of R2
One Kaggle GPU session that:
1. Computes **host-style val PQ / SQ / RQ / TP / FP / FN** on fold-0 physical val JPEGs using the **already-trained** YOLO + refiner weights from the 0.350 run.
2. Sweeps **inference knobs only** (no YOLO retrain).
3. Writes `submission.csv` from the **best val-PQ operating point**.
4. Returns a short report. Stop.

If the 0.350 weights are **not** on disk / in a Kaggle dataset, Stage A is: save them from the completed kernel output (`best.pt` + `crop_refiner_r34.pth`) into a Kaggle dataset, then run this notebook against that dataset. Do not retrain from scratch unless weights are gone.

## Stage A — Diagnose (must exist before any new training)
On the 128 unique val physical JPEGs, for **each annotator observation separately**, then report both `pq_mean` and `pq_max` across annotators (same protocol as `evaluate_val_pq`).

Print a table:

```
conf  min_area  yolo_mask_fallback  overlap_mode  PQ  SQ  RQ  TP  FP  FN  n_pred
```

Default cell of that table must reproduce the 0.350 settings:
`conf=0.25, min_area=100, yolo_mask_fallback=off, overlap_mode=trim`.

## Stage B — Inference-only search (cheap)
Grid (keep it small):

- `conf ∈ {0.10, 0.15, 0.20, 0.25, 0.35}`
- `min_area ∈ {50, 100, 200, 400}`
- `yolo_mask_fallback ∈ {0, 1}`
  - 1 means: if crop-refiner mask area < min_area, paste the YOLO-seg mask (resized to native 2048) instead of dropping the instance
- `overlap_mode ∈ {trim, allow}`
  - `trim` = current greedy pixel wipe
  - `allow` = keep full masks; only drop a candidate if mask-IoU with a higher-score instance > 0.5 (NMS), do **not** carve pixels

Pick the row that maximizes **pq_mean**. Freeze it. Do not average rows.

Also dump rows-per-val-image histogram vs GT-instance histogram. If we predict ~7 and GT is ~11, the winner should be a lower-conf / fallback-on row.

## Stage C — One test submission
Run Stage 3 inference on 180 test disks with the frozen knobs.
Contract:
- columns `filament_id,segmentation_rle`
- 0 rows on empty disks (keep the 0.350 policy)
- 180 unique stems **or fewer** (empty disks omitted)
- 0 invalid RLEs, 0 zero-area masks
- print n_rows, mean/p50/p90 rows/image, n_empty_stems, elapsed

Do **not** claim a leaderboard number.

## Explicitly forbidden in R2
- YOLO11m / more YOLO epochs
- Residual full-disk U-Net
- 14-model ensemble
- Training on full MAGFiLO Dataverse
- Changing crop geometry, letterbox, or cat-4 policy in this pass
- Fake letterbox “fixes” that are not ablated
- Documentation-only churn

## If and only if weights are missing
Retrain **exactly** the 0.350 config (YOLO11s-seg @ 1024, 20 ep, batch 2, device 0, refiner 20 ep, boxes-from-gt, ultralytics pinned) then do A–C in the same kernel. Still no m-model.

## Return footer

```
## PLAN_OR_DIFF_SUMMARY
## FILES_TOUCHED
## VAL_PQ_TABLE
## CHOSEN_OPERATING_POINT
## TEST_CSV_STATS
## QUESTIONS_FOR_GROK
```

Start by confirming weights exist. Then implement A. Stop after the val table if you cannot run test inference in the same session — paste the table to Grok first.
