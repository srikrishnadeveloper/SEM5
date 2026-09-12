# GROK MASTER — R3 HOST-SAFE TRIM SUBMIT
Date: 2026-09-06

## What happened

R2 ran correctly. Weights from `notebooka464bcfe13` were used. Training skipped.
The kernel then submitted `overlap_mode=allow`.

Host rejected it:

> Invalid Submission! Submissions may not contain overlapping masks.
> That is, predictions of the same filament may not overlap.

`allow` keeps two masks if instance IoU ≤ 0.5. Touching filaments still share pixels. Host requires **zero shared pixels** per disk.

The 0.350 green submission is still valid. This error does not erase it.

## VAL_PQ_TABLE — what actually mattered

| Mode | Best val PQ_mean | Notes |
|---|---|---|
| allow (illegal) | 0.4596 at conf=0.20, min_area=50 | RQ jump from keeping overlaps |
| trim (legal) | 0.4389 at conf=0.20 or 0.25, min_area=400 | Realizable |
| 0.350 baseline trim | 0.4357 at conf=0.25, min_area=100 | Already on public LB 0.350 |

- `fallback=1` matched `fallback=0` on almost every row. YOLO-mask fallback is dead. Ignore it.
- allow − trim ≈ +0.02 PQ is **not available** on the host.
- Best legal minus baseline = 0.0032 < 0.005. Do not overclaim. Still submit the legal winner once so we get a new LB number.

## Frozen submit knobs (R3)

```
conf = 0.20
min_area = 400
yolo_fallback = 0
overlap_mode = trim
tta = True
```

Plus a **mandatory host sanitizer** after arbitration, before RLE:

1. Sort accepted masks by confidence desc (then area desc).
2. Greedy pixel carve: `mask[occupied > 0] = 0`.
3. Drop if area < min_area after carve.
4. Per disk, assert `sum(mask_i & mask_j) == 0` for all i != j. Fail the notebook if any overlap remains.

Do **not** rerun Stage A or the 80-point sweep. That wasted 4 hours on CPU decode. Stage C only (~3 min).

## Notebook contract

- Attach the same two inputs: competition data + `notebooka464bcfe13`.
- Cells 6/7 must skip train again.
- Replace Stage C call with explicit:

```
python v8_1/3_infer_cascade.py \
  --conf 0.20 --min-area 400 --yolo-fallback 0 \
  --overlap-mode trim --tta \
  --weights <resolved best.pt> \
  --refiner-weights <resolved crop_refiner_r34.pth> \
  --out /kaggle/working/submission.csv
```

- Audit cell must decode every stem group and prove zero pairwise overlap.

## Forbidden

- overlap_mode=allow in any file that writes submission.csv
- Retrain
- YOLO11m / residual U-Net
- Another 80-point sweep

## After the run, paste to Grok

1. Confirm train skip
2. Confirm overlap assert passed
3. CSV: n_rows, mean/p50/p90, empty stems
4. Public LB number if it scores
