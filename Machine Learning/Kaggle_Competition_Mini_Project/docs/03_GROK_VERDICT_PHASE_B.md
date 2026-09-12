# GROK VERDICT — Phase B first commit
# Status: ACCEPT PLUMBING. DO NOT TREAT SMOKE PQ AS REAL.
# Paste this entire file into Antigravity.

The four files compiled and the unit tests match Kirillov arithmetic. That commit is accepted.

The smoke run is **not** accepted as evidence the cascade works on MAGFiLO.

---

## What passed (keep it)
- `test_pq.py` 7/7. Merged = TP0 FP1 FN2. Fragmented = TP0 FP2 FN1. Empty-empty = PQ 1. Correct.
- Zero 2048 mask encodes through pycocotools and comes back as `PPPP4` with decode sum 0. No `PPP2`. Correct.
- Adaptive crop formula and residual IoU<0.30 + greedy non-overlap are in the smoke script. Good.
- You stopped before training. Good.

## What failed review

Smoke used **8 synthetic disks**, not the 707 real JPEGs sitting in `data/`.
Reported `TP=16, FP=29, FN=0`, mean_PQ=0.5113.

FN=0 with random/stub weights means GT was built from the proposals (or the stub is guaranteed to cover GT). That only tests "CSV writer runs." It does not test:

- reading a real GONG JPEG
- rasterizing a COCO polygon
- solar-disk mask on a real limb
- `filament_id` stems from real filenames
- JSON `file_name` that does not exist on disk

Do not quote 0.5113 anywhere as model quality.

---

## Next commit (this is the only work you may do now)

Do these three things, then stop again.

### 1. Official sample submission
Download or copy `sample_submission.csv` from the Kaggle competition data.
Print: columns, n_rows, n_unique stems, how empty disks are represented (missing row vs zero-RLE vs other).
Write that into `BUILD_REPORT.md` under "Submission contract."
Row policy stays: encode via `encode_mask()`, never hardcode `PPPP4`.

### 2. `scripts/convert_coco_to_yolo.py`
- Key off JSON `image_id`, not pooled filename.
- One annotator observation = one YOLO label file / sample. No polygon OR.
- Drop `category_id == 4`.
- Single class `filament` (id 0) for YOLO-seg. Chirality is not scored.
- Before writing labels, print:
  - JSON images
  - JPEGs on disk
  - `file_name` missing on disk (the 778-vs-707 gap)
  - JPEGs with zero annotations
  - polygons skipped (too few points, zero area)
- Exit non-zero if more than 5% of JSON `file_name`s are missing, unless `--allow-missing`.
- Write `data/yolo_seg/data.yaml` and a `conversion_report.json`.

### 3. Real-image smoke (replace synthetic default)
Change `scripts/smoke_cascade.py` so `--n 8` loads **8 real train JPEGs + their COCO instances**.

Stub proposer may stay random/heuristic. GT must be real polygons rasterized at 2048.

Print per disk:
- stem
- n_gt
- n_pred
- PQ SQ RQ TP FP FN

Then totals. FN will not be 0. If it is 0 on real GT with stub weights, you are still leaking pred into GT — fix that.

Also write `submissions/smoke_real8.csv` and print rows-per-image.

Still CPU. Still no YOLO training script. Still no notebook.

---

## Forbidden on this commit
- `v8_1/1_train_yolo.py` or any ultralytics train call
- Installing a long train run
- Quoting synthetic PQ as a baseline in PLAN.md / README
- Unioning residual + YOLO masks

## Footer when done
PLAN_OR_DIFF_SUMMARY
FILES_TOUCHED
VERIFICATION (include the 8-line per-disk PQ table and the missing-file counts)
QUESTIONS_FOR_GROK
