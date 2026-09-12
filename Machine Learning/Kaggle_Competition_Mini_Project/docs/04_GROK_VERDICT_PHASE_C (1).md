# GROK VERDICT — Phase B second commit
# Status: ACCEPT. 778-FILE GAP CLOSED. PHASE C SCRIPTS APPROVED WITH FROZEN SPECS.
# Paste this entire file into Antigravity.

Plumbing on real MAGFiLO is accepted. Synthetic 0.51 is dead. Stub PQ = 0 with FN = 28 is the honest result we asked for.

---

## What this commit proved
- Host data pack is 707 train JPEG + 180 test JPEG + 1 JSON. No official `sample_submission.csv`. Your 180-row empty baseline is a valid stand-in, not a host file. Label it that way in BUILD_REPORT.
- Empty disk = one row `{stem}_1` + `encode_mask(zeros)`. Keep that.
- JSON `file_name` missing on disk = **0**. The old 958−180≈778 guess was wrong. Kaggle unique train is **707**. 707+180=887. The other ~71 MAGFiLO uniques are not in this Kaggle pack. Stop looking for them.
- 1154 annotator observations / 707 files / 8199 polygons. Converter keyed by `image_id`, no OR, class 0 only. Correct.
- Year GroupKFold did not split a physical JPEG across train/val (579+128=707). Keep that invariant. Add a unit test that fails if any `file_name` is in both splits.
- Real smoke: 8 real stems, real `cv2.fillPoly` GT, stub preds, all 35 RLEs decode to 2048×2048, FN≠0. Accepted.

## Answers

### 1. YOLO size / resolution
**`yolo11s-seg` @ 1024. Not 640. Not n. Not m.**

640 erases Unidentifiable filaments (~1.6k px). n is too weak. m waits until s has holdout PQ printed.

First train config (frozen):
- model: `yolo11s-seg.pt`
- imgsz: 1024
- epochs: 20
- batch: 4, accum: 2 (drop batch to 2 if T4 OOM)
- amp: True
- mosaic: 0.5, close_mosaic: last 2 epochs
- degrees / flipud / fliplr allowed; do not use copy-paste until s baseline exists
- workers: 2 on Kaggle
- project: `runs/v8_1_yolo_s1024`
- Refuse to start if `not torch.cuda.is_available()`

Optional 1-epoch subset smoke (`--max-train 32`) may run on CUDA only.

### 2. Multi-annotator val PQ
Never score a physical JPEG against only `img_ids[0]`.

For each val **file_name**:
- rasterize each annotator observation separately
- compute Kirillov PQ vs each
- record `pq_mean` and `pq_max`

Sweep thresholds on **`pq_mean`**. Report both in the log.
Do not average PQ across duplicate YOLO val samples of the same JPEG — that triple-counts busy disks. Loop unique `file_name`s.

Test-set inference is single-GT; local mean is the conservative selector.

### 3. Approval for `v8_1/`
Yes. Write the three scripts. Do **not** launch a full train until the scripts exist, compile, and the YOLO script aborts cleanly on CPU.

---

## Phase C file contract

```
v8_1/config.py
v8_1/1_train_yolo.py
v8_1/2_train_crop_refiner.py
v8_1/3_infer_cascade.py
```

Frozen behavior:

**1_train_yolo.py**
- Reads `data/yolo_seg/data.yaml`
- Spec above
- Writes `runs/v8_1_yolo_s1024/weights/best.pt`
- After train (or `--eval-only`): run unique-file val PQ at conf in {0.15,0.25,0.35} and print a table. This can wait until weights exist.

**2_train_crop_refiner.py**
- Train only on train-fold instances (the 579 files / 951 observations).
- Crop side `min(512, max(256, int(1.2*max(w,h))))`, letterbox to 384 for the batch.
- 3-ch raw+CLAHE+unsharp
- Unet resnet34, 1 class, AMP
- Do not start this train until YOLO `best.pt` exists unless `--boxes-from-gt` (allowed for a parallel GT-crop pretrain)

**3_infer_cascade.py**
- YOLO boxes/masks → pad → crop refine → residual proposals with IoU<0.30 vs every YOLO instance → greedy score non-overlap
- Residual OFF by default (`--residual 0`). Turn on only after an ablation prints residual-on vs residual-off `pq_mean`
- Adaptive crop 256–512
- 4-flip TTA on crops only
- Empty disk: one encoded-zero row
- Writes `submissions/v8_1_cascade.csv`
- Prints n_images, n_rows, rows/image mean-p50-p90, n_empty, seconds

Still forbidden: 14-model average, DataParallel, hardcoded PPP2/PPPP4, training on Dataverse-full, claiming a LB score.

---

## What you return
Implement the three scripts + `config.py`. Compile them. Run:

```
python v8_1/1_train_yolo.py --help
python v8_1/1_train_yolo.py --dry-run
```

`--dry-run` must print the frozen config and exit 0 on CPU without training.

Then stop. Do not start a 20-epoch run until I see that dry-run output.

Footer: PLAN_OR_DIFF_SUMMARY / FILES_TOUCHED / VERIFICATION / QUESTIONS_FOR_GROK
