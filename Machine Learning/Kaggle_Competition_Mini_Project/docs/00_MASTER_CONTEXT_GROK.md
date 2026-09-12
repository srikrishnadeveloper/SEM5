# CURRENT AUTHORITY — 2026-09-05 (evening)

**Master / red-team: Grok.** Executor: Antigravity. Human: copy/paste bridge.
ChatGPT is **inactive**. Do not follow `00_MASTER_CONTEXT_CHATGPT.md` when it conflicts with this file.

## Scoreboard (do not inflate)

| Run | PQ | Status |
|---|---|---|
| Semantic ensembles v1–v4 | 0.22 → 0.26 → **0.30** | Verified historically |
| V8.1 YOLO11s-seg → crop U-Net | **0.350 claimed** | User-reported public LB, 1231 rows, 178/180 disks |
| Public notebook titles | 0.55 / 0.60 | **UNVERIFIED** (JS LB pages did not render here) |
| Ghosh two-stage paper | PQ 0.4145 | Off-Kaggle MAGFiLO 1.0 |

0.350 is a real instance-pipeline win over 0.30. It is **not** SOTA. Do not write “breakthrough / superior to public benchmarks.” Target remains **>0.40**, then the 0.55 cluster.

## Frozen architecture (do not revert)

YOLO11s-seg @ 1024 → instance boxes → adaptive square crop 256–512 → ResNet-34 U-Net @ 384 → 4-flip crop TTA → score-ordered assembly → pycocotools Fortran RLE.

- Residual full-disk semantic branch: **OFF** until a val-PQ ablation says otherwise.
- No 14-model logit ensemble.
- No training on full Dataverse MAGFiLO (leakage: 180 test disks are in the public release).
- No `nn.DataParallel`. YOLO train: single `cuda:0`.

## Host facts we will not relitigate without new evidence

- LB metric = Panoptic Quality, Kirillov match **IoU > 0.5**, unique 1–1. Aug 7 change, Aug 12 rescore.
- CSV: `filament_id,segmentation_rle`. One row per **actual** instance.
- Empty disks: V8.1 shipped **0 rows** and scored 0.350. Do not go back to dummy `PPP2` or extra zero-mask FPs unless `sample_submission.csv` contradicts.
- Categories 1–4 are all filaments for this **single-class** task. Include Ambiguous (id=4) as class 0. Chirality is not scored.
- Images: 2048×2048 grayscale GONG H-α. Train JSON: 8199 anns / 1154 image records / ~707 unique JPEGs.

## What 0.350 actually says

1231 predictions / 180 test disks ≈ **6.84 / disk**. Hidden test has ~2045 MAGFiLO annotations on those 180 disks (~11.4 if all annotator copies are GT). Either:

- host GT uses one annotator → we are **under-detecting** (~40% FN mass), or
- host GT keeps copies → unique matching creates a PQ noise floor.

R2 must print val **PQ, SQ, RQ, TP, FP, FN** before any new architecture. If RQ ≪ SQ, chase recall (conf, YOLO-mask fallback). If SQ is the hole, chase refiner / crop geometry.

## Stale files in this chat

`/workspace/attachments/{config,1_train_yolo,2_train_crop_refiner,3_infer_cascade}.py` are **pre-R1**. They still drop cat-4, BATCH=16, fake letterbox, empty-row zero-masks. Production is the notebook SHA `331ce02d…` that scored 0.350. Antigravity must edit **that** tree, not these attachments.
