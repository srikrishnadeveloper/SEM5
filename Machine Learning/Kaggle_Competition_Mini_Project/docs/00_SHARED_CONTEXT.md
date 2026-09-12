# SHARED CONTEXT — Solar Filament Segmentation Challenge 2026
# Master: Grok. Builder: Antigravity. Human pastes builder output back to Grok.
# Last aligned: 2026-09-05. Do not invent facts that contradict this file.

## Goal
Public LB **> 0.40 first**, then chase the **0.55 cluster**. Metric is **Panoptic Quality**, not Dice.

## Competition
- Kaggle: https://www.kaggle.com/competitions/filament-segmentation-2026
- IEEE BigData Cup 2026 Challenge 02, Phoenix 14–17 Dec 2026
- Prize: up to $3,000 (NSO). Source code + report required for cup money.
- Deadline recovered from API: **15 Nov 2026**. Confirm on Kaggle.
- Live field (early Sep 2026 API): **~508 teams**. Public top cluster **~0.55–0.56**. Our last verified rank **#298 @ ~0.30**.
- Official LB metric name: `panoptic_quality`.

## What is scored
Kirillov PQ, unique match if IoU **> 0.5**:

PQ = SQ × RQ
SQ = mean IoU of matched instances
RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)

- Merged neighbors → usually 1 FP + 2 FN. Dice can still look fine.
- Fragments → 1 FN + many FPs.
- One RLE per whole disk (semantic union) is catastrophic RQ.
- Dice 0.84 can still be PQ 0.41 (Ghosh two-stage paper on MAGFiLO 1.0).
- IEEE committee also cares about MIoU / barbs. Do **not** optimize MIoU before RQ is healthy.

## Submission format (do not invent)
CSV columns:
- `filament_id` like `{GONG_STEM}_{k}` e.g. `20150125172714Mh_2`
- `segmentation_rle` = COCO RLE **counts** string

Rules that have bitten this team:
- One **row per predicted instance**, not one row per image.
- RLE must be **Fortran / column-major**. Use:
  `mask_utils.encode(np.asfortranarray(mask.astype(np.uint8).reshape(h, w, 1)))[0]["counts"]`
- Empty disk: team code used sentinel `"PPP2"`. **Verify against `sample_submission.csv` before shipping.** If sample uses a real zero-mask RLE, use that. Never emit dummy polygons.
- Chirality (Left/Right/Unidentifiable) is **not** a scored CSV column.

## Dataset — MAGFiLO v1.0 (Kaggle split)
- Full public MAGFiLO: 10,244 filaments, 1,593 files, **958 unique** disks, 2011–2022, 2048×2048 GONG H-α JPEG.
- Kaggle train: **8,199** annotations.
- Hidden test: **180 disks, ~2,045** annotations.
- Those 180 test disks **exist in the public Dataverse**. Training on full MAGFiLO = leakage. Use only Kaggle `train/` IDs.
- Multi-annotator: 548 once / 185 twice / 225 thrice. Cohen κ ≈ 0.66 Left/Right. PQ has a noise floor. Do **not** OR multi-annotator copies into one union mask.
- Categories: Left=1, Right=2, Unidentifiable=3. Drop category_id=4 if present.
- Mean area: Left 2253 / Right 2416 / Unidentifiable **1683 px**. Small Unidentifiable filaments are the FN tail. Downsampling 2048→512 as the only scale kills them.
- Images are grayscale. Do not treat as RGB-meaning color.

Host discussion on split: Kaggle discussion 729396.

## Why this team is stuck at ~0.30
Verified submissions:
| Sub | Method | Instances | LB |
|---|---|---|---|
| 1 | Single U-Net R34 512px | ~250 | 0.22 |
| 2 | 5-fold Unet++ 512px loose | ~1200 | 0.26 |
| 3 | 9-model ens 1024px t=0.46 area=200 | 1760 | 0.30 |
| 4 | 14-model ens 1024px t=0.54 area=750 | ~650 | ≤0.30 |

Root causes (not "need more U-Nets"):
1. Semantic models + connectedComponents ≠ instance identity. Adjacent filaments merge.
2. 14-model logit average **glues** fragments and **blurs** barbs. Same inductive bias.
3. Val Dice 0.67–0.69 ≠ PQ. They optimized the wrong number.
4. High area cut (750) drops Unidentifiable filaments.
5. Possible empty-mask / RLE / 2D-vs-3D Fortran bugs historically.
6. V8 "seed-refine" exists as incomplete scripts (fold-0 only, Stage-1 not finished). Notebooks in the zip are corrupt JSON. Use `.py`.

Public evidence that 0.30 is **not** the ceiling:
- Public kernel title: "Solar Filament Unet Segmentation | 0.55+" (lamhuy8904).
- Ghosh two-stage (YOLO11m → crop seg): **PQ 0.4145, Dice 0.8433** off-Kaggle on MAGFiLO 1.0.
- Other public kernels: YOLO+U-Net (ektarr), Mask R-CNN + U-Net refiner (nomannic19), blended-tile native-res U-Net (anthonytherrien), MAGFiLO V9 evaluation-aligned (muhammadshoaibaltaf).

## What actually wins
Priority order:
1. **True instances** (YOLO11-seg, or YOLO-det → crop U-Net, or Mask R-CNN / Mask2Former).
2. **PQ-aware post**: min-area, merge/split, score threshold swept on local PQ not Dice.
3. Native / tiled high-res for SQ once RQ works.
4. EdgeAttNet / CLAHE / unsharp / disk mask as **crop refiner features**, not as the instance proposer.

Do **not** train another 14-model Unet++/DeepLab ensemble.

## Existing code the builder must reuse, not rewrite from scratch
Repo: srikrishnadeveloper/SolarFilamentSegmentationChallenge-2026
Zip "kaggle reduced to fit in the chatgpt.zip" is the real latest (V7/V8). Public GitHub README is marketing around v1–v4.

Keep / port:
- Fortran 3D RLE encoder
- Solar disk mask (Hough / R≈0.93)
- CLAHE + unsharp 3-channel stack
- GroupKFold by year / filename prefix
- Dual-T4 model-parallel inference pattern (never `nn.DataParallel` in notebooks)
- Existing 14 ckpts only as **crop refiners or fallback**, not as the LB engine

Throw away as the scoring path:
- Full-disk logit averaging of 14 semantic models
- Threshold 0.80 (zeros Dice)
- Training on full 10,244 MAGFiLO
- Downsample-only 256/512 full-disk as the only scale

## Hard engineering rules (from LESSONS_LEARNED)
- Fold indices are 0-based.
- `lr_scheduler.step()` after `optimizer.step()`.
- VRAM: `memory_reserved`, not just allocated. `del` + `empty_cache` between folds.
- Never `nn.DataParallel` in Jupyter. Alternate models on cuda:0 / cuda:1.
- Adaptive `in_channels` from checkpoint.
- `PYTHONUNBUFFERED=1`, console tqdm, `flush=True` on Kaggle.
- New Kaggle tokens are `KGAT_…`. Use `KAGGLE_API_TOKEN` + `~/.kaggle/access_token`, not only legacy kaggle.json.
- Do not put secrets in committed files.

## Realistic scoreboard
| PQ | Meaning |
|---|---|
| 0.22–0.30 | This team's semantic ensemble |
| 0.34 | Public U-Net notebook (one version) |
| 0.39–0.41 | Two-stage / Ghosh band |
| 0.45–0.52 | Dedicated instance + PQ post |
| 0.55–0.56 | Current public top cluster |
| ≫0.56 | Unlikely; κ=0.66 and annotator splits |

First ship target: **local val PQ ≥ 0.42 and a valid test CSV**. Then submit.
