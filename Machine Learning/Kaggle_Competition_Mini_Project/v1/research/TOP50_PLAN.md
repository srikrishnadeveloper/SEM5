# Solar Filament Segmentation — Top-50 Research & Battle Plan

> Compiled from 6 parallel subagent literature reviews (Aug 2026).  
> Use this as the master reference for all model / training / post-processing decisions.  
> **Last updated:** 2026-08-25.

---

## 1. TL;DR — what it takes to get near top 50

The friend’s U-Net++ EfficientNet-B4 5-fold ensemble is a **strong reference**, but not a silver bullet.  Research on the MAGFiLO / IEEE BigData Cup task shows the highest-reported single-model result (EdgeAttNet) reaches ~0.70 multi-scale IoU (MIoU), versus ~0.58 for a plain U-Net.  To move from the friend’s ~138th range toward top 50, you generally need a **diverse probability-level ensemble** (5-fold or multi-architecture), **metric-optimized inference** (threshold, area filters, TTA), and **boundary-sensitive training/post-processing**.

The 3 current Colab runs are a good first ensemble, but the pipeline still makes several sub-optimal choices that can be fixed without extra training budget:

1. The model-selection metric in `train.py` is **relaxed Panoptic Quality**, not the competition MIoU/pairwise-IoU.
2. The encoder is `efficientnet_b0/b1/b3` at 512/768/1024; literature suggests `efficientnet-b4` or `tu-efficientnet-b4` with `unetplusplus` is a better single-model baseline.
3. The input is 1-channel; ImageNet transfer is better with 3-channel replication or a learned channel adapter.
4. TTA is only on account 3 and uses 3 flips; adding `hv` (4 flips) and multi-scale TTA helps.
5. Post-processing does not optimize threshold / `min_area` on out-of-fold predictions.
6. The probability map is resized to 2048 and thresholded — correct — but no CRF / edge smoothing / hole filling is applied.

---

## 2. What the leaderboard actually rewards

From the research, the Kaggle/IEEE BigData Cup 2026 scoring is expected to include:

- **Multi-scale IoU (MIoU)** — box-counting at cell sizes 1,2,4,8,16,32,64,128,256,512.  This metric penalizes missing fine barbs / thin filament edges more than standard IoU.
- **Pairwise IoU** — for each image, every overlapping (gt_i, pred_j) pair is scored, then averaged.  This is **not** the usual pixel-wise IoU.
- **AP@IoU θ = 0.5:0.05:0.95** — COCO-style average precision at 10 IoU thresholds.
- **Hit rate / miss rate** — recall-oriented; missing small filaments is heavily penalized.
- **Possible time constraint** — source code and inference speed may be checked, so very heavy ensembles need a lightweight fallback.

**Implication:** do not optimize for accuracy, pixel-wise Dice, or plain PQ.  Implement the real metrics locally and use them for early stopping, threshold search, and ensemble weighting.

---

## 3. The 3 current accounts — what they are and what they miss

| Account | Fold | Model | Resolution | Epochs | TTA | Strength | Weakness |
|---|---|---|---|---|---|---|---|
| 1 | 0 | U-Net + EfficientNet-B0 | 512 | 10 | No | Fast, pipeline check | Too small, only 10 epochs, no TTA |
| 2 | 1 | U-Net + EfficientNet-B1 | 768 | 15 | No | Better capacity | Still no TTA, metric not optimized |
| 3 | 2 | U-Net + EfficientNet-B3 | 1024 | 30 | Yes (4 flips) | Strong resolution + TTA | Longest, may not finish before disconnect |

**Key missing pieces:**

- All three use plain `unet` instead of `unetplusplus` or `deeplabv3plus`.
- None replicates grayscale to 3 channels.
- None uses focal/tversky loss or boundary loss.
- No mask-aware cropping or copy-paste augmentation.
- No probability-map ensemble across accounts.
- No OOF threshold / area search.

**What to do with their outputs:** download the `submission.csv` and `best_fold_*.pth` from each, then ensemble the three probability maps (or average decoded RLE masks).  This is the **first real ensemble**.

---

## 4. Phased battle plan

### Phase 0 — Fix validation / evaluation (no extra GPU needed)

- [ ] Switch `train.py` early stopping from `panoptic_quality` to `mIoU_multiscale` or `mIoU_pairwise`.
- [ ] Add a `metrics.py` function that computes MIoU, pairwise IoU, and AP locally on OOF predictions.
- [ ] Add threshold sweep (0.30–0.80) and `min_area` sweep on OOF to pick per-fold settings.
- [ ] Compute a baseline score on OOF for the current 3 models so you know the gap.

### Phase 1 — Quick training upgrades (same Colab budget)

- [ ] Change all accounts to `unetplusplus` + `tu-efficientnet-b4` (or `efficientnet-b4` if `tu-` fails).
- [ ] Use 3-channel input by replicating the grayscale image.
- [ ] Replace loss with `Focal Tversky` or `BCE + Dice + Focal` with tunable weights.
- [ ] Add heavier Albumentations: copy-paste, CLAHE, GridMask, Gauss/ISO noise, random gamma, elastic/grid distortion.
- [ ] Add mask-aware cropping: 70% of crops centered on a filament pixel.
- [ ] Train on the full downsampled 2048 image at 768/1024 (do not use small patches that lose context).
- [ ] Use FP16 AMP, gradient accumulation, and a lower learning rate (`1e-4` with `CosineAnnealing`).

### Phase 2 — 5-fold full cross-validation ensemble

- [ ] Train 5 models, one per fold, on Colab (reuse the 3 accounts sequentially or add 2 more).
- [ ] Save **probability maps** (not just RLE) for the test set from each fold.
- [ ] Average the 5 probability maps, then threshold at full 2048.
- [ ] Use OOF from all 5 folds to optimize the final threshold and area filter.

### Phase 3 — Stronger architectures (after baseline is solid)

- [ ] Try `unetplusplus` with `tu-efficientnet-b4` or `tu-efficientnet-b5`.
- [ ] Try `deeplabv3plus` with `tu-efficientnet-b4` + ASPP.
- [ ] Try `segformer-b3/b4` with `mit-b3/b4` (if it fits T4 memory).
- [ ] Implement a simplified edge branch: add an auxiliary edge/boundary map prediction and a boundary loss.
- [ ] If enough time, attempt a simplified EdgeAttNet-style edge-guided attention block.

### Phase 4 — Inference and post-processing polish

- [ ] Full-resolution inference with tiling (768/1024 tiles, 64–128 overlap) for account 3.
- [ ] D4 TTA (8 transforms) on the strongest account.
- [ ] Multi-scale TTA: run at 0.8x, 1.0x, 1.25x, average probabilities.
- [ ] CRF post-processing on averaged probabilities.
- [ ] Morphological closing (fill small gaps), opening (remove specks), hole filling.
- [ ] No-filament classifier: return empty prediction if image-level probability is low.
- [ ] Validate RLE round-trip, no overlaps, all image IDs present.

---

## 5. Architecture findings

- **EdgeAttNet (U-Net + Edge-Guided Multi-Head Self-Attention)** is the highest-reported method on MAGFiLO: ~0.645 pairwise / 0.703 multiscale MIoU.  It explicitly learns edge maps and injects them into attention Key/Query.  This is the **single-model target**.
- **U-Net++** consistently beats plain U-Net by aggregating multi-scale skip connections; the friend’s notebook already uses it with EfficientNet-B4.
- **DeepLabV3+ with ASPP** is strong for multi-scale objects; worth testing with `tu-efficientnet-b4`.
- **SegFormer-B3/B4** (MiT encoder + MLP decoder) is efficient and avoids positional-encoding interpolation issues, but may need more data.
- **Swin / ConvNeXt** backbones give better global context than EfficientNet but use 2–3x memory.  Use the smaller variants (Swin-T, ConvNeXt-T) on T4.
- **HRNet** keeps high-resolution features throughout; good for 2048 images but heavy.
- **Flat U-Net** (0.26M params, SCA/CSA blocks) is an option if a time/efficiency constraint is added; not the first choice for accuracy.

### Recommended architecture stack for the next run

1. **Primary:** `unetplusplus` + `tu-efficientnet-b4` (ImageNet pre-trained).
2. **Backup:** `deeplabv3plus` + `tu-efficientnet-b4`.
3. **High-risk/high-reward:** simplified EdgeAttNet or boundary-attention branch.

---

## 6. Input, pretraining, and normalization findings

- **Channel replication:** for 1-channel H-alpha, replicate to 3 channels before feeding ImageNet pre-trained encoders.  `smp` with `in_channels=1` adapts the first conv, but replication is the standard top-Kaggle practice and preserves pre-trained filters.
- **ImageNet pretraining still helps** even on solar data because low-level features (edges, texture) transfer.
- **Solar-specific normalization:** compute mean/std from the training set or use per-image percentile normalization (0.5–99.5 percentiles) instead of ImageNet stats.
- **Limb darkening correction:** fit a second-order cosine model to radial intensity and normalize.  This is a common classical preprocessing step and reduces limb false positives.
- **Add a radial distance channel:** the distance from disk center as an extra input channel helps the model understand limb vs disk center.
- **Denoising:** light Gaussian blur (σ=0.5–1.0) or non-local means as augmentation/test-time robustness.

---

## 7. Loss function findings

For highly imbalanced filament segmentation, the research strongly favors composite losses:

- **BCE + Dice:** stable baseline.  Current pipeline already has this.
- **Focal Loss:** focus on hard examples.  Use `gamma=2.0` initially, then `3.0`, with `alpha=0.25–0.7` to up-weight foreground.
- **Tversky Loss:** asymmetric version with `alpha=0.3`, `beta=0.7` penalizes false negatives more than false positives — good for recall.
- **Focal Tversky:** `alpha=0.3/0.7`, `beta=0.7/0.3`, `gamma=1.33`.  Strong for tiny objects.
- **Asymmetric Focal Tversky (MONAI):** `delta=0.7`, `gamma=0.75` designed for binary imbalanced segmentation.
- **Lovász-Softmax:** directly optimizes IoU.  Best used as auxiliary loss or for final fine-tuning.
- **Boundary Loss / Hausdorff DT Loss:** improves thin-filament boundary precision; critical for MIoU.
- **OHEM Cross-Entropy:** select top-K hardest pixels per batch; set `min_kept=50k–100k`.
- **Per-pixel reweighting:** foreground can be ~0.1% of pixels; inverse-frequency weighting gives foreground weight ~500x.
- **Unified Focal Loss:** generalizes Dice + CE into one framework; robust for extreme imbalance.

### Recommended loss stack

```
loss = 0.4 * Dice + 0.3 * Focal + 0.2 * BCE + 0.1 * Boundary
```

or

```
loss = 0.5 * FocalTversky(alpha=0.3, beta=0.7, gamma=1.33) + 0.3 * Dice + 0.2 * Boundary
```

Tunable via env vars in `config.py`.

---

## 8. Data augmentation findings

- **Geometric:** horizontal/vertical flips (p=0.5), rotation ±45° (p=0.8).  Avoid 90° if chirality matters.
- **CLAHE:** `clip_limit=2.0, tile_grid_size=(8,8), p=0.5` for H-alpha contrast enhancement.
- **Elastic/Grid distortion:** `A.ElasticTransform(alpha=50, sigma=120*0.02)` and `A.GridDistortion` with p=0.5.
- **Noise/Intensity:** `GaussNoise`, `ISONoise`, `RandomBrightnessContrast`, `RandomGamma` with p=0.4–0.8.
- **GridMask:** p=0.3 for regularization.
- **Copy-paste of filament masks:** extract filament patches and paste onto images with few filaments.  Reported +7% Dice in medical tasks.
- **Mosaic:** combine 2–4 training images into one; increases filament density per batch.
- **Mask-aware cropping:** 70% of crops centered on filament, 30% random.  Avoids background-only batches.
- **Oversampling:** sample images with more filaments 2–3x more often.
- **Multi-scale training:** randomly resize between 384 and 1024 during training.
- **Label smoothing / SVLS:** small boundary smoothing (0.1–0.2) to account for annotator uncertainty.

### Recommended augmentation pipeline

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.8),
    A.OneOf([
        A.ElasticTransform(alpha=50, sigma=2.4, p=1.0),
        A.GridDistortion(p=1.0),
    ], p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
    ], p=0.4),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8),
    A.GridMask(num_grid=4, ratio=0.5, p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])
```

---

## 9. Training tricks findings

- **Mixed precision (AMP):** essential on T4; ~50% memory reduction and 2–3x throughput.
- **Gradient accumulation:** effective batch 8–16 with batch size 1–2 on T4.
- **AdamW + CosineAnnealing + warm restarts:** start LR 1e-4, min 1e-6, weight decay 1e-5.
- **SWA (Stochastic Weight Averaging):** average weights during last 25% of training; ~1% mIoU gain.
- **EMA (Exponential Moving Average):** maintain shadow weights with decay 0.999; use for inference.
- **Snapshot ensembling:** cyclic LR, save checkpoints at cycle minima; average predictions.
- **Deep supervision:** auxiliary heads at decoder levels 2 and 3 with weights 0.4 and 0.2.
- **Two-stage training:** first 70% epochs with BCE/Dice, last 30% with Lovász or Focal Tversky.
- **Focal curriculum:** start with `gamma=3.0`, decay to `2.0`.
- **Pretrain on full MAGFiLO:** if the competition only provides 704 training images, the full 1,593-image MAGFiLO set is valuable for pretraining (chirality labels can be ignored or used as auxiliary).

---

## 10. Test-time augmentation and ensembling findings

- **Always average probabilities, not masks.**  Averaging after sigmoid preserves uncertainty.
- **4-flip TTA (none, h, v, hv):** already added to `infer.py`; make sure all accounts use it.
- **D4 group (8 views):** identity, r90, r180, r270, h, v, transpose, anti-diagonal.  ~1–2% mIoU gain, 8x inference cost.
- **Multi-scale TTA:** run at 0.8x, 1.0x, 1.25x; bilinear resize probabilities back to 2048 before averaging.
- **Sliding-window / tiled inference:** for 2048 full image, use 768–1024 tiles with 64–128 overlap and Gaussian/Hann blending.
- **TTA at full resolution:** never threshold before upsampling; always threshold at 2048.
- **Weighted ensemble:** weight models by validation `mIoU_multiscale`; optimize weights on OOF.
- **5-fold ensemble:** the single biggest ensembling gain.  Train all 5 folds and average.
- **Model soup:** average model weights of same architecture in same loss basin; cheap if same model.
- **CRF after averaging:** `pydensecrf` with probability map and original image; 0.5–1% gain.
- **Temperature scaling:** calibrate logits with T > 1 before sigmoid if using Dice/Focal.

### Practical inference order

1. Load image, preprocess, optionally correct limb darkening.
2. Run model on 2048 / tiled / multi-scale.
3. Apply TTA and average **all** probability maps.
4. Apply CRF on the average probability map.
5. Threshold at full 2048.
6. Morphological closing/opening, hole filling.
7. Watershed if instances touch.
8. Filter by area and disk mask.
9. Encode to RLE.

---

## 11. Post-processing findings

- **Threshold optimization:** grid search 0.30–0.80 on OOF; do not use fixed 0.5.
- **Area filter tuning:** `min_area` and `max_area` should be optimized on OOF.  Typical `min_area` 50–200 px; `max_area` ~10% of disk.
- **Morphological opening:** remove salt noise with disk radius 1–2.
- **Morphological closing:** fill small gaps with disk radius 2–3.
- **Hole filling:** `remove_small_holes` with area threshold.
- **Edge smoothing:** light Gaussian blur (σ=1.0–1.5) on binary mask then re-threshold.
- **Watershed for touching filaments:** distance transform + watershed with `min_distance=10–20`.
- **Disk masking:** zero predictions outside 0.95 Rsun; use sunpy or radial distance.
- **Off-limb truncation:** predictions beyond disk are guaranteed false positives.
- **NMS for overlapping instances:** mask-based non-max suppression if CRF/ensemble produces overlaps.
- **No-filament classifier:** if image-level max probability is below a learned threshold, return empty.
- **Bilinear probability upsample vs nearest binary:** always upsample **probability** with bilinear, threshold at 2048.  Nearest-upsampled binary masks create zigzag/blocky patterns.
- **RLE encoding:** `np.asfortranarray(mask.astype(np.uint8))`, `pycocotools.mask.encode`, decode counts to UTF-8 string.
- **Submission validation:** decode all RLEs, check for overlaps, confirm all image IDs present.

---

## 12. Metric implementation findings

Implement these in `metrics.py` and use them for model selection:

- **mIoU_pairwise:** for each image, for every overlapping (gt_i, pred_j), compute `|gt ∩ pred| / |gt ∪ pred|`, average.
- **mIoU_multiscale:** for each overlapping pair, compute intersection ratio at cell sizes 1,2,4,8,16,32,64,128,256,512, integrate with trapezoidal rule.
- **AP@IoU 0.5:0.05:0.95:** COCO-style AP over 10 IoU thresholds.
- **Hit rate / miss rate:** TP/(TP+FN) and FN/(TP+FN).
- **PQ approximation:** SQ * RQ with IoU threshold 0.5 matching.

**Use `mIoU_multiscale` as the primary early-stopping metric.**

---

## 13. Pseudo-labeling and semi-supervised findings

- **Teacher-student self-training:** train a model, predict on 180 test images, keep pixels with confidence > 0.9, add to training set, retrain.
- **Multi-model agreement:** only keep pseudo-labels where all models in the ensemble agree.
- **Consistency regularization:** strong augment unlabeled images before pseudo-labeling.
- **Cross-pseudo supervision (CPS):** two networks with different initialization supervise each other.
- **Extra data:** BBSO, CHASE/HIS, HAS/MHAS datasets can augment training after format conversion and domain adaptation.

**Pseudo-labeling is an end-game technique; do it after the 5-fold ensemble is solid.**

---

## 14. Domain-specific findings

- **Filaments are elongated dark structures above polarity inversion lines.**  Shape priors (aspect ratio, directionality) help.
- **AR vs QS filaments:** active-region filaments are ~50 Mm, quiescent up to ~200 Mm.  Multi-scale receptive fields are needed.
- **Barbs are fine lateral protrusions;** MIoU specifically penalizes missing them.
- **Sunspots look dark like filaments but are compact;** use shape/elongation to discriminate.
- **Limb darkening and foreshortening near the limb reduce contrast.**  Radial distance channel or limb darkening correction helps.
- **Multi-wavelength fusion (H-alpha + AIA 304 Å):** not available in competition, but if you can source it, large gain.
- **Magnetic field / chirality:** not needed for segmentation, but chirality as auxiliary task can improve representation.

---

## 15. What to do right now (priority order)

1. **Let the 3 current Colab runs finish.**  Do not restart them.  Their outputs are the first ensemble.
2. **While they run, implement the competition metrics** in `metrics.py` and a threshold/area sweep script.
3. **Generate a new set of no-Drive notebooks** with:
   - `unetplusplus`
   - `tu-efficientnet-b4` (or `efficientnet-b4`)
   - 3-channel input
   - Focal Tversky / boundary loss
   - Heavy augmentation (copy-paste, CLAHE, GridMask)
   - 4-flip TTA and threshold/area search
4. **After current runs finish, re-run all 3 accounts with the new notebooks** — or expand to 5 folds.
5. **Download all `submission.csv` and `best_fold_*.pth` files** and build a probability-level ensemble.
6. **Optimize the final threshold and area filter on combined OOF predictions.**

---

## 16. What is likely to be a waste of time

- Training a single plain U-Net at 512 for 10 epochs and hoping it wins.
- Optimizing for pixel-wise Dice / PQ instead of MIoU / pairwise IoU.
- Thresholding at low resolution and then upsampling with nearest neighbor.
- Running TTA but voting on binary masks instead of averaging probabilities.
- Using a huge model like Mask2Former / SegFormer-B5 on T4 (memory will kill it).
- GANs, adversarial training, or heavy transformer re-implementation without a working baseline.

---

## 17. Account-by-account next-config recommendations

| Account | Next config | Why |
|---|---|---|
| 1 (fast) | `unetplusplus` + `tu-efficientnet-b4`, 768, 15 epochs, 4-flip TTA, Focal Tversky | Better single model than B0/512, quick enough |
| 2 (medium) | `deeplabv3plus` + `tu-efficientnet-b4`, 1024, 20 epochs, 4-flip TTA | Diverse architecture, good multi-scale |
| 3 (strong) | `unetplusplus` + `tu-efficientnet-b5`, 1024, 30 epochs, 4-flip TTA + multi-scale | Highest capacity, uses full T4 memory |

If a 4th/5th Colab account is available:

- Fold 3: `unetplusplus` + `tu-efficientnet-b3`, 1024
- Fold 4: `unetplusplus` + `tu-resnet50`, 1024

---

## 18. Key numbers to aim for

- **Plain U-Net baseline on MAGFiLO:** ~0.57 pairwise / 0.58 multiscale MIoU.
- **EdgeAttNet reported SOTA:** ~0.645 pairwise / 0.703 multiscale MIoU.
- **A competitive Kaggle leaderboard score** is likely in the 0.65–0.72 MIoU range for top 50.
- **Friend at 138th:** unknown exact score, but likely ~0.55–0.62 range.  A well-executed 0.65+ ensemble should beat it.
- **Val dice / PQ numbers** from current Colab runs are not directly comparable to MIoU; use the local MIoU implementation to judge.

---

## 19. Final verdict on “can we beat the friend / get top 50?”

**Yes, it is possible**, but only if you move from a single-model 3-account setup to a **probability-ensemble 5-fold pipeline with metric-optimized post-processing**.  The friend’s notebook is essentially one strong U-Net++ model.  A multi-resolution, multi-encoder, 5-fold ensemble with TTA and threshold/area search will almost always beat a single notebook on a segmentation Kaggle, **provided** it is well-executed.

However, **guarantees are impossible** because:

- The exact public leaderboard is unknown.
- Free Colab T4 sessions may disconnect before long 30-epoch runs finish.
- Some advanced techniques (CRF, EdgeAttNet, pseudo-labeling) require significant extra code and runtime.

**Realistic target:** with the Phase 0–2 changes, expect a 0.60–0.65 MIoU.  With Phase 3–4, 0.65–0.72 is reachable.  Top 50 likely needs ≥ 0.65, so the goal is realistic if you can get the 5-fold ensemble and inference tuning right.

---

## 20. References and further reading

- EdgeAttNet (arXiv 2025) — U-Net + Edge-Guided MHSA for MAGFiLO.
- U-Net++ (Nested U-Net architecture) — multi-scale skip connections.
- DeepLabV3+ with ASPP — multi-scale context.
- SegFormer — efficient transformer-based segmentation.
- Flat U-Net (2025) — SCA/CSA, lightweight.
- MONAI Asymmetric Focal Tversky / Boundary / Hausdorff losses.
- Albumentations — heavy augmentation library.
- PySODMetrics — multi-scale IoU reference.
- pycocotools — RLE encode/decode.
- pycrfdense / pydensecrf — CRF post-processing.

