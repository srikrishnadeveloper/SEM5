# Verification & Recommendations — Solar Filament Segmentation 2026

> Verification of `01_competition_brief.md` and `public_baseline_resnet_unet_summary.md`, plus a concrete alternative design if the public ResNet34-UNet 512×512 baseline is not enough for a top-100 finish.

---

## 1. How this was verified

- All URLs were checked with `web_search` / `webfetch` and the returned snippets were compared with the claims.
- The public baseline Kaggle notebook was downloaded from its GitHub mirror (`avikds/Kaggle-Notebooks-Avik`) and the source cells were inspected to confirm architecture, numbers, and post-processing.
- The current IEEE Big Data Cup 2026 page, the MAGFiLO website/dataset, the Scientific Data paper, and the EdgeAttNet paper/repo were used to cross-check the competition task, metrics, and a stronger baseline.

---

## 2. `01_competition_brief.md` — fact/URL/formula check

| File | Line(s) | Claim | Verdict | Evidence / correction |
|------|---------|-------|---------|----------------------|
| `01_competition_brief.md` | 9–10 | Competition is run by the Earth-Space AI Research (ESAIR) Lab and is an IEEE BigData Cup 2026 track. | **Correct** | Kaggle org `esairlab` lists the competition and the challenge is listed at `bigdataieee.org/BigData2026/cup/`. |
| `01_competition_brief.md` | 13 | Kaggle URL `https://www.kaggle.com/competitions/filament-segmentation-2026` | **Correct / exists** | Search results show the live competition with the same slug and a `$3,000` prize pool. Direct `webfetch` of Kaggle returns no content (JS/login wall), but the URL pattern is confirmed by the IEEE page and multiple search snippets. |
| `01_competition_brief.md` | 14 | IEEE BigData Cup page `https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/` | **Correct** | Page is live and describes the task, data, and evaluation. |
| `01_competition_brief.md` | 16 | Sponsors: U.S. NSF and NSO | **Correct** | NSF grant `#2209912`, `#2433781` and the NSO GONG program are acknowledged in the MAGFiLO Scientific Data paper and on `mlecofi.net/magfilo`. |
| `01_competition_brief.md` | 34 | MAGFiLO DOI `10.1038/s41597-024-03876-y` | **Correct** | Resolves to the *Scientific Data* paper and the MAGFiLO Dataverse record. |
| `01_competition_brief.md` | 38 | 10,244 filaments from 1,593 GONG H-α observations, 2011–2022. | **Correct for the full MAGFiLO dataset** | Confirmed by the paper, the MAGFiLO website, and the NSO blog. **Note:** the *competition* training fold contains 1,154 images / 8,199 annotations (see baseline notebook output below), not the full 1,593. |
| `01_competition_brief.md` | 40 | Competition images are `2048 × 2048` pixels. | **Correct** | The Kaggle dataset description explicitly states: "An H-alpha observation is an image of size `2048 x 2048` pixels ... converted from FITS to JPEG." |
| `01_competition_brief.md` | 45 | Chirality labels are left, right, or unidentifiable. | **Incomplete** | The Kaggle/MAGFiLO COCO `categories` list four labels: `Left`, `Right`, `Unidentifiable`, and `Ambiguous` (see `mlecofi.net/magfilo`). The *final* public MAGFiLO paper says ambiguous cases were reviewed and relabeled, so the released v1.0 has only three, but the competition JSON still exposes four. |
| `01_competition_brief.md` | 47 | Public MAGFiLO v1.0 Kaggle dataset: 704 JPEG images and 8,212 filaments. | **Correct** | Confirmed by the Kaggle dataset page and `mlecofi.net/magfilo`. |
| `01_competition_brief.md` | 80–83 | Submission is a CSV with `filament_id` and `segmentation_rle`; RLE counts only; use `pycocotools`. | **Mostly correct, one overstatement** | The running code in the baseline uses `pycocotools.mask.encode(...)` on a Fortran-ordered `(2048,2048)` `uint8` mask and writes only `rle_encoded['counts'].decode('utf-8')`. `annToMask` and `decode` are **not needed** here and are misleading in the brief. |
| `01_competition_brief.md` | 89–99 | Evaluation: 70 % quantitative (Mean Dice, Panoptic Quality, etc.), 30 % qualitative. | **Likely incorrect / not publicly confirmed** | The IEEE BigData Cup proposal and the EdgeAttNet paper both state the task is evaluated with **IoU, precision, recall, AP@IoU, hit rate, miss rate, and Multi-scale IoU (MIoU)**. No public source mentions a 70/30 Dice/PQ split. This should be treated as unverified. |
| `01_competition_brief.md` | 109–113 | Panoptic Quality (PQ) formula: `PQ = Σ IoU(TP) / (|TP| + 0.5|FP| + 0.5|FN|)` with IoU > 0.5 match. | **Formula is mathematically correct for the standard PQ metric** | It matches Kirillov et al. (CVPR 2019, arXiv:1801.00868 / DOI `10.1109/CVPR.2019.00963`). **However, it is not clear this is the competition metric.** The competition appears to use `mIoU_pairwise` and `mIoU_multiscale` (see §4). |
| `01_competition_brief.md` | 124 | PQ paper DOI `10.1109/CVPR.2019.00963` | **Correct** | Confirmed by ML Anthology, OpenAIRE, and IEEE CVPR 2019 records. |
| `01_competition_brief.md` | 131–134 | Dates: 10 July launch, 15 Nov deadline, 30 Nov winners, 14–17 Dec conference. | **Partially correct / conflicting** | IEEE BigData 2026 is indeed 14–17 Dec 2026 in Phoenix, AZ. The competition deadline is reported as 15 Nov 2026 on the UMSL LinkedIn post and on `azim-a.com/noteworthy` (which says the competition was launched 8 Jul 2026, not 10 Jul). The IEEE Big Data Cup page lists **Registration opens 15 July, Final submission 6 November**, which conflicts with the 15 Nov Kaggle deadline. The brief should cite both or flag the discrepancy. |
| `01_competition_brief.md` | 140 | Prize pool up to $3,000. | **Correct** | Confirmed by the Kaggle competitions listing and the UMSL LinkedIn post. |
| `01_competition_brief.md` | 174–177 | Organizers and affiliations. | **Correct / plausible** | The names match the authorship of the MAGFiLO paper and the IEEE proposal. |
| `01_competition_brief.md` | 181 | Citation format. | **Plausible** | Citation matches the challenge name and Kaggle URL. |
| `01_competition_brief.md` | 188 | Self-evaluation notebook `https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook` | **Not independently verifiable** | No public snippet was returned by search; Kaggle code pages require login. The URL follows Kaggle convention, but treat as unverified. |
| `01_competition_brief.md` | 191 | EdgeAttNet paper `https://arxiv.org/abs/2509.02964` | **Correct** | Live arXiv page with the stated title and authors. |
| `01_competition_brief.md` | 192 | Flat U-Net paper `https://iopscience.iop.org/article/10.3847/1538-4357/adadff` | **Correct** | Live *ApJ* article, published 13 Feb 2025. |
| `01_competition_brief.md` | 193 | PQ paper `https://arxiv.org/abs/1801.00868` | **Correct** | Live arXiv page for Kirillov et al., accepted CVPR 2019. |

---

## 3. `public_baseline_resnet_unet_summary.md` — fact/URL/formula check

| File | Line(s) | Claim | Verdict | Evidence / correction |
|------|---------|-------|---------|----------------------|
| `public_baseline_resnet_unet_summary.md` | 10 | Notebook URL `https://www.kaggle.com/code/avikdas567/solar-filament-segmentation-resnet-u-net-pipeline?scriptVersionId=334756212` and title. | **URL pattern confirmed / source inspected** | The GitHub mirror of the same notebook exists and contains identical source code and outputs. `scriptVersionId=334756212` appears in search snippets. The title matches the downloaded notebook. |
| `public_baseline_resnet_unet_summary.md` | 17 | Runtime: Kaggle dual T4, PyTorch 2.10.0+cu128, CUDA 12.8. | **Not independently verifiable** | The code only checks `torch.__version__` and `torch.cuda.device_count()`. Specific PyTorch/CUDA build numbers come from the Kaggle environment at runtime and cannot be confirmed externally. |
| `public_baseline_resnet_unet_summary.md` | 44–51 | Dataset size: 1,154 frames, 8,199 annotations, 4 categories, ~7.1 filaments/image. | **Correct** | Notebook cell output: `Unique Observation Frames (Images): 1154`, `Discrete Filament Targets (Annotations): 8199`, `Distinct Structural Categories: 4`. |
| `public_baseline_resnet_unet_summary.md` | 57–60 | Categories: Left, Right, Unidentifiable, Ambiguous. | **Correct** | Matches the MAGFiLO website and the Kaggle dataset description. |
| `public_baseline_resnet_unet_summary.md` | 70–72 | `bbox` `[x,y,w,h]`, `area`, polygon `segmentation`, `id`. | **Plausible / standard COCO** | Confirmed by the MAGFiLO dataset description on Kaggle. |
| `public_baseline_resnet_unet_summary.md` | 75 | Same image file indexed under multiple `image_id`s due to multi-annotator setup. | **Confirmed in code** | Notebook source comments and the MAGFiLO paper describe the multi-annotator pipeline. |
| `public_baseline_resnet_unet_summary.md` | 81–84 | `SolarFilamentDataset`: grayscale load, `cv2.fillPoly`, resize to 512×512 (`INTER_AREA` for image, `INTER_NEAREST` for mask), random flips. | **Correct** | Matches the `SolarFilamentDataset.__getitem__` source (cell 21 of the notebook). |
| `public_baseline_resnet_unet_summary.md` | 96–103 | ResNet34UNet: pseudo-RGB, ResNet-34 ImageNet weights, encoder stages 256→128→64→32→16 with channels 64→512. | **Correct** | Matches `class ResNet34UNet` in the notebook (cell 24). |
| `public_baseline_resnet_unet_summary.md` | 118–121 | Decoder uses `nn.Upsample`, skip concatenations, final `Conv2d(32,1,1)`. | **Correct** | Matches the decoder and `self.final_head` in the notebook. |
| `public_baseline_resnet_unet_summary.md` | 127–145 | `ScientificDiceLoss` (smooth `1e-5`) and `BCEWithLogitsLoss` with `0.6*Dice + 0.4*BCE`. | **Correct** | Source code matches exactly. The markdown mentions Focal Loss in cell 13, but the loss used at run time is the BCE+Dice combo. |
| `public_baseline_resnet_unet_summary.md` | 163–166 | 85/15 random split → 980 train / 174 val; `KFold` imported but unused. | **Correct** | Confirmed by notebook cell 28 source and output. |
| `public_baseline_resnet_unet_summary.md` | 170–172 | Batch size 16, `num_workers=4`, `pin_memory=True`. | **Correct** | Confirmed by notebook cell 28. |
| `public_baseline_resnet_unet_summary.md` | 176–180 | AdamW `lr=1e-3`, `wd=1e-4`, `CosineAnnealingLR(T_max=10)`, AMP, `DataParallel`, 5 epochs. | **Correct** | Confirmed by notebook cell 28. |
| `public_baseline_resnet_unet_summary.md` | 184–191 | Training log with validation Dice ~0.45–0.53. | **Correct** | Notebook output (cell 29) matches the table exactly. |
| `public_baseline_resnet_unet_summary.md` | 197–201 | Validation function `calculate_batch_dice_score`: `(2.0 * intersection / (union + 1e-8))`. | **Mathematically correct, but variable name is misleading** | In the code `union = flat_predictions.sum() + flat_ground_truth.sum()`, i.e. the *sum of the two mask areas*, not the true set union (`sum − intersection`). With the `2.0 *` numerator, this is the standard **Dice** coefficient. It is **not** IoU. The summary should clarify that the function returns Dice despite the local variable name. |
| `public_baseline_resnet_unet_summary.md` | 209–217 | Post-processing: threshold 0.45, upscale 2048, 5×5 ellipse close, 8-connected CC, 250 px area filter, RLE, blank fallback. | **Correct** | Matches the notebook inference cell (cell 33). |
| `public_baseline_resnet_unet_summary.md` | 218 | Submission shape `(3215, 2)` for 180 test images. | **Correct** | Notebook output: `[INFO] Shape configuration of final submission matrix payload: (3215, 2)` and `[INFO] Active instances registered in structural test queue: 180`. |
| `public_baseline_resnet_unet_summary.md` | 287–299 | Limitations list. | **Mostly fair / opinion** | Each point is supported by the notebook (short 5-epoch run, single split, 512×512, pseudo-RGB, no TTA, etc.). |

---

## 4. The most important correction — evaluation metrics

`01_competition_brief.md` (lines 89–123) describes the scoring as 70 % quantitative (Mean Dice + Panoptic Quality) and 30 % qualitative. **This is not supported by the public sources** and appears to be wrong for the Kaggle/IEEE challenge.

The authoritative sources describe the following metrics:

1. **IEEE BigData Cup 2026 proposal** (`bigdataieee.org/BigData2026/cup/solar-filament-segmentation/`):
   > “performance evaluated using IoU, precision, recall, AP@IoU θ, hit rate, miss rate, and a proposed Multi-scale Intersection over Union (MIoU) metric.”

2. **EdgeAttNet paper** (arXiv:2509.02964, §IV):
   - **Pairwise IoU** (`mIoU_pairwise`): per-image, all pairs `(gt_i, pt_j)` with non-zero intersection are scored with standard IoU.
   - **Multiscale IoU** (`mIoU_multiscale`): a scale-aware metric that downsamples masks at multiple resolutions and integrates the intersection ratio, making it sensitive to barbs/edges.

3. The **public ResNet34-UNet baseline only reports `Validation Mean Dice`** and never computes the competition-style mIoU. That is a real gap: **training to Dice may not align with the leaderboard**.

**Recommendation:** any alternative design must explicitly compute `mIoU_pairwise` and `mIoU_multiscale` (or the closest COCO-style AP/IoU proxy) for validation and model selection, not just Dice.

---

## 5. Concrete alternative high-level approach for a top-100 finish

The public ResNet34-UNet baseline stops at ~0.48 validation Dice after 5 epochs at 512×512. That is far below the EdgeAttNet-reported `mIoU_pairwise = 0.6451` / `mIoU_multiscale = 0.7032` on the MAGFiLO test split. A top-100 finish (currently ~300 teams, so top third) should be achievable with a method that beats the public baseline on the *actual* competition metrics.

### 5.1 Recommended architecture: **EdgeAttNet-style Edge-Guided U-Net**

Use the design from the EdgeAttNet paper (arXiv:2509.02964, GitHub `dasjar/EdgeAttNet`):

- **Backbone:** standard U-Net encoder–decoder with 4 downsampling stages (64 → 128 → 256 → 512 channels, spatial resolution 2048 → 1024 → 512 → 256 → 128 if full image is used).
- **Edge prior branch:** a small convolutional head that predicts an edge map directly from the input H-α image.
- **EG-MHSA bottleneck:** two Edge-Guided Multi-Head Self-Attention blocks at the U-Net bottleneck. The edge map linearly transforms the Query and Key matrices, steering self-attention toward filament boundaries and barbs without positional encodings.
- **Output:** single-channel logit mask of the same spatial size as the input.
- **Why this is concrete:** the paper reports this exact architecture beats plain U-Net (`mIoU_p=0.5724` → `0.6451`), U-Net+MHSA, and is **parameter-efficient** (22.7 M params).

**Colab-sensible variant:** if the full `UNetEdgeTransformer` implementation is not yet usable (the public repo currently has circular imports and the model class is not in the downloaded files), implement the same idea in `segmentation-models-pytorch`:

```python
import segmentation_models_pytorch as smp

# Strong, compact encoder
model = smp.Unet(
    encoder_name="timm-efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    activation=None,
)
```

and add a learned **edge-prior branch** that:
1. Computes a soft edge map from the input (e.g., a 2-layer conv stack or a Canny/LoG pre-processing channel).
2. Modulates attention in the bottleneck or is used as an auxiliary loss.

If memory is tight, swap to `timm-efficientnet-b2` or a `resnet34` encoder but **replace the first conv to accept 1 channel** and copy/average ImageNet weights.

### 5.2 Training recipe

| Item | Recommendation |
|------|----------------|
| **Preprocessing** | Use the EdgeAttNet solar-specific pipeline: Hough Circle Transform to find the solar disk, mask off-limb pixels, **radial flattening** to remove limb darkening, `3×3` Gaussian blur (`σ=0.7`), then **CLAHE** inside the disk mask. This is directly from the paper and improves faint-filament visibility. |
| **Input resolution** | Do **not** train at 512×512. Use random crops from the original `2048×2048` at **1024×1024** (or 768×768 on free Colab). At inference, use overlapping tiles and average logits to recover the full `2048×2048` mask. |
| **Crop strategy** | Sample crops that contain at least one filament with probability 0.7; otherwise sample random negative crops. This balances the long-tail of small filaments. |
| **Augmentation** | Horizontal + vertical flips only (preserve physical orientation), mild brightness/contrast, and CLAHE-level variations. Avoid arbitrary rotation, which distorts chirality-related barb orientation. |
| **Loss** | Optimize for the competition metrics. Use a combination of **Dice + Boundary + BCE/Focal**. Specifically: `Loss = 0.4*Dice + 0.3*BCEWithLogits + 0.2*BoundaryLoss (or clDice) + 0.1*EdgeBCE`. Boundary/clDice losses improve thin barb recall, which is heavily weighted by `mIoU_multiscale`. |
| **Optimizer / schedule** | AdamW, `lr=1e-4` (or 1e-3 with warmup), `weight_decay=1e-4`, mixed precision, gradient clipping. Use `CosineAnnealingWarmRestarts` or `OneCycleLR` over 30–50 epochs instead of 5. |
| **Cross-validation** | Run a **5-fold split stratified by observation month / instrument code** (the two letters in the filename, e.g. `Bh`, `Th`) so that validation is not temporally/instrumentally identical to training. This is more reliable than the baseline's single 85/15 random split. |
| **Validation metrics** | Compute **pairwise IoU** and **multiscale IoU** on the validation set, matching the EdgeAttNet paper. Early-stop on the average of the two, not on Dice. Also compute COCO-style AP@0.5 as a sanity check. |
| **External data** | The public MAGFiLO v1.0 Kaggle dataset (704 images / 8,212 masks) may be used **only if the competition rules permit external data and it does not overlap with the hidden test set**. Pre-train on it, then fine-tune on the competition train fold. If in doubt, do not use the public annotations as additional supervision for the same observations in the competition test set. |

### 5.3 Post-processing

1. **Full-resolution inference:** tile the `2048×2048` image with overlapping `1024×1024` tiles (stride 512), forward through the model, and merge logits with a Gaussian weight / average to avoid tile seams.
2. **TTA:** average predictions from horizontal, vertical, and combined flips at test time.
3. **Thresholding:** use **hysteresis** — start from a high confidence seed (`t_high = 0.6`) and grow through `t_low = 0.35`. This keeps filaments connected while suppressing faint noise.
4. **Morphology:** close with a small ellipse (`cv2.MORPH_CLOSE`, `3×3` or `5×5`) to repair breaks; then open with a `2×2` to remove pepper noise. Tune the kernel on the OOF validation set.
5. **Connected components:** 8-connectivity. For each component, compute area and solidity. Keep components above an area threshold learned from validation (e.g., 150–300 px), but **do not over-filter** because small barbs matter for `mIoU_multiscale`.
6. **Splitting over-merged regions:** if a connected component is much larger than the ground-truth median or has low solidity, apply a **watershed on the distance transform** or skeleton-based splitting to separate merged filaments.
7. **RLE encoding:** use `pycocotools.mask.encode` on a Fortran-ordered `(2048,2048)` `uint8` mask, write `{image_id}_{counter}`, and emit a blank mask row if an image has no predicted filament.

### 5.4 First-mover plan for a single Colab run

**Goal:** get a first submission that is clearly stronger than the public ResNet34-UNet baseline within 3–4 days on a free Colab T4.

| Day | Task | Colab-specific notes |
|-----|------|----------------------|
| **0. Setup** | Mount Drive, install `segmentation-models-pytorch`, `pycocotools`, `albumentations`. Download Kaggle data via `kagglehub` or `!kaggle competitions download -c filament-segmentation-2026` with a Kaggle API token. | Use `torch.cuda.amp` to fit 1024×1024 inputs on a T4 (~16 GB). |
| **1. Data pipeline** | Load COCO JSON, rasterize polygons, split into 5 folds (save indices to Drive), implement the EdgeAttNet-style preprocessing (disk mask, radial flattening, CLAHE). | Build masks once and cache as `.npz` on Drive to save time. |
| **2. Baseline model** | Train one `smp.Unet` + `timm-efficientnet-b3` fold for ~10 epochs at 768×768 (batch 4) with Dice+BCE. Verify the full pipeline produces a valid `submission.csv`. | This is a sanity check; expect >0.55 validation mIoU_pairwise. |
| **3. Stronger model** | Implement the edge-prior branch (or import EdgeAttNet if the repo is fixed) and train at 1024×1024 (batch 2, gradient accumulation 4 → effective batch 8) for 30–50 epochs with the boundary-aware loss. | Save best and last checkpoints to Drive every 5 epochs. |
| **4. Validation & tuning** | Run out-of-fold (OOF) inference for the trained fold, compute `mIoU_pairwise` and `mIoU_multiscale`. Use OOF to search the best threshold, morphological kernel, and area filter. | Do this on CPU to free GPU memory. |
| **5. Test inference** | Use the trained model (or a 2-fold ensemble if time permits) with TTA and tiled full-resolution inference on the 180 test images, then submit. | Keep the notebook `train-only` until the final cell; avoid accidentally calling the Kaggle submit API with unverified files. |

**Expected target for top-100:** with ~300 teams, a strong validation `mIoU_pairwise` in the **0.60–0.66** range and `mIoU_multiscale` in the **0.65–0.70** range should put you in the top third, especially when paired with good post-processing and TTA. The public ResNet34-UNet baseline is far below this, so the gap is large enough that the EdgeAttNet-style approach is worth the effort.

---

## 6. What to double-check before committing code

1. **Rules page:** confirm the Kaggle competition’s final metric and external-data policy. The IEEE proposal says any approach is welcome, but Kaggle may restrict additional datasets or test-set usage.
2. **Deadlines:** the brief says 15 Nov, but the IEEE Big Data Cup page says 6 Nov for the cup. Reconcile this with the Kaggle competition page once logged in.
3. **Evaluation code:** do **not** rely on the brief’s Panoptic Quality / Mean Dice description. Implement and validate the actual `mIoU_pairwise` and `mIoU_multiscale` functions (or the closest COCO-style AP/IoU equivalent) before training.
4. **EdgeAttNet repo:** the current `dasjar/EdgeAttNet` repo has the model class hidden/missing in the downloaded files. Either fix the import paths or re-implement from the paper; do not assume the public code runs out-of-the-box.

---

## 7. Bottom line

- The competition brief is generally accurate on URLs, dataset provenance, and submission format, but it is **wrong or unverified on the scoring** and slightly off/incomplete on launch dates and chirality categories.
- The public baseline summary is an accurate digest of the ResNet34-UNet notebook, including the exact training log, dataset counts, and post-processing.
- If that baseline is insufficient for a top-100 finish, the clearest concrete upgrade is an **EdgeAttNet-style edge-guided U-Net** trained at higher resolution with solar-specific preprocessing and a loss/validation scheme aligned to the competition’s **pairwise and multiscale IoU** metrics.
