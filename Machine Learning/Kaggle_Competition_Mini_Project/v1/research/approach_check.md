# Solar Filament Segmentation 2026 — Critical Approach Review

**Goal:** Identify every gap, risk, and improvement opportunity for a mini-project that aims to finish *above the 100th position* in the Kaggle *Solar Filament Segmentation 2026* competition.

**Scope inspected:**
- `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\research\`
- `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\data\`
- All code that currently exists in the repository.

**Files present at inspection:**

| Path | Size / Contents |
|------|-----------------|
| `research\01_competition_brief.md` | 7,981 bytes (193 lines) |
| `research\public_baseline_resnet_unet_summary.md` | 13,622 bytes (320 lines) |
| `research\public_baseline_resnet_unet.ipynb` | 963,443 bytes (35 cells) |
| `data\filament-segmentation-2026.zip` | 703,574,877 bytes (~671 MB) |
| `data\MAGFiLO_1.0_Kaggle_2026\train\train_images\` | 707 JPEG images, 2048 × 2048 |
| `data\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json` | 48,616,700 bytes |
| `data\MAGFiLO_1.0_Kaggle_2026\test\test_images\` | 180 JPEG images, 2048 × 2048 |
| `code\` | **empty** |
| `notebooks\` | **empty** |
| `submissions\` | **empty** |
| `plots\` | **empty** |
| `requirements.txt` / `README.md` / `.gitignore` | **not present** |

The only executable artefact is the public baseline notebook and its summary; the team has not yet produced its own training, validation, or submission pipeline.

---

## 1. Data understanding

### 1.1 What the data actually is

The competition uses the **MAGFiLO v1.0** COCO-style dataset (`01_competition_brief.md` lines 34–47):
- 1,154 `image` entries, but only **707 unique JPEG files** (confirmed by JSON inspection).
- 8,199 filament annotations (`public_baseline_resnet_unet.ipynb` Cell 7 output; also JSON inspection).
- All images are **2048 × 2048** 8-bit grayscale saved as `.jpeg` (PIL inspection: `(2048, 2048) L`).
- Filament chirality classes in the JSON are `Left`, `Right`, `Unidentifiable`, `Ambiguous` (`01_competition_brief.md` line 57–61; JSON `categories`).

### 1.2 Hidden structural issues found

- **Multi-`image_id` per file.** 296 of the 707 unique files appear under **more than one `image_id`** in the COCO JSON (JSON inspection). This is because the same observation was annotated by multiple experts. The baseline notebook groups annotations by `image_id` and treats each `image_id` as a separate sample (Cell 21, lines 315–332). The risk: the same physical image can end up in both training and validation if a random split is performed at the `image_id` level, causing **data leakage** and overly optimistic validation scores.
- **Ambiguous class is empty.** Only 3 of the 4 declared categories have annotations: `Left` 2,535, `Right` 2,590, `Unidentifiable` 3,074, `Ambiguous` 0 (JSON inspection). Any model or visualisation expecting 4 balanced classes will fail or be misleading.
- **Extreme size imbalance.** Filament areas range from **9 px to 37,739 px**, mean ≈ 2,120 px, std ≈ 2,742 px, with a long right tail (Cell 10 output; JSON inspection). A 512 × 512 training resolution collapses small objects to sub-pixel size and washes out fine barbs.
- **Highly elongated shapes.** Bounding-box aspect ratios range from 0.122 to 12.78 (mean 1.38, Cell 10 output). Standard isotropic convolutions and simple disk-shaped morphological kernels are a poor match for these long, thin structures.
- **No class or temporal distribution study.** There is no saved EDA for station codes in filenames (e.g. `Ch`, `Lh`, `Mh`, `Th`), solar-cycle year, or chirality-vs-hemisphere patterns. These are potential domain-shift features that should drive the validation split.

### 1.3 Gaps / risks

| # | Risk | Evidence | Recommended fix |
|---|---|---|---|
| 1.1 | **Data leakage through multi-annotator `image_id`s** | 296 files map to multiple `image_id`s; baseline splits on `image_id` (Cell 28, lines 541–545). | Aggregate all annotations by **physical `file_name`**, then split by file. Use `GroupShuffleSplit` or group-stratified K-Fold. |
| 1.2 | **Small filaments disappear at 512 × 512** | Min area 9 px, 25 % below 670 px (Cell 10 output). | Train at 1024 × 1024 or higher; use patch-based sampling around filament masks; keep original 2048 inference. |
| 1.3 | **Empty `Ambiguous` category causes confusion** | JSON `categories` has 4 classes but only 3 appear in `annotations` (JSON inspection). | Remove or handle `Ambiguous` explicitly; do not plot/optimise for a non-existent class. |
| 1.4 | **No EDA saved as reusable notebooks/plots** | `notebooks\` and `plots\` are empty. | Create `notebooks/01_eda.ipynb` with area/aspect/chirality/centroid/limb distributions and save PNGs to `plots/`. |
| 1.5 | **No validation that masks rasterise correctly** | `cv2.fillPoly` used directly on polygon lists (Cell 21, lines 360–365). | Visualise overlays on a few samples; check total mask area vs. COCO `area`; verify hole handling. |

---

## 2. Train / validation split strategy

### 2.1 Current state

The baseline uses a single random 85 / 15 split at the `image_id` level (`public_baseline_resnet_unet.ipynb` Cell 28, lines 541–545):

```python
train_size = int(0.85 * len(train_dataset))
val_size   = len(train_dataset) - train_size
dataset_train_split, dataset_val_split = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(GLOBAL_SEED)
)
```

This produces 980 train / 174 validation samples. `KFold` is imported (Cell 4, line 49) but **never used** (`public_baseline_resnet_unet_summary.md` line 165).

### 2.2 Why this is a blocker for a top-100 finish

- **Leakage through multi-annotator files** (see §1.1).
- **No stratification.** A random split does not control for filament count, size, chirality, or year/station. Some validation images may have 10+ filaments, others 1; the validation Dice is therefore very noisy (observed: 0.3789–0.5290 across 5 epochs, Cell 29 output).
- **Single split gives unreliable leaderboard estimate.** With only 174 validation samples, the confidence interval on PQ/Dice is wide.
- **No temporal split.** Filament morphology and solar-activity level vary with the solar cycle. A time-based holdout is the strongest proxy for the hidden test set.

### 2.3 Recommended split design

1. **Primary unit = physical file name**, not `image_id`.
2. **Group stratification:**
   - Group variable: `file_name` (after aggregating all `image_id`s).
   - Stratification target: number of filaments per file (binned), year (from filename), and/or chirality histogram.
3. **Use 5-fold Group K-Fold.** Train five models; the out-of-fold predictions become a validation ensemble and a robust metric estimate.
4. **Hold out a clean time split:** e.g. train on 2011–2019, validate on 2020–2022, to test generalisation across the solar cycle.
5. **Never use the test set for hyperparameter tuning.** The test folder (`test_images`, 180 images) must remain untouched until the final submission.

---

## 3. Model choice

### 3.1 Current state

The baseline is a **ResNet-34 U-Net** (`public_baseline_resnet_unet.ipynb` Cell 24, lines 402–485):
- Input is forced to pseudo-RGB by `torch.cat([x, x, x], dim=1)` (Cell 24, line 455).
- Uses `torchvision.models.resnet34(weights=models.ResNet34_Weights.DEFAULT)` (Cell 24, line 407).
- Decoder uses only bilinear `nn.Upsample` + `Conv2d → BatchNorm → ReLU` blocks (Cell 24, lines 421–451).
- Trained for only **5 epochs** with validation Dice below 0.55 (Cell 29 output).

### 3.2 Gaps / risks

| # | Issue | Why it hurts | Fix |
|---|---|---|---|
| 3.1 | **Pseudo-RGB channel replication** | ImageNet conv1 expects natural-image RGB correlations; replicating a solar H-α channel three times wastes capacity and keeps the first layer from learning solar-specific filters. | Replace `conv1` with a single-channel 7×7 and average the three ImageNet channel weights, or use `timm`/`smp` encoders that natively accept 1-channel input. |
| 3.2 | **ResNet-34 is shallow for 2048 → 2048** | 34-layer encoder has limited multi-scale context; competition winners typically use ResNet-50/101, EfficientNet-B4/B5, ConvNeXt, or transformer backbones. | Upgrade to **ResNet-50 / EfficientNet-B4 / ConvNeXt-T / SegFormer-B2 / Swin-UNETR**. |
| 3.3 | **Plain bilinear decoder** | No attention, no ASPP, no deep supervision, and no explicit boundary branch. Thin filaments and barbs are lost. | Use **UNet++**, **DeepLabV3+**, **SegFormer**, or **Swin-UperNet**; add a boundary/edge head. |
| 3.4 | **No pretrained solar / remote-sensing weights** | ImageNet is far from H-α imagery. Pretrained weights on satellite or medical imaging are closer, but the rules allow them as long as no other *filament ground truth* is used. | Explore `timm` ImageNet-21k or remote-sensing pretrained backbones; fine-tune conv1 for grayscale. |
| 3.5 | **5 epochs is far too few** | Public leaderboard will be dominated by models trained 50–200 epochs with heavy augmentations. | Train for 50–100 epochs with early stopping on **validation PQ**, not Dice. |

### 3.3 Concrete architecture experiments

- `smp.Unet(encoder_name="tu-efficientnet_b4", encoder_weights="imagenet", in_channels=1)`
- `smp.DeepLabV3Plus(encoder_name="tu-resnet50", in_channels=1)`
- `smp.UnetPlusPlus` with `timm-resnest50d` backbone
- A two-head network: segmentation head + boundary/distance-transform head for barb preservation.

---

## 4. Resolution

### 4.1 Current state

- Images and masks are resized from **2048 × 2048 → 512 × 512** for both training and inference (Cell 21, lines 368–369; Cell 33, line 680).
- Mask upscaling uses `cv2.INTER_NEAREST` (Cell 33, line 680).

### 4.2 Why 512 × 512 is probably too low for top 100

- A filament of area **9 px** at 2048 becomes ~0.5 px at 512 and is **invisible** to the network.
- The 25th percentile area is **670 px**; at 512 this becomes ~42 px, and the 75th percentile (**2,438 px**) becomes ~152 px.
- Thin barbs (the competition explicitly names them as a hard part, `01_competition_brief.md` line 164) are erased.
- `INTER_NEAREST` upscaling produces blocky, pixelated boundaries that lower the IoU/Dice at the original 2048 resolution.

### 4.3 Recommended resolution strategy

| Mode | Resolution | Notes |
|------|------------|-------|
| Training | **1024 × 1024** (2× downscale) or **patches of 1024/2048** | Fits a T4/V100 with batch 2–4 + mixed precision. |
| Validation | Same as training | Use a hold-out set at the same resolution. |
| Test inference | **Full 2048 × 2048** | Train a model at 1024 and run sliding-window/whole-image inference at 2048; or train a second model at 2048 with gradient checkpointing. |
| Upsampling at post-processing | **Bilinear on probabilities, then threshold** | Do not threshold at 512 and then `INTER_NEAREST` upscale. Upscale probability maps with bilinear interpolation, then threshold at 2048. |

**Patch training idea:** crop 1024 × 1024 windows that contain at least one filament mask, then also sample negative windows. At test time, use overlapping windows and average the predicted probability maps (gaussian weighting) before thresholding.

---

## 5. Loss

### 5.1 Current state

The notebook defines `ScientificDiceLoss` (Cell 27, lines 500–513) and `nn.BCEWithLogitsLoss` (Cell 28, line 551), combined as `0.6 * Dice + 0.4 * BCE` (Cell 29, line 584). However, the abstract and markdown claim the loss is **Dice + Focal** (`public_baseline_resnet_unet_summary.md` line 147; Cell 2, lines 24–29). The code and the text are inconsistent.

### 5.2 Gaps / risks

| # | Issue | Why it hurts | Fix |
|---|---|---|---|
| 5.1 | **Markdown says Focal, code uses BCE** | The reported method is not what is actually trained. | Decide on the final loss and delete the misleading text. |
| 5.2 | **No class / foreground weighting** | Filaments occupy a tiny fraction of the 2048 disk; background dominates BCE. | Use **Focal Loss** (`γ=2`, `α=0.25`) or **Tversky loss** to up-weight false negatives. |
| 5.3 | **Dice computed over whole batch** | `ScientificDiceLoss` flattens the entire batch (Cell 27, lines 508–512). Large empty images can dominate the metric. | Compute Dice **per image** and average; this matches the competition’s mean-Dice evaluation. |
| 5.4 | **No boundary / edge loss** | Barbs and thin filaments need explicit boundary supervision. | Add a **boundary BCE** or **Hausdorff / clDice / boundary IoU** term. |
| 5.5 | **Loss does not match PQ** | Training optimises pixel-level Dice/BCE, not instance-level PQ. | Use an instance-aware surrogate such as **Lovász hinge** or add a post-processing-aware penalty term. |

### 5.3 Recommended loss combinations

- `0.5 * Focal + 0.3 * Dice + 0.2 * BoundaryBCE`
- `Tversky(α=0.3, β=0.7) + Focal`
- `Lovász-Softmax / Lovász-Hinge` for direct IoU optimisation
- Add **boundary loss** using a 3-pixel dilated/eroded edge mask.

---

## 6. Augmentations

### 6.1 Current state

Only random horizontal and vertical flips are used (Cell 21, lines 372–377):

```python
if random.random() > 0.5:
    processed_frame = np.fliplr(processed_frame).copy()
    processed_mask = np.fliplr(processed_mask).copy()
if random.random() > 0.5:
    processed_frame = np.flipud(processed_frame).copy()
    processed_mask = np.flipud(processed_mask).copy()
```

### 6.2 Gaps / risks

- **Too weak for a top-100 model.** Two flips give only 4× data variation. Solar filaments have large intra-class appearance variation, noise, and low contrast; the model will overfit.
- **No photometric augmentations.** H-α images vary in contrast, seeing, and limb darkening. No CLAHE, brightness, gamma, blur, noise.
- **No spatial distortion.** Elongated filaments benefit from random rotation, scaling, elastic/grid distortion, and random cropping.
- **No solar-specific augmentations.** Limb-darkening correction, solar-disk masking, and meridian/rotation augmentations can improve generalisation.

### 6.3 Recommended augmentation pipeline ( Albumentations )

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_aug = A.Compose([
    A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.GridDistortion(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),
], bbox_params=None, keymap={'image': 'image', 'mask': 'mask'})
```

**Important:** if chirality classification is added later, horizontal flips swap Left ↔ Right and vertical flips may alter the hemispheric rule; either drop chirality augmentation or adjust labels.

---

## 7. Post-processing

### 7.1 Current state

The inference pipeline (Cell 33, lines 663–720) does:

1. Sigmoid → threshold **0.45**.
2. Resize binary mask 512 → 2048 with `cv2.INTER_NEAREST`.
3. `cv2.MORPH_CLOSE` with a **5 × 5 ellipse**.
4. 8-connected connected components (`ndimage.label` with `structure=np.ones((3,3))`).
5. Area filter: **discard components < 250 px**.
6. `pycocotools.mask.encode` each instance with Fortran-ordered `(2048,2048)` `uint8`.
7. Fallback: if no instance, emit a blank mask row `{img_id_base}_1`.

### 7.2 Gaps / risks

| # | Issue | Why it hurts | Fix |
|---|---|---|---|
| 7.1 | **Threshold 0.45 is hand-picked** | Not optimised for PQ/Dice. A lower threshold may recover thin barbs; a higher one reduces false positives. | Search threshold on a validation set using the **competition PQ** metric. |
| 7.2 | **Upscaling a binary 512 mask with INTER_NEAREST** | Produces jagged boundaries; loses sub-pixel accuracy needed for IoU. | Upscale the **probability map** with bilinear interpolation at 2048, then threshold. |
| 7.3 | **5 × 5 ellipse closing** | Disk-shaped closing can merge nearby filaments into one blob, causing many-to-one PQ errors. | Use a smaller structuring element, or replace with **closing + opening + watershed on distance transform**. |
| 7.4 | **250 px area filter** | At 2048, a real filament of area 100–200 px would be discarded; at the same time, fragments above 250 px survive. | Tune the area filter on validation; consider **area per component relative to image-level percentile** or use a soft probability threshold per component. |
| 7.5 | **No instance separation** | 8-connected components cannot split two filaments that touch; the PQ metric penalises over-merging. | Add **watershed** seeded by distance-transform maxima, or use an instance branch in the model. |
| 7.6 | **Blank-mask fallback for empty frames** | Emits a zero-area row `{img_id_base}_1` (Cell 33, lines 710–717). This is probably a harmless no-op but could be counted as a false positive if the evaluator does not ignore empty masks. | If no filament is detected, **omit the image entirely** from the CSV or verify against the self-evaluation notebook that a blank row is accepted. |
| 7.7 | **No TTA / no ensembling** | Single forward pass. | Apply horizontal/vertical/90° flips at test time, average probability maps, then post-process. |

### 7.3 Recommended post-processing grid

```python
# after averaging TTA probability maps at 2048
binary = (probs > best_threshold).astype(np.uint8)
# optional: opening to remove specks, then closing to bridge small gaps
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
# split touching instances
dt = scipy.ndimage.distance_transform_edt(cleaned)
seeds = (dt > 0.5 * dt.max())  # or local maxima
markers = ndimage.label(seeds)[0]
instances = watershed(-dt, markers, mask=cleaned)
# filter by validation-tuned area
for each instance:
    if area > min_area and area < max_area:
        encode with pycocotools
```

---

## 8. Metric optimisation

### 8.1 Current state

The only validation metric in the notebook is `calculate_batch_dice_score` (Cell 27, lines 515–526), used to print a mean validation Dice (Cell 29, line 609). The competition, however, uses a **mixed score**: 70 % quantitative (mean Dice, PQ, IoU distributions, one-to-many / many-to-one relations) and 30 % qualitative (report, visual quality, code) (`01_competition_brief.md` lines 91–106).

### 8.2 Gaps / risks

| # | Issue | Why it hurts | Fix |
|---|---|---|---|
| 8.1 | **No Panoptic Quality (PQ) computation** | PQ is the core instance-level metric; not optimising it means the leaderboard score is ignored during development. | Implement the competition PQ formula and track it as the **primary validation metric** for early stopping. |
| 8.2 | **No one-to-many / many-to-one analysis** | The evaluation penalises fragmented and merged predictions (Cell 2, lines 15–18; `01_competition_brief.md` lines 100–122). | Report these relations per validation fold and tune post-processing to minimise them. |
| 8.3 | **Validation Dice is noisy** | Observed validation Dice bounces between 0.3789 and 0.5290 over 5 epochs (Cell 29 output). | This is partly due to the small/imbalanced split; use **5-fold group CV** and PQ for stable selection. |
| 8.4 | **No per-image metric logging** | Cannot identify whether the model fails on small/large/limb filaments. | Log per-image Dice, IoU, TP/FP/FN, and area distribution in `mlflow` / `wandb`. |

### 8.3 PQ implementation target

```python
def panoptic_quality(pred_rles, gt_rles, iou_thr=0.5):
    # match pred/gt masks by IoU > 0.5
    # TP = 1-to-1, penalise 1-to-many and many-to-1
    # PQ = sum_iou_tp / (tp + 0.5*fp + 0.5*fn)
    ...
```

Also implement **mean image Dice**, **mean IoU**, and **mAP@0.5** using `pycocotools.cocoeval` on the validation set in COCO format.

---

## 9. Reproducibility

### 9.1 Current state

- Seed is set to `GLOBAL_SEED = 2026` in Cell 4, lines 74–75.
- `cudnn.deterministic = True` and `cudnn.benchmark = False` (Cell 4, lines 71–72).
- No `requirements.txt`, no `environment.yml`, no pinned Python/PyTorch/CUDA version.
- All file paths are hard-coded to `/kaggle/input/competitions/...` (Cell 7, lines 92–94), so the notebook does **not** run locally or on Colab without edits.

### 9.2 Gaps / risks

| # | Issue | Fix |
|---|---|---|
| 9.1 | **No environment definition** | Create `requirements.txt` (or `environment.yml`) and a `README.md` with install instructions. |
| 9.2 | **No `worker_init_fn`** | Random flips in `__getitem__` are not seeded per worker. Add `worker_init_fn` and `torch.Generator` per worker. |
| 9.3 | **DataParallel non-determinism** | `nn.DataParallel` can produce slightly different gradient order across runs. Prefer `DistributedDataParallel` or single-GPU. |
| 9.4 | **No hyperparameter / config file** | Use `hydra`, `argparse` with a `configs/default.yaml`, or a `json` config to log all settings. |
| 9.5 | **No checkpoint / experiment logging** | Save `best.pth`, `last.pth`, and a `metrics.csv` per run. Use `wandb` or `mlflow`. |
| 9.6 | **No `.gitignore`** | The `data/` folder (700 MB+) and `__pycache__` will be committed. Add `.gitignore` excluding `data/`, `submissions/*.csv`, model checkpoints, and `plots/` scratch. |

---

## 10. Colab training

### 10.1 Current state

There is **no Colab notebook** in `notebooks\`. The only notebook is the Kaggle baseline.

### 10.2 Gaps / risks

- Free Colab gives a T4 with ~15 GB VRAM. Training 2048 × 2048 full images is impossible; even 1024 × 1024 with batch > 2 is tight.
- Data must be downloaded from Kaggle or uploaded to Google Drive every session.
- No persistent checkpointing strategy.

### 10.3 Recommended Colab setup

1. Create `notebooks/colab_filament_training.ipynb`.
2. Mount Drive and cache the dataset at `/content/drive/MyDrive/filament_data/`.
3. Install dependencies in one cell:
   ```python
   !pip install -q pycocotools torchmetrics albumentations timm segmentation-models-pytorch
   ```
4. Use **1024 × 1024** training with batch size 2–4 and `torch.cuda.amp`.
5. Save the best checkpoint to Drive every epoch.
6. Use `wandb` for live metric tracking (optional but recommended).
7. Export the final `submission.csv` to `submissions/`, then download.

For 2048 training, Colab Pro (A100 40 GB) or Kaggle P100 are more realistic.

---

## 11. Kaggle submission

### 11.1 Current state

The baseline writes `submission.csv` in the working directory (Cell 33, line 720) with shape `(3215, 2)` for 180 test images (Cell 33 output, lines 730–732). The columns are `filament_id` and `segmentation_rle`.

### 11.2 Gaps / risks

| # | Issue | Fix |
|---|---|---|
| 11.1 | **CSV saved in working directory, not `submissions/`** | Write to `submissions/submission_<model>_<fold>.csv` and keep a versioned list. |
| 11.2 | **No validation of RLE format** | Load each mask back with `pycocotools.mask.decode` and check shape `(2048, 2048)` and non-negativity. |
| 11.3 | **Pandas default quoting may wrap RLE in quotes** | Add `quoting=csv.QUOTE_NONE` or verify that `,` / `"` do not appear in RLE counts. The header must remain. |
| 11.4 | **No ensemble / fold selection** | Generate one CSV per fold, then merge by averaging probability maps or by selecting the best fold via validation PQ. |
| 11.5 | **No diagnostic before upload** | Run the public `self-evaluation-notebook` (`01_competition_brief.md` line 189) locally on a validation split to estimate private LB before submitting. |

### 11.3 Submission checklist

- [ ] Header exactly `filament_id,segmentation_rle`
- [ ] Each `filament_id` = `<image_id>_<counter>`, unique across all rows
- [ ] RLE `counts` are UTF-8 strings, no quotes, no size prefix
- [ ] Every encoded mask decodes to `(2048, 2048)` `uint8`
- [ ] Final file is a single `.csv` under 1 GB (Kaggle limit)

---

## 12. Code quality

### 12.1 Current state

- All code lives in a single 963 KB notebook with verbose variable names and heavy prose.
- `code\` is **empty**; there are no modular `.py` files.
- Warnings are suppressed globally (`warnings.filterwarnings('ignore')`, Cell 4, line 61).
- No `README`, no `requirements.txt`, no tests, no linting.

### 12.2 Gaps / risks

| # | Issue | Why it hurts for a top-100 / college report | Fix |
|---|---|---|---|
| 12.1 | **No modular project structure** | Cannot run ablation studies, version-control changes, or reuse components. | Create `src/dataset.py`, `src/model.py`, `src/losses.py`, `src/metrics.py`, `src/train.py`, `src/infer.py`. |
| 12.2 | **Hard-coded Kaggle paths** | Notebook cannot run on Colab or local GPU. | Centralise paths in a `config.yaml` or `config.py` with `BASE_PATH` selection. |
| 12.3 | **No logging** | Cannot debug training stalls or metric crashes. | Replace `print()` with `logging` or `loguru`; log config, metrics, and exceptions. |
| 12.4 | **No error handling for missing images** | `cv2.imread` returns `None` and the loader silently inserts a blank 2048×2048 image (Cell 21, lines 348–351). This can hide corrupt data. | Raise a clear `FileNotFoundError` and keep a missing-image manifest. |
| 12.5 | **No type hints or unit tests** | Hard to maintain and peer-review. | Add type hints, docstrings, and a small `tests/test_dataset.py` that checks mask rasterisation. |

### 12.3 Suggested project layout

```
Kaggle_Competition_Mini_Project/
├── configs/
│   └── base.yaml
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_colab_training.ipynb
├── submissions/
│   └── submission_resnet34_baseline.csv
├── plots/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 13. Lab-report requirements

Per the active `AGENTS.md` rule (`C:\Users\srik2\Desktop\College\AGENTS.md`), any college deliverable that must be submitted as a PDF must follow the `lab-report-pdf` skill.

### 13.1 What the rules require

- Use `C:\Users\srik2\.opencode\skills\lab-report-pdf\` with `capture_run.py` + `build_pdf.py`.
- `spec.json` section keys must be only: `heading`, `paragraph` (single string), `bullets`, `table`, `plots`/`screenshots`, `code_file`, `output_file`.
- **Never use `paragraphs` (plural)** — it is silently skipped by `build_pdf.py`.
- No Python `None` in `spec.json`.
- Validate JSON with `python -c "import json; json.load(open(...))"` before building.
- Every code block must come from a real file; every output from a real run; every plot from a real PNG.
- Polished PDF style: Poppins text, Consolas code on dark `#1E1E1E`, running header with Name/Class/RegNo, tables with `#DDE7F5` headers, plots capped ~85 mm and captioned.
- After each working session, append an entry to `C:\Users\srik2\Desktop\College\WORKLOG.md` (goal, deliverable full path, what was done, lessons learned).

### 13.2 How this applies to the Kaggle mini-project

- The final **report PDF** should live at the project root, e.g. `research\Solar_Filament_Segmentation_Report_Srikrishna_O_S.pdf` or, if treated as a single assignment, `Kaggle_Competition_Mini_Project\Assignment_Kaggle_Solar_Filament_Srikrishna_O_S.pdf`.
- All code snippets in the report must be taken from `src/*.py` or the official notebook, not re-typed.
- All tables (EDA counts, ablation results, metric comparisons) must be generated from real runs.
- All plots must be saved to `plots/` first and referenced in `spec.json` under `plots`.
- The profile file at `C:\Users\srik2\.lab_report_profile.json` must be used for the running header.

### 13.3 Gaps / risks

- **No `spec.json` draft yet.** The team should create `research/spec.json` as soon as the first experiment is run.
- **No `WORKLOG.md` entry for this project yet.** Add one now, because the project already has research artefacts.
- **No guarantee that code/output/plot sources are preserved.** If the report is written before the final pipeline is frozen, the PDF will reference stale or non-existent files.

---

## 14. Prioritised action plan for breaking into the top 100

| Priority | Action | Expected impact | Owner / file |
|----------|--------|-----------------|--------------|
| **P0** | **Stop using the public baseline notebook as the main pipeline.** Refactor into `src/` modules and start running real experiments. | Removes leakage, enables ablation, improves code score. | `code/`, `src/` |
| **P0** | **Fix the validation split.** Aggregate by `file_name`; use 5-fold group CV stratified by year / filament-count bins. | Eliminates leakage and gives reliable leaderboard estimate. | `src/dataset.py` |
| **P0** | **Implement competition metrics as validation metrics (PQ, mean Dice, IoU, TP/FP/FN).** | The model will optimise what the leaderboard actually measures. | `src/metrics.py` |
| **P1** | **Raise training resolution to 1024 × 1024 with patch sampling; keep 2048 inference.** | Preserves barbs and small filaments. | `src/dataset.py`, `src/infer.py` |
| **P1** | **Upgrade the model.** Try `smp.DeepLabV3Plus` / `UNet++` with `timm` EfficientNet / ResNeSt / ConvNeXt backbones; replace pseudo-RGB with 1-channel conv1. | Biggest single accuracy gain. | `src/model.py` |
| **P1** | **Add a strong augmentation pipeline** (Albumentations: scale, rotate, elastic, brightness/contrast, CLAHE, noise, blur). | Reduces overfit and improves limb/noise robustness. | `src/dataset.py` |
| **P1** | **Use Focal + Dice + Boundary loss, and train for 50–100 epochs with early stopping on PQ.** | Aligns loss with the metric and with the long-tailed size distribution. | `src/losses.py`, `src/train.py` |
| **P2** | **Add test-time augmentation and simple ensembling across folds.** | Usually +2–5 % PQ in segmentation competitions. | `src/infer.py` |
| **P2** | **Tune post-processing (threshold, morphological kernel, area filter, watershed) on OOF predictions.** | Reduces fragmentation / over-merging penalties. | `src/infer.py` |
| **P2** | **Create Colab and local notebooks, add `requirements.txt`, `.gitignore`, and `README.md`.** | Satisfies reproducibility and code-quality portion of the score. | repo root |
| **P3** | **Prepare the lab-report PDF with `lab-report-pdf`, validate `spec.json`, and append `WORKLOG.md` entries.** | 30 % of the competition score is qualitative; the college report must be polished and real. | `research/spec.json`, `WORKLOG.md` |

---

## 15. Bottom line

The repository currently contains useful **research notes** and a **public baseline notebook**, but it does **not** yet contain a competitive training pipeline. To finish above the 100th position, the team must:

1. Treat the public baseline as a reference, not a solution.
2. Build a modular, reproducible pipeline in `src/`.
3. Fix the validation split to avoid leakage and stratify by physical files.
4. Move from 512 × 512 ResNet-34 + BCE/Dice to a higher-resolution, stronger architecture with Focal/Dice/Boundary loss and heavy augmentations.
5. Optimise for the competition metric (PQ) rather than crude mean Dice.
6. Add TTA, ensembling, and careful post-processing tuned on validation.
7. Produce a polished, real-data lab report PDF and keep the `WORKLOG.md` up to date.

Without these changes, the most likely outcome is a score near the public baseline — well outside the top 100.
