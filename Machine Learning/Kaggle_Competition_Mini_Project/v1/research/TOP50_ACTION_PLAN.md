# Solar Filament Segmentation — Top-50 Code Audit & Action Plan

> Compiled from 6 parallel subagents (code audit, public solutions, Kaggle recipes, T4 Colab optimization, losses/augmentations, metrics/post-processing).  
> This is the **master checklist** for turning the current pipeline into a top-50 solution.  
> **Last updated:** 2026-08-25.

---

## 1. The top-50 gap in one paragraph

The current 3-account Colab setup is a **first-generation baseline**.  It trains plain `unet` with small EfficientNet encoders on downsampled full images, optimizes the wrong metric (relaxed Panoptic Quality), has no mask-aware cropping, no OOF threshold search, no probability-level ensemble, and no boundary-aware training.  The friend’s U-Net++ EfficientNet-B4 5-fold notebook is essentially the same idea but executed better.  To beat it and reach top 50, the code must move to **U-Net++ / DeepLabV3+ with EfficientNet-B4, 3-channel ImageNet input, Focal Tversky + boundary loss, heavy augmentation, a 5-fold probability ensemble, D4/multi-scale TTA, and metric-optimized post-processing (threshold/area search, CRF, hole filling).**  Most of these are code changes, not compute.

---

## 2. Code audit — what is broken or sub-optimal

### 2.1 Critical (must fix before the next training run)

1. **Missing `ToTensorV2()` in `dataset.py` augmentation**  
   The Albumentations pipeline ends with `A.Normalize` and the code manually converts in `__getitem__`.  Add `ToTensorV2()` and remove manual conversion.  This fixes dtype/order bugs and lets Albumentations handle normalization correctly.

2. **Wrong model-selection metric in `train.py`**  
   The code uses `panoptic_quality` to pick the best checkpoint.  It should use `mIoU_multiscale` (or `mIoU_pairwise`) because that is the competition metric.  Without this the “best” model is not the best for the leaderboard.

### 2.2 High impact (do immediately after critical fixes)

3. **1-channel input instead of 3-channel replication**  
   `config.py` sets `IN_CHANNELS = 1` and `smp` adapts the first conv.  Switch to 3-channel input by replicating the grayscale image and using `A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`.  This preserves ImageNet pre-trained first-conv filters.

4. **Wrong architecture default**  
   `config.py` defaults to `unet`.  Switch to `unetplusplus` (or `deeplabv3plus`) with `tu-efficientnet_b4` encoder.

5. **Mask-aware cropping is implemented but never called**  
   `_sample_crop()` in `dataset.py` exists but `__getitem__` resizes the full image instead.  Wire it in: training should crop, validation should resize.

6. **No OOF threshold/area search**  
   `postprocess.py` has `threshold_search` but it is never used.  Run a grid search on OOF predictions for `threshold` and `min_area` using `mIoU_multiscale`.

7. **No probability-map ensemble**  
   There is no code to average probability maps from multiple folds/models.  Inference thresholds single models separately.

8. **Early stopping is not implemented**  
   `PATIENCE` is in `config.py` but `train.py` never uses it.

### 2.3 Medium impact (top-50 polish)

9. **Suboptimal loss** — current `Dice + Focal + BCE` is okay, but `Focal Tversky` or `Asymmetric Focal Tversky` with a boundary/edge component is better for thin filaments.
10. **Missing augmentations** — no `GridMask`, `ISONoise`, `RandomGamma`, `CopyAndPaste`, heavy elastic/grid distortion.
11. **Insufficient TTA** — only 4 flips.  Add D4 (8 transforms) and multi-scale TTA.
12. **No EMA/SWA** — EMA with decay 0.999 is free and improves generalization.
13. **No deep supervision** — add auxiliary heads at decoder levels 2/3 for multi-scale training.
14. **No CRF post-processing** — `pydensecrf` or bilateral edge smoothing.
15. **Memory-inefficient Colab configs** — account 1/2 use batch size 2 on T4; use 1 with accumulation.
16. **No `num_workers` env var** — hard-coded `num_workers=0`.
17. **No warmup LR** — plain CosineAnnealing; use warmup + cosine.

### 2.4 Low impact / end game

18. No limb-darkening correction.
19. No radial distance channel.
20. No explicit hole filling after morphological closing.
21. No no-filament image classifier.
22. No mask-based NMS for overlapping instances.
23. Missing env vars for new features (EMA, TTA scales, CRF, etc.).
24. `requirements.txt` uses `>=` instead of pinned versions.

---

## 3. Concrete step-by-step implementation plan

### Phase 0 — Stop training the wrong thing (do today)

These changes should be made before any new Colab run.  They do not need extra training budget.

#### Step 0.1: Fix `dataset.py` augmentation

```python
from albumentations.pytorch import ToTensorV2

def get_augmentations(image_size, is_train=True):
    ...
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.8),
            A.OneOf([...], p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(gamma_limit=(80,120), p=0.5),
            A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5), p=0.3),
            A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
            A.GridMask(num_grid=4, ratio=0.5, p=0.3),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ToTensorV2(),
        ])
```

Then in `__getitem__` remove `torch.from_numpy(...)` and return Albumentations output directly.

#### Step 0.2: Switch validation metric to `mIoU_multiscale`

In `train.py`:

```python
val_miou_ms = metrics.mean_mIoU_multiscale(pred_masks_list, gt_masks_list)['mIoU_multiscale_micro']
is_best = val_miou_ms > best_miou
```

Also add `FILAMENT_VAL_METRIC` env var so it can be changed without code edits.

#### Step 0.3: 3-channel input

In `dataset.py`:

```python
if img.ndim == 2:
    img = np.stack([img]*3, axis=-1)  # (H, W, 3)
```

In `model.py`:

```python
model = smp.UnetPlusPlus(
    encoder_name=encoder,
    encoder_weights="imagenet" if pretrained else None,
    in_channels=3,
    classes=1,
    activation=None,
)
```

In `config.py`:

```python
IN_CHANNELS = int(os.environ.get("FILAMENT_IN_CHANNELS", "3"))
```

#### Step 0.4: Use `unetplusplus` + `tu-efficientnet_b4`

In `config.py`:

```python
MODEL_NAME = os.environ.get("FILAMENT_MODEL_NAME", "unetplusplus")
ENCODER = os.environ.get("FILAMENT_ENCODER", "tu-efficientnet_b4")
```

#### Step 0.5: Enable mask-aware cropping

In `dataset.py.__getitem__`:

```python
if self.is_train:
    img, mask = self._sample_crop(img, mask)
else:
    img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
```

Also add an env var `FILAMENT_MASK_AWARE_CROP`.

#### Step 0.6: Add early stopping

In `train.py`:

```python
patience_counter = 0
best_miou = -1.0
for epoch in range(1, EPOCHS + 1):
    ...
    if is_best:
        best_miou = val_miou_ms
        patience_counter = 0
        torch.save(...)
    else:
        patience_counter += 1
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

### Phase 1 — Upgrade training (next Colab run)

#### Step 1.1: Add Focal Tversky + Boundary loss

Add to `losses.py`:

```python
import segmentation_models_pytorch.losses as smp_losses

class CombinedTop50Loss(nn.Module):
    def __init__(self, bce_w=0.2, dice_w=0.3, tversky_w=0.3, boundary_w=0.2,
                 tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=4/3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = smp_losses.DiceLoss(mode='binary', from_logits=False)
        self.tversky = smp_losses.TverskyLoss(
            mode='binary',
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            from_logits=True,
        )
        self.boundary = BoundaryLoss(idc=[1])
        self.w = [bce_w, dice_w, tversky_w, boundary_w]

    def forward(self, logits, target, dist_map=None):
        loss = self.w[0] * self.bce(logits, target) \
             + self.w[1] * self.dice(torch.sigmoid(logits), target) \
             + self.w[2] * self.tversky(logits, target) \
             + self.w[3] * self.boundary(torch.sigmoid(logits), dist_map)
        return loss
```

Alternatively use `AsymmetricFocalTverskyLoss` from MONAI.  Compute `pos_weight` from dataset foreground/background pixel ratio (~500x).

#### Step 1.2: Add Copy-Paste / Mosaic / heavy augmentation

Implement copy-paste by extracting filament patches from images with many filaments and pasting onto images with few.  In Albumentations 2.0+ use `A.CopyAndPaste` with instance masks/bboxes or write a custom `CopyPasteAugmentation` class.

Also add to the train transform:

```python
A.OneOf([
    A.ElasticTransform(alpha=50, sigma=120*0.02, alpha_affine=120*0.02, p=1.0),
    A.GridDistortion(p=1.0),
    A.OpticalDistortion(distort_limit=0.4, shift_limit=0.5, p=1.0),
], p=0.8),
```

#### Step 1.3: Multi-scale training

Randomly resize between `0.8*image_size` and `1.25*image_size` before crop.  Use `A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.25), ratio=(0.9, 1.1), p=1.0)`.

#### Step 1.4: Add EMA/SWA

```python
from torch.optim.swa_utils import AveragedModel, update_bn

ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

for images, masks in train_loader:
    ...
    optimizer.step()
    ema_model.update_parameters(model)

# Validation / inference use ema_model
update_bn(train_loader, ema_model)
```

For SWA, start averaging at 75% of epochs with a higher LR.

#### Step 1.5: Warmup + cosine LR scheduler

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=MIN_LR)
scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])
```

#### Step 1.6: FP16 AMP + gradient accumulation

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer.zero_grad()
for i, (images, masks) in enumerate(train_loader):
    with autocast():
        logits = model(images)
        loss = criterion(logits, masks) / accumulation_steps
    scaler.scale(loss).backward()
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

#### Step 1.7: T4 memory safety

- batch size = 1 for `unetplusplus` + `tu-efficientnet_b4` at 1024
- gradient accumulation = 8–16 (effective batch 8–16)
- `decoder_use_batchnorm='inplace'` and `pip install inplace-abn --no-build-isolation`
- enable encoder gradient checkpointing:
  ```python
  if hasattr(model.encoder, 'set_grad_checkpointing'):
      model.encoder.set_grad_checkpointing(True)
  ```
- `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'`
- call `torch.cuda.empty_cache()` after validation

---

### Phase 2 — 5-fold full CV and probability ensemble

#### Step 2.1: Save OOF probability maps

For each fold, after training run inference on the validation split and save `npy` float16 probability maps per image (`submissions/oof_probs_fold_{k}/<image_id>.npy`).

#### Step 2.2: Train 5 folds

Use 5 Colab accounts (or 3 accounts sequentially).  Each account uses a different `FILAMENT_VAL_FOLD` (0–4).  All use the same base config from Phase 0/1 but different random seeds/backbones for diversity:

| Fold | Architecture | Encoder | Resolution | TTA |
|---|---|---|---|---|
| 0 | unetplusplus | tu-efficientnet_b4 | 1024 | 4 flips |
| 1 | unetplusplus | tu-efficientnet-b3 | 1024 | 4 flips |
| 2 | deeplabv3plus | tu-efficientnet-b4 | 1024 | 4 flips |
| 3 | unetplusplus | tu-resnet50 | 1024 | 4 flips |
| 4 | unetplusplus | tu-efficientnet-b5 | 768 | 4 flips |

#### Step 2.3: Average probability maps

```python
import os, glob, numpy as np

def ensemble_prob_maps(image_id, prob_dirs):
    avg = None
    for d in prob_dirs:
        path = os.path.join(d, f"{image_id}.npy")
        if os.path.exists(path):
            p = np.load(path).astype(np.float32)
            if avg is None:
                avg = p
            else:
                avg += p
    avg /= len(prob_dirs)
    return avg
```

For test set, do the same: each fold produces `test_prob_fold_{k}/<image_id>.npy`, then average.

---

### Phase 3 — Inference and post-processing

#### Step 3.1: D4 + multi-scale TTA

In `infer.py`:

```python
def d4_tta(model, image):
    # image: (1, 3, H, W) tensor
    preds = [model(image)]
    for dim in ([-1], [-2], [-1,-2]):
        hflip = torch.flip(image, dims=dim)
        p = model(hflip)
        for d in dim:
            p = torch.flip(p, dims=[d])
        preds.append(p)
    for k in [1, 2, 3]:
        rot = torch.rot90(image, k=k, dims=[-2,-1])
        p = model(rot)
        p = torch.rot90(p, k=-k, dims=[-2,-1])
        preds.append(p)
    # transpose
    tr = torch.transpose(image, -1, -2)
    p = model(tr)
    p = torch.transpose(p, -1, -2)
    preds.append(p)
    return torch.stack(preds).mean(dim=0)
```

For multi-scale, resize image to `0.8x, 1.0x, 1.25x`, run D4 at each scale, resize probability back to 2048, average.

#### Step 3.2: Threshold / area grid search on OOF

```python
from code import metrics
from skimage.morphology import remove_small_objects, remove_small_holes

def optimize_threshold_area(oof_probs, gt_instances, thresholds, min_areas):
    best = -1
    best_params = (0.5, 100)
    for thr in thresholds:
        for min_area in min_areas:
            scores = []
            for prob, gt in zip(oof_probs, gt_instances):
                binary = (prob > thr).astype(np.uint8)
                # connected components + area filter
                labeled = skimage.measure.label(binary, connectivity=2)
                filtered = remove_small_objects(labeled, min_size=min_area)
                # convert back to binary mask
                pred_mask = (filtered > 0).astype(np.uint8)
                pred_mask = remove_small_holes(pred_mask, area_threshold=100)
                # split by connected components for pairwise IoU
                pred_instances = [m for m in instance_masks_from_label(filtered)]
                s = metrics.mean_mIoU_multiscale([pred_instances], [gt_instances])['mIoU_multiscale_micro']
                scores.append(s)
            mean = np.mean(scores)
            if mean > best:
                best = mean
                best_params = (thr, min_area)
    return best_params, best
```

Use `thresholds = np.arange(0.30, 0.75, 0.02)` and `min_areas = [50, 100, 200, 500]`.

#### Step 3.3: CRF on averaged probabilities

Install in Colab:

```bash
!pip install -U cython
!pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

Function in `postprocess.py`:

```python
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_dense_crf(image, prob, n_iter=5):
    h, w = image.shape[:2]
    # image must be uint8 (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    image = image.astype(np.uint8)
    # prob (H, W) -> (2, H, W)
    u = np.stack([1 - prob, prob], axis=0).astype(np.float32)
    unary = unary_from_softmax(u)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(3,3), compat=3,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13),
                           rgbim=image, compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(n_iter)
    Q = np.array(Q).reshape((2, h, w))
    return Q[1]
```

#### Step 3.4: Morphological cleaning

```python
from skimage.morphology import opening, closing, remove_small_objects, remove_small_holes, disk

def clean_mask(mask, open_r=1, close_r=2, min_hole=100, min_obj=100):
    if open_r > 0:
        mask = opening(mask, disk(open_r))
    if close_r > 0:
        mask = closing(mask, disk(close_r))
    mask = remove_small_holes(mask, area_threshold=min_hole)
    mask = remove_small_objects(mask, min_size=min_obj)
    return mask.astype(np.uint8)
```

#### Step 3.5: No-filament classifier

```python
def is_empty_image(prob, img, area_threshold=50, prob_threshold=0.1):
    # rule + learned threshold
    if prob.max() < prob_threshold:
        return True
    binary = (prob > prob_threshold).astype(np.uint8)
    labeled = skimage.measure.label(binary, connectivity=2)
    return all(r.area < area_threshold for r in skimage.measure.regionprops(labeled))
```

If empty, the image should have **no RLE rows** in the submission (COCO convention).

---

### Phase 4 — Advanced (if still not top 50)

1. **EdgeAttNet**: adapt the official EG-MHSA module from https://github.com/dasjar/EdgeAttNet as a drop-in replacement for the U-Net++ decoder attention.
2. **Extra data**: pre-train on full MAGFiLO (1,593 images) then fine-tune on the 704 competition images.
3. **Pseudo-labeling**: ensemble the 5-fold models on the 180 test images, keep pixels where all models agree > 0.9, add as pseudo-labeled training data, re-train.
4. **Detection + segmentation two-stage**: YOLO / CondInst for filament detection, then U-Net for pixel mask.
5. **Temporal consistency**: if you can group test images by date, enforce consistency across nearby frames.

---

## 4. Recommended Colab account configs

| Account | Fold | Model | Encoder | Resolution | Epochs | Batch | Accum | TTA |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | unetplusplus | tu-efficientnet_b4 | 1024 | 20 | 1 | 8 | 4 flips |
| 2 | 1 | deeplabv3plus | tu-efficientnet_b4 | 1024 | 20 | 1 | 8 | 4 flips |
| 3 | 2 | unetplusplus | tu-efficientnet_b3 | 1024 | 25 | 1 | 8 | 4 flips + 3 scales |
| 4 (new) | 3 | unetplusplus | tu-resnet50 | 1024 | 20 | 1 | 8 | 4 flips |
| 5 (new) | 4 | unetplusplus | tu-efficientnet_b5 | 768 | 30 | 1 | 16 | 4 flips |

**All accounts:**

- `IN_CHANNELS=3`, `3-channel input`
- `VAL_METRIC=mIoU_multiscale`
- `LOSS=CombinedTop50Loss` (Focal Tversky + Boundary)
- `PATIENCE=7`, `LR=1e-4`, `WD=1e-5`, `WARMUP=5`
- `AMP=1`, `GRADIENT_CHECKPOINTING=1`, `CUDA_ALLOC_CONF=expandable_segments:True`

---

## 5. Key public resources to use

| Resource | URL | Use |
|---|---|---|
| Kaggle competition | https://kaggle.com/competitions/filament-segmentation-2026 | data, leaderboard, metric |
| EdgeAttNet repo | https://github.com/dasjar/EdgeAttNet | SOTA edge attention architecture |
| EdgeAttNet paper | https://arxiv.org/abs/2509.02964 | MIoU metric details |
| Flat U-Net | https://zenodo.org/records/14610155 | lightweight fallback |
| Compound U-Net | https://doi.org/10.5281/zenodo.17230604 | training/inference/post scripts |
| clDice | https://github.com/jocpae/clDice | topology-aware loss/metric |
| PySODMetrics | search for `PySODMetrics` | MIoU reference implementation |
| MAGFiLO data page | https://www.mlecofi.net/magfilo | full dataset, docs |
| augmentation-engine (PyPI) | https://pypi.org/project/augmentation-engine | chirality-aware augmentation |
| ResNet-UNet starter | https://github.com/avikds/Kaggle-Notebooks-Avik/blob/main/solar-filament-segmentation-resnet-unet.ipynb | baseline notebook |

---

## 6. Final checklist before each submission

- [ ] All 5 folds trained and `best_fold_{k}.pth` downloaded.
- [ ] OOF probability maps saved for all folds.
- [ ] Threshold and `min_area` optimized on combined OOF.
- [ ] Test probability maps generated by each fold and averaged.
- [ ] CRF / morphological cleaning applied to the averaged probability map.
- [ ] Thresholded at **2048** full resolution, never at low res.
- [ ] Disk mask applied (off-limb pixels zeroed).
- [ ] Empty images return no RLE rows.
- [ ] RLE `counts` decoded to UTF-8 string.
- [ ] Submission validated: all image IDs present, RLE round-trips correctly.

---

## 7. What to do right now

1. **Let the 3 current runs finish** — their models are your first ensemble.
2. **Apply Phase 0 fixes locally** while Colab trains: `dataset.py` ToTensor, `train.py` metric, `model.py` 3-channel, `config.py` defaults.
3. **Generate a new set of no-Drive notebooks** with Phase 0/1 configs and new payloads.
4. **After current runs finish, train the 5-fold ensemble** using the new config.
5. **Run OOF threshold/area search and build the averaged test submission**.

---

## 8. Expected score improvement

- Current plain U-Net at 512/768 with PQ metric: likely **0.45–0.55 MIoU**.
- After Phase 0 + 1 (better architecture, metric, loss, augmentation): **0.55–0.63 MIoU**.
- After 5-fold ensemble + TTA + threshold search: **0.62–0.68 MIoU**.
- After CRF + EdgeAttNet-ish edge attention + pseudo-labeling: **0.67–0.73 MIoU**.

Top 50 is likely **≥ 0.65**, so the path is realistic but requires disciplined execution.

