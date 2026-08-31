# Public Baseline Summary: Solar Filament Segmentation — ResNet-U-Net Pipeline

A concise digest of the Kaggle notebook `public_baseline_resnet_unet.ipynb` for the Solar Filament Segmentation 2026 competition.

---

## 1. Notebook title and source URL

- **Title in notebook:** *Automated Solar Filament Delineation and Chirality Segmentation Framework — A High-Fidelity Deep Learning Pipeline for GONG H-Alpha Solar Imagery*
- **Source / public Kaggle notebook:** https://www.kaggle.com/code/avikdas567/solar-filament-segmentation-resnet-u-net-pipeline?scriptVersionId=334756212
- **Competition context:** Solar Filament Segmentation 2026, based on the **MAGFiLO v1.0** dataset of manually annotated GONG H-alpha solar filaments. The competition is also highlighted as the IEEE Big Data Cup 2026 track on pixel-precise filament segmentation. ([MAGFiLO Kaggle dataset](https://www.kaggle.com/datasets/esairlab/magfilo-v1-0-segmentation-of-solar-filaments); [Scientific Data paper](https://doi.org/10.1038/s41597-024-03876-y); [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/))

---

## 2. Environment and dependencies

- **Runtime:** Kaggle dual NVIDIA T4 GPUs (~32 GB combined VRAM), PyTorch 2.10.0+cu128, CUDA 12.8.
- **Key libraries used:**
  - `torch`, `torchvision`, `torch.nn`, `torch.optim`
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `opencv-python` (`cv2`), `PIL`, `scipy.ndimage`
  - `sklearn.model_selection.KFold` *(imported but not used in the shown cells)*
  - `pycocotools.mask` (for COCO RLE encoding)
- **Reproducibility seeding:** `GLOBAL_SEED = 2026`; determinism is enforced on `random`, `numpy`, `torch`, and `cudnn`.

> Note: The notebook sets `cudnn.deterministic = True` and `cudnn.benchmark = False`, which aids reproducibility but can hurt throughput.

---

## 3. Data loading and COCO JSON structure

### File paths (Kaggle input layout)

```python
TRAIN_JSON_PATH = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG_DIR = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images"
TEST_IMG_DIR  = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
```

### Top-level COCO JSON keys

The train JSON has the standard COCO sections: `info`, `licenses`, `categories`, `images`, `annotations`.

### Dataset size

| Entity | Count |
|--------|-------|
| Images / observation frames | **1,154** |
| Filament annotations | **8,199** |
| Categories | **4** |
| Approx. filaments per image | ~7.1 |

### Categories

The four chirality / identification classes are:

1. `Left`
2. `Right`
3. `Unidentifiable`
4. `Ambiguous`

(These names appear in the EDA countplot.)

### Annotation structure

Each annotation includes:

- `image_id` — links to the `images` list
- `category_id` — chirality / confidence class
- `bbox` — `[x, y, w, h]` (floating-point minimum bounding box)
- `area` — pixel area
- `segmentation` — list of closed polygons (flat `[x0, y0, x1, y1, ...]`); each polygon has `>= 6` values
- `id` — unique annotation id

The notebook notes that the same image file may be indexed under **multiple `image_id`s** (multi-annotator setup), so the data loader groups annotations by `image_id` into a single combined mask.

### `SolarFilamentDataset`

- Loads each JPEG as **grayscale** (`cv2.IMREAD_GRAYSCALE`).
- **Rasterizes** polygon segmentations into a binary mask using `cv2.fillPoly`.
- **Resizes** images and masks from `2048 × 2048` → `512 × 512`:
  - Image: `INTER_AREA`
  - Mask: `INTER_NEAREST`
- Normalizes image tensor to `[0, 1]` and adds a channel dimension.
- Training augmentations: **random horizontal + vertical flips** (`np.fliplr` / `np.flipud`).
- Test mode returns only the image and the filename/id.

---

## 4. Model architecture

### Class: `ResNet34UNet`

A U-Net-style encoder–decoder with a **ResNet-34** backbone from `torchvision`.

- **Input:** single-channel `512 × 512` H-alpha image.
- **Preprocessing trick:** replicates the grayscale channel 3 times to make pseudo-RGB:

  ```python
  x = torch.cat([x, x, x], dim=1)
  ```

- **Backbone:** `models.resnet34(weights=models.ResNet34_Weights.DEFAULT)` (ImageNet-pretrained).

- **Encoder stages**

  | Stage | Output resolution | Channels |
  |-------|------------------|----------|
  | `init_conv` (`conv1+bn1+relu`) | 256×256 | 64 |
  | `maxpool` | 128×128 | 64 |
  | `layer1` | 128×128 | 64 |
  | `layer2` | 64×64 | 128 |
  | `layer3` | 32×32 | 256 |
  | `layer4` | 16×16 | 512 |

- **Decoder / upsampling path**

  - Uses `nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)`.
  - Skip connections concatenate encoder features at each scale.
  - Final decoder blocks use `Conv2d → BatchNorm2d → ReLU`.
  - Output head: `Conv2d(32, 1, kernel_size=1)`.

### Loss functions

Two loss functions are defined/used:

1. **`ScientificDiceLoss`** — Dice loss with `smooth = 1e-5`:

   ```python
   class ScientificDiceLoss(nn.Module):
       def forward(self, target_predictions, ground_truth):
           probabilities = torch.sigmoid(target_predictions)
           ...
           return 1.0 - dice_coefficient
   ```

2. **`nn.BCEWithLogitsLoss`** — standard binary cross-entropy.

The training loop combines them with **0.6 × Dice + 0.4 × BCE**:

```python
loss_dice = criterion_dice(outputs, masks)
loss_bce  = criterion_bce(outputs, masks)
total_loss = 0.6 * loss_dice + 0.4 * loss_bce
```

> Note: The markdown text in the notebook also references **Focal Loss** mathematically, but the running code uses `BCEWithLogitsLoss`, not Focal Loss.

---

## 5. Training setup

### Data split

```python
train_size = int(0.85 * len(train_dataset))
val_size   = len(train_dataset) - train_size
dataset_train_split, dataset_val_split = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(GLOBAL_SEED)
)
```

- **Train samples:** 980
- **Validation samples:** 174
- This is a **single random 85/15 split**; `KFold` is imported but not used.

### DataLoaders

```python
train_loader = DataLoader(dataset_train_split, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(dataset_val_split,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
```

### Optimizer, scheduler, and AMP

- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`
- **Scheduler:** `CosineAnnealingLR(optimizer, T_max=10)`
- **Mixed precision:** `autocast()` + `GradScaler()`
- **Multi-GPU:** `nn.DataParallel` when more than one GPU is available.
- **Epochs:** `NUM_EPOCHS = 5`

### Observed training log (from notebook outputs)

| Epoch | Duration | Train Loss | Validation Dice |
|-------|----------|------------|-----------------|
| 1 | 25.4 s | 0.7040 | 0.4510 |
| 2 | 22.4 s | 0.6250 | 0.5033 |
| 3 | 22.4 s | 0.5642 | 0.3789 |
| 4 | 23.1 s | 0.4794 | 0.5290 |
| 5 | 23.5 s | 0.3670 | 0.4821 |

The validation Dice is noisy and still well below 0.6, suggesting the model is under-trained for this task.

### Validation metric

```python
def calculate_batch_dice_score(predictions, ground_truth, threshold=0.5):
    activated_predictions = (torch.sigmoid(predictions) > threshold).float()
    ...
    return (2.0 * intersection / (union + 1e-8)).item()
```

---

## 6. Post-processing

The test/inference pipeline converts a **single semantic mask** into **per-filament RLE instances**:

1. **Probability → binary:** threshold `0.45` on `torch.sigmoid(logits).squeeze()`.
2. **Upscale:** resize from `512 × 512` back to original `2048 × 2048` with `cv2.INTER_NEAREST`.
3. **Morphological close:** `5 × 5` ellipse structuring element (`cv2.MORPH_CLOSE`) to repair breaks.
4. **Connected components:** `scipy.ndimage.label` with an **8-connected** neighborhood (`structure=np.ones((3,3))`).
5. **Area filter:** discard components smaller than **250 pixels** to reduce fragmentation.
6. **RLE encoding:** `pycocotools.mask.encode` on a Fortran-ordered `(2048, 2048)` uint8 mask; `counts` decoded to a UTF-8 string.
7. **Submission format:** `filament_id` = `{image_base}_{instance_counter}` (e.g. `20110120105534Ch_1`), plus `segmentation_rle` column.
8. **Empty frame fallback:** if no instance remains, encode a blank `2048 × 2048` mask and append `{image_id}_1`.

Output: `submission.csv` (in the notebook the resulting DataFrame has shape **(3215, 2)** for 180 test images).

---

## 7. Key code snippets and useful functions

### COCO ingestion

```python
with open(TRAIN_JSON_PATH, 'r') as file_handle:
    coco_database = json.load(file_handle)

df_images = pd.DataFrame(coco_database['images'])
df_annotations = pd.DataFrame(coco_database['annotations'])
df_categories = pd.DataFrame(coco_database['categories'])
```

### Polygon rasterization

```python
unified_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
for target_annotation in associated_annotations:
    polygon_sequences = target_annotation['segmentation']
    for sequence in polygon_sequences:
        if len(sequence) >= 6:
            vector_points = np.array(sequence, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(unified_mask, [vector_points], 1.0)
```

### Training step with AMP

```python
optimizer.zero_grad()
with autocast():
    outputs = segmentation_network(frames)
    loss_dice = criterion_dice(outputs, masks)
    loss_bce = criterion_bce(outputs, masks)
    total_loss = 0.6 * loss_dice + 0.4 * loss_bce

scaler.scale(total_loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Post-processing and RLE

```python
binary_semantic_mask = (probabilities > 0.45).astype(np.uint8)
upscaled_mask = cv2.resize(binary_semantic_mask, (2048, 2048), interpolation=cv2.INTER_NEAREST)

structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed_mask = cv2.morphologyEx(upscaled_mask, cv2.MORPH_CLOSE, structuring_element)

labeled_instances, total_components = ndimage.label(closed_mask, structure=np.ones((3, 3)))

for comp_idx in range(1, total_components + 1):
    single_instance_mask = (labeled_instances == comp_idx).astype(np.uint8, order='F')
    if single_instance_mask.sum() < 250:
        continue
    rle_encoded = mask_util.encode(single_instance_mask)
    rle_string = rle_encoded['counts'].decode('utf-8')
    submission_registry.append({
        'filament_id': f"{img_id_base}_{instance_counter}",
        'segmentation_rle': rle_string
    })
```

---

## 8. Limitations and things to improve

### Limitations

1. **Very short training:** only **5 epochs**; validation Dice is noisy and plateaus below 0.55.
2. **No real cross-validation:** `KFold` is imported but unused; the split is a single random 85/15 hold-out.
3. **Resolution downscaling:** training and inference are run at `512 × 512`, which may collapse small filaments (< 500 px in the original mask). The data distribution has a long tail of small targets.
4. **Pseudo-RGB channel replication:** the single-channel image is duplicated to 3 channels so ResNet-34 ImageNet weights can be used, but this wastes capacity and does not re-initialize the first conv layer for true grayscale learning.
5. **Loss mismatch with text:** the notebook’s markdown proposes Focal Loss, but the implemented loss is `0.6*Dice + 0.4*BCE`.
6. **No class-aware / instance-aware head:** the model only predicts a single binary filament mask; chirality categories are loaded during analysis but not used by the segmentation model.
7. **Heuristic post-processing:** the `0.45` probability threshold, `5 × 5` closing, and `250 px` area filter are hand-picked and likely sub-optimal across the whole test set.
8. **No competition metric feedback:** the training loop only reports image-level Dice, not the competition IoU/AP/hit-rate style metrics.
9. **No test-time augmentation (TTA), no ensembling, no COCO mask-based evaluation.**

### Suggested improvements

- Train for **more epochs** (e.g. 50–100) with early stopping on the competition-style validation metric.
- Use **stratified / 5-fold cross-validation** and/or hold-out by time/instrument to get robust performance estimates.
- Experiment with **patch-based training** or train at higher resolution (e.g. 1024×1024, 2048×2048 with gradient checkpointing) to preserve thin barbs.
- Replace the first ResNet conv layer with a **1-channel input** and copy/fine-tune ImageNet weights, or use a grayscale-pretrained backbone.
- Try alternative decoders (`DeepLabV3+`, `UNet++`, `Swin-UNETR`) and encoders (`ResNet-50`, `EfficientNet`, `Swin Transformer`).
- Add **TTA** and ensemble multiple folds / models.
- Tune post-processing thresholds and morphological parameters with a held-out validation set using the actual competition metric.
- Include **chirality / category information** if the final task benefits from multi-class or auxiliary supervision.
- Add a proper COCO evaluation loop (`pycocotools.cocoeval`) to validate IoU and instance-level metrics before submission.

---

**Sources**

- Public Kaggle notebook: https://www.kaggle.com/code/avikdas567/solar-filament-segmentation-resnet-u-net-pipeline?scriptVersionId=334756212
- MAGFiLO v1.0 dataset documentation: https://www.kaggle.com/datasets/esairlab/magfilo-v1-0-segmentation-of-solar-filaments and https://www.mlecofi.net/magfilo
- MAGFiLO paper (Scientific Data, 2024): https://doi.org/10.1038/s41597-024-03876-y
- IEEE Big Data Cup 2026 — Solar Filament Segmentation: https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/
