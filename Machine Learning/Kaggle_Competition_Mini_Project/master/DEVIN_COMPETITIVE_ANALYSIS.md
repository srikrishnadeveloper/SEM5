# Devin Competitive & Architectural Intelligence Report

**Project:** Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup & Kaggle)  
**Authority:** ChatGPT Master / Antigravity execution  
**Research scope:** Winning thin-curvilinear instance-segmentation recipes, YOLOv8-seg vs YOLO11-seg, metric-aligned ensembling/TTA, and Panoptic-Quality calibration.  
**Date compiled:** 2026-09-11

---

## 0. Executive Verdict

The project’s **0.360** public leaderboard (LB) score on the re-scored Panoptic Quality (PQ) board is already at the honest frontier; the next realistic production target is **0.50+**.  This report’s central conclusion is that the largest remaining gains are **not** from switching to YOLO11, but from four lower-risk, metric-aligned interventions:

1. **Keep mask resolution as high as the architecture allows** — the Ultralytics YOLO-seg mask prototype is hard-coded to **1/4 of the input resolution** and is the primary boundary-degradation bottleneck for 2–6 px filaments.
2. **Never ensemble native 2048 masks with downsampled/cascade masks**, and never average binarized masks — fuse in **logit/probability space** with one threshold.
3. **Replace the current two-flip TTA with true probability-map fusion** (or disable it if only binary masks are accessible), because concatenating already-thresholded masks creates duplicate FPs.
4. **Calibrate confidence, NMS IoU, and `min_area` against the local PQ evaluator**, not COCO mAP, using the expected marginal contribution to `PQ = SQ × RQ`.

YOLO11-seg is a parameter-efficient architectural refresh (C3k2 + C2PSA attention), but its segmentation head, prototype resolution, and loss are identical to YOLOv8-seg.  It is worth an A/B run, but only after the resolution, TTA, and calibration levers are exhausted.

---

## 1. Winning Methodologies for Thin Curvilinear Instance Segmentation

### 1.1 The competitive landscape for filament-shaped objects

| Competition / Task | Winning strategy | Resolution | Key heads/losses | Post-processing | Relevant to us |
|---|---|---|---|---|---|
| **Sartorius Cell Instance Segmentation** (neurite filaments) [^1][^2][^3] | YOLOX / EffDetD7 detection at 1536, then UPerNet/Swin mask head + Mask R-CNN mask head | **1536 px**, LIVECell pretrain | UPerNet, Mask R-CNN head, BCE/Dice-like mask loss | **Probability-mask averaging** after ROIAlign; bbox-driven mask; *bilinear* mask resize, not nearest | Yes — thin elongated cells, bbox→mask refinement |
| **HuBMAP — Hacking the Kidney** [^4][^5] | Top-5 all **U-Net** families; tiling + heavy TTA + DenseCRF | Patch 2560 scaled to 1280, **512 px overlap** | U-Net / FPN / LinkNet, EfficientNet-B4 | DenseCRF on patch probabilities, 3-fold TTA (H/V/transpose), fixed threshold 0.4 | Yes — medical curvilinear FTUs, CRF crisping |
| **HuBMAP — Hacking the Human Vasculature** [^6] | Cascade R-CNN, Mask R-CNN, HTC, + YOLOv6m; WBF boxes + **mean mask averaging** | Instance crops | Cascade/Mask head | `score = conf_bbox × mean(mask[mask>0.5])`; area > 80 px | Yes — explicit thin-vessel mask scoring |
| **SenNet + HOA Blood Vessel** [^7][^8] | 2D Unet++ (EffNet-B5 + SCSE) + 3D nnU-net/DynUNet; d4 TTA | 2D/3D patch, original res | **Boundary DoU loss**, Lovasz loss, focal/Dice | Probability ensemble, d4 TTA, small-FP filtering | Yes — **Boundary DoU / clDice / topology losses** |
| **Sartorius 8th place** [^9] | Cascade Mask R-CNN + pseudo-labels + **Weighted Masks Fusion (WMF)** | 1024–1536 | Mask R-CNN | WMF = WBF extended to masks; confidence-weighted mask fusion | Yes — direct mask-ensemble recipe |

### 1.2 Sartorius — the canonical “thin neurite” win

The 1st-place solution by *rist/takuoko/tascj* (Rist et al.) is the most instructive direct analogue [^1]:

* **Detection backbone:** YOLOX with CB-DBS-FPN at **1536×1536** and CSPDarknet-YOLOX-PAFPN, plus EfficientDet-D7, all pretrained on the large **LIVECell** dataset.
* **Mask refinement:** Two Mask R-CNNs and four **UPerNet** segmentation heads (Swin / ResNet-101) were trained on crops generated from **ground-truth bounding boxes**.
* **Critical detail — mask sampling:**  
  > “We used ROIAlign to crop & resize input image & mask and used `grid_sample` to paste a prediction to its bbox location **before thresholding**. It is also important to resize the training target (mask) using **bilinear interpolation then threshold it**, instead of using nearest-neighbor interpolation.”

  This is exactly the mistake the project avoids in `infer.py` (bilinear probability resize, threshold at full resolution) and the opposite of the `INTER_NEAREST` upsampling that was identified as a public-baseline error.
* **Ensemble:** Boxes were fused with **Weighted Boxes Fusion (WBF)**; the final masks were produced by **averaging raw probabilities** from UPerNet and Mask R-CNN and thresholding once.

**Transferable principle:** for thin objects, **detection quality and high-resolution mask refinement are coupled**.  A strong box is necessary to crop a high-quality mask; a poor box truncates long filaments (this was the failure mode of the V8.1 cascade).  The winning recipe is *box-driven, probability-averaged, one-threshold*.

### 1.3 HuBMAP kidney / vasculature — tiling, TTA, and CRF

The 2021 HuBMAP top solutions were overwhelmingly **U-Net-based semantic/instance hybrids** [^4][^5]:

* **RobinSmits/KaggleHuBMAP** (0.9437 private LB, 4th/5th place): ensemble of U-Net, FPN, and LinkNet with **EfficientNet-B4**, patch 2560 scaled to 1280 with **512 px overlap**, 3-fold TTA (H-flip, V-flip, transpose), and **DenseCRF** on probabilities.
* The published comparison of the top-5 algorithms showed that **all five used U-Net** and scored mean Dice ~0.88 on kidney glomeruli [^5].

The 2023 *Hacking the Human Vasculature* 4th-place solution (SDSRV.AI, GoN) explicitly described the mask-ensemble logic [^6]:

> “We utilize **Weighted Boxes Fusion (WBF)** … For masks generated by the ensemble of models we utilize the **mean average approach**.  For post-processing we filter out small instances (area < 80 px) and refine instance score using `score_instance = score_bbox × mean(mask[mask > 0.5])`.”

**Transferable principle:**
* **TTA** (flips + transpose) is safe when the *probabilities* are averaged before thresholding; it is dangerous when per-view binary masks are combined.
* **DenseCRF** or a learned boundary refinement can recover crisp boundaries from low-res probability maps — directly relevant because YOLO’s prototype mask is also low-resolution.
* **Instance score = box confidence × mask quality** is a strong calibration recipe and can be reproduced in the project’s `InstanceCalibrator`.

### 1.4 SenNet blood vessel — boundary & topology losses

The 2024 SenNet + HOA *Hacking the Human Vasculature in 3D* 4th-place solution (Igor Krashenyi) [^7] and the published post-mortem [^8] make a strong case for **boundary-aware and topology-aware losses**:

> “The best results I was able to get using the efficientnet family models with **UnetPlusPlus decoder and SCSE attention** … The 4th-place solution used a mixture of 2D and 3D models with **d4 TTA** and a 2-fold setup.  The 2D model used a **Boundary DoU loss** … the 3D model was heavily inspired by nnU-Net.”

Key losses for thin, connected structures:

| Loss / module | What it optimizes | When it helps | Reference |
|---|---|---|---|
| **clDice / soft-clDice** | Topology (skeleton overlap) | Prevents broken filaments, preserves connectivity | Shit et al., CVPR 2021 [^10] |
| **Boundary Loss / Boundary DoU** | Boundary distance, not just region | Reduces dilation, helps thin 2–6 px edges | Kervadec et al., MIDL 2019 [^11]; Zhang et al. [^12] |
| **Lovász-Softmax** | Direct surrogate for IoU | Better than BCE/Dice for high-IoU matching | Berman et al., CVPR 2018 [^13] |
| **PointRend** | Adaptive high-res boundary points | Renders crisp boundaries without full high-res feature maps | Kirillov et al., CVPR 2020 [^14] |
| **DenseCRF** | Fully-connected CRF on probabilities | Removes speckle, snaps to edges | Krähenbühl & Koltun, 2011 [^15] |

**For the Solar Filament task:**
* clDice is attractive because filaments are topological networks; however, the YOLO mask head uses a **BCE + Dice loss** (`BCEDiceLoss` in `ultralytics/utils/loss.py` [^16]) and cannot be directly swapped without forking the trainer.
* The most practical short-term win is to add a **post-train boundary refinement** or a **DenseCRF / learned edge-aware head** inspired by EdgeAttNet’s EG-MHSA [^17] on top of the YOLO output.
* PointRend is theoretically ideal but requires either a custom mask head or a second stage that adaptively samples boundary points around the YOLO mask edges.

### 1.4.1 Solar-filament-specific baselines: EdgeAttNet, Flat U-Net, Compound U-Net

These models were designed directly for full-disk Hα filament segmentation and are the strongest published results on MAGFiLO-style data:

| Model | Task | Reported metric | Value | Params | Source |
|---|---|---|---|---|---|
| **EdgeAttNet** | MAGFiLO filament segmentation | mIoU_multiscale | **0.7032** | 22.7 M | Solomon et al., arXiv:2509.02964 [^17] |
| **U-Net + MHSA (with PE)** | MAGFiLO | mIoU_multiscale | 0.6601 | ~35 M | EdgeAttNet paper [^17] |
| **Flat U-Net** | Full-disk Hα (non-MAGFiLO splits) | DSC | **0.82** | Lightweight | Zhu et al., *ApJ* 980, 176 (2025) [^36] |
| **Compound U-Net** | MHAS / HAS | Test IoU / F1 | 0.677 / 0.803 (MHAS), 0.714 / 0.831 (HAS) | — | Zenodo artifact [^37] |

**Important caveats for the Kaggle task:**

* **EdgeAttNet** is evaluated with `mIoU_pairwise` and `mIoU_multiscale` [^17], not Kirillov PQ at `IoU > 0.5`.  Its multiscale metric averages the intersection ratio across downsampled scales, which is more forgiving of thin, barbed boundaries than a hard 0.5 IoU match.
* **Flat U-Net** and **Compound U-Net** report DSC / IoU / F1 on *semantic* or smaller non-MAGFiLO test sets, not on the 180-image instance-PQ task.
* These papers nevertheless establish the value of **edge-guided attention**, **lightweight attention blocks** (SCA/CSA), and **barb-aware heads** for solar filaments.  The YOLO-seg pipeline can borrow the *edge-aware attention* idea by attaching a small EdgeAttNet-like refiner or by training an auxiliary edge map, without replacing the whole detector.

### 1.5 Synthesis — what Grandmasters do

1. **Resolution first:** train and infer at the largest resolution that fits in VRAM; avoid downsampled cascades.
2. **Box → mask decoupling:** a strong detector provides RoIs, and a high-resolution segmentation head refines each RoI (Mask R-CNN / UPerNet / crop refiner).
3. **Probability fusion, not mask union:** ensemble/TTA methods average *logits or probabilities* and threshold **once**.
4. **Boundary-aware training/loss:** boundary, Lovász, clDice, or topology losses outperform plain BCE/Dice on thin structures.
5. **Calibration over mAP:** confidence, NMS IoU, and area thresholds are tuned on the *target metric* (PQ), not COCO mask AP.
6. **Post-processing that preserves boundaries:** DenseCRF, PointRend, or bilinear probability upsampling; avoid nearest-neighbor and avoid morphological dilation.

---

## 2. YOLOv8-seg vs. YOLO11-seg: Architecture & Attention

### 2.1 Structural differences

The official YAMLs show that YOLO11 is a **parameter- and compute-refresh** of the YOLOv8 backbone/neck, but the segmentation head itself is unchanged.

**YOLOv8l-seg backbone/neck/head (simplified) [^18]:**

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # P2/4
  - [-1, 3, C2f, [128, True]]        # P3/8
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]
  ...
  - [[15, 18, 21], 1, Segment, [nc, 32, 256]]  # (P3, P4, P5)
```

**YOLO11l-seg backbone/neck/head [^19]:**

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]   # P3/8
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]          # c3k=True here
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]              # NEW: attention after SPPF

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  ...
  - [[16, 19, 22], 1, Segment, [nc, 32, 256]]
```

### 2.2 C3k2 and C2PSA: what they actually do

* **C3k2** is a C2f-like CSP block with an optional inner `C3k` (kernel-size-configurable C3) or a plain `Bottleneck` [^20].  In the default `yolo11l-seg.yaml` the flag `c3k` is **False** for most stages, so the block is effectively a **Bottleneck-based C2f with a small expansion ratio `e=0.25`**.  This is the main reason YOLO11l has **27.6 M parameters** vs. YOLOv8l’s **46.0 M** [^21].  Fewer parameters can mean faster training/inference but also less representational capacity; for the filament task, where the model already struggles with fine boundaries, capacity matters.

* **C2PSA** is a CSP-style block whose active branch is a stack of `n` `PSABlock` modules [^22].  Each `PSABlock` is a Transformer-style block: multi-head self-attention (`Attention`) followed by a point-wise FFN [^23]:

```python
class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
```

In YOLO11, `C2PSA` is inserted **after SPPF at the deepest feature level (P5/32)**.  For `imgsz=2048`, P5 has spatial size **64×64**.  Attention at this scale can aggregate **global context** across the whole solar disk, which may help resolve long filaments and barb connectivity.  However, it cannot add spatial detail that is not present in the 64×64 feature map, and it introduces OOM/memory pressure that is already tight at 2048/batch=1.

**Verdict for filaments:**  C2PSA is the only YOLO11 component that could plausibly improve long-range/barbed-filament reasoning, but the bottleneck for thin 2–6 px structures is **the mask prototype resolution**, not backbone context.  YOLO11 is a reasonable A/B experiment, but not a guaranteed upgrade.

### 2.3 Prototype masks: the real resolution bottleneck

Both YOLOv8-seg and YOLO11-seg use the same `Segment` head and `Proto` module [^24]:

```python
class Proto(nn.Module):
    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)          # c2 = number of prototypes (default 32)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))
```

The `Segment` head calls `Proto` on the **P3/8 feature map** and predicts 32 mask prototypes [^25]:

```python
class Segment(Detect):
    def __init__(self, nc=80, nm=32, npr=256, ...):
        super().__init__(nc, ...)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)   # ch[0] = P3 channels
        ...
```

Because P3 is stride-8 and the Proto module has a **single 2× transposed convolution**, the raw prototype tensor is **stride-4** and its spatial size is **1/4 of `imgsz`**:

| `imgsz` | P3 size | Proto output size | Proto channels (prototypes) |
|---|---|---|---|
| 640 | 80×80 | 160×160 | 32 |
| 1024 | 128×128 | 256×256 | 32 |
| 2048 | 256×256 | **512×512** | 32 |

`mask_ratio=4` (default) controls how the *training GT masks* are downsampled onto the loss canvas — it does **not** raise the prototype output resolution [^26][^27].  Therefore, even at `imgsz=2048`, the model reasons about instance masks at **512×512** and then upsamples to 2048×2048.

For a solar filament only **2 px wide at 2048**, that filament is **0.5 px wide in the 512×512 prototype grid**.  It is geometrically undersampled; the boundary information must be hallucinated by the 4× bilinear upsample in `process_mask` or `process_mask_native`.

Ultralytics implements two post-processing paths [^28]:

```python
# Default (process_mask) -- faster, less accurate at native res
def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = crop_mask(masks, boxes=bboxes * ratios)        # crop at 1/4 res
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear')[0]
    return masks.gt_(0.0).byte()

# Retina masks (process_mask_native) -- more accurate
def process_mask_native(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]
    masks = crop_mask(masks, bboxes)                        # crop at native res
    return masks.gt_(0.0).byte()
```

Observations for the project:

* The current `predict_pytorch_gpu.py` calls `model.predict(..., retina_masks=False)` and then manually resizes the returned masks with `F.interpolate(..., mode='nearest')`.  This is **suboptimal**: `retina_masks=True` will use the native bilinear `process_mask_native` path and is the correct setting.
* The threshold is applied at `0.0` logits, equivalent to `0.5` probability (because `sigmoid(0.0) = 0.5`).  To avoid boundary dilation, the threshold should be applied **once** after any TTA/probability fusion, not per view.
* To get materially sharper masks, the prototype grid must be increased.  The only Ultralytics-supported ways are (a) train at a larger `imgsz` (already at 2048), or (b) **surgery on `Proto`’s transposed-convolution stride** (e.g. replace `2×2` with `4×4` or `8×8`) and retrain from scratch [^29].  Surgery is high-risk and not directly supported by the `yolo11l-seg.yaml` config; it would break COCO-pretrained weight loading.

### 2.4 Training hyperparameters for grayscale / scientific imagery

Solar Hα images are **grayscale, narrow dynamic range, and radially symmetric**.  The standard COCO-oriented Ultralytics defaults must be overridden:

```yaml
# data.yaml additions
channels: 1               # Ultralytics supports single-channel since PR #20310 [^30]

# Training overrides
model.train(
    data='data.yaml',
    imgsz=2048,
    batch=1,                # T4 16 GB with AMP
    epochs=60,
    patience=15,
    device=0,
    amp=True,
    single_cls=True,        # categories 1-4 all map to class 0
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,   # no color jitter on grayscale
    degrees=10.0,           # mild rotation
    fliplr=0.5, flipud=0.5, # solar rotational symmetry
    scale=0.5,              # project already uses 0.5
    translate=0.0,
    mosaic=0.0,             # BANNED: full-disk solar geometry
    close_mosaic=0,
    copy_paste=0.0,
    overlap_mask=True,      # faster; set False only if instance separation degrades
    mask_ratio=4,           # only affects GT rasterization, not output res
    retina_masks=False,     # training-time flag is irrelevant; use at inference
)
```

The most important non-defaults are:

* **`mosaic=0.0`**: the full-data experiment already proved that mosaic cuts the circular disk and long filaments, regressing 0.360 → 0.330–0.340 [^31].
* **`hsv_*=0.0`**: hue/saturation/value jitter is meaningless for grayscale Hα intensity.
* **`single_cls=True`**: all filament categories collapse to class 0.
* **`overlap_mask`**: default `True` merges all instance masks into a single index mask per image for speed.  For filaments that do not overlap this is fine; if the dataset contains overlapping annotator observations (it does, by duplicate physical files), `overlap_mask=False` may better preserve instance boundaries.

### 2.5 YOLOv8 vs. YOLO11 COCO benchmark

| Model | Params | mAPbox 50-95 | mAPmask 50-95 | T4 TRT10 ms (640) |
|---|---|---|---|---|
| YOLOv8l-seg | 46.0 M | 52.3 | 42.6 | 2.79 [^21] |
| YOLO11l-seg | 27.6 M | 53.4 | 42.9 | 7.8 [^21] |
| YOLO11x-seg | 62.1 M | 54.7 | 43.8 | 15.8 [^21] |

The mask-AP gap between YOLOv8l and YOLO11l is **0.3 points**; between YOLOv8l and YOLO11x it is **1.2 points**.  This is modest.  For a 2–6 px task, the practical gap may be even smaller because COCO masks are typically 20+ px wide and do not stress the prototype-resolution bottleneck.

**Recommendation:** do not treat YOLO11 as the primary path.  Run a controlled A/B with `yolo11l-seg` and `yolo11x-seg` only after `retina_masks`, TTA, and calibration are locked.

---

## 3. Metric-Aligned Ensembling & TTA under Panoptic Quality

### 3.1 Why the 0.360 → 0.350 ensemble regressed

The project’s forensics [^31] identified two failure modes:

1. **Segmentation Quality (SQ) drop:** the V8.1 cascade produced masks from a **1024×1024 model** that were then upscaled to 2048.  Their boundaries were already smoothed/dilated.  Averaging them with the crisp native-2048 Moonshot masks degraded the mean IoU of matched pairs.
2. **Recognition Quality (RQ) drop:** the ensemble added **146 uncorroborated false positives**, and each FP adds `0.5` to the PQ denominator:

```
PQ = Σ IoU_TP / (TP + 0.5·FP + 0.5·FN)
```

**Mathematical intuition for boundary dilation.**  Let the ground-truth mask be `G`, and suppose Moonshot predicts the exact mask `M1 = G`.  V8.1 predicts a dilated mask `M2 = G ∪ Δ`, where `|Δ| / |G| = 0.087` (the observed +8.7 % area dilation).  If we **average the binarized masks** and threshold at 0.5, the result is `M2` (because `M1` has value 1 on `G` and 0 on `Δ`; `M2` has value 1 on both; their mean is 1 on `G` and 0.5 on `Δ`; threshold 0.5 keeps `G ∪ Δ`).  The IoU of `M2` with `G` is:

```
IoU(M2, G) = |G| / (|G| + |Δ|) = 1 / 1.087 ≈ 0.919
```

This is still above the 0.5 matching threshold, but the *SQ* contribution dropped from `1.0` to `0.919`.  More importantly, if Moonshot has `IoU = 0.52` on a borderline filament and V8.1 has `IoU = 0.48`, the fused mask may fall **below** 0.5 and become a false positive, while the GT becomes an FN.  This is exactly the “borderline matches slipping below 0.5” failure reported in `03_MODEL_HISTORY.md`.

### 3.2 Grandmaster rules for mask ensembling

A safe ensemble pipeline must satisfy:

1. **Same-resolution inputs only.**  Never fuse 2048 native masks with 1024/512 masks.
2. **Cluster by overlap.**  Group proposals from different models/views whose mask IoU (or box IoU) exceeds a low threshold (e.g. `0.3`).
3. **Fuse in probability space.**  For each cluster, average the **per-pixel logit/probability maps** (not the binarized masks) and then threshold **once**.  If only binarized masks are available, use **intersection or best-confidence**, not union.
4. **One zero-overlap sanitizer.**  After fusion, sort instances by calibrated score and greedily carve pixels.  Assert zero shared pixels.
5. **Confidence recalibration.**  Replace raw box confidence with `score = conf_box × mean(mask_probability)` or a learned `P(IoU > 0.5)` from `InstanceCalibrator`.

The **Weighted Masks Fusion (WMF)** implementation from the Sartorius 8th-place solution [^9] is the reference pattern:

```python
def weighted_masks_fusion(all_pred_masks, all_pred_boxes, all_scores, inmodels,
                          num_models, conf_type='model_weighted', iou_thr=0.55):
    """
    all_pred_masks: (N, H, W) float32 probability or binary masks
    all_pred_boxes: (N, 4)
    all_scores:     (N,)
    inmodels:       (N,) model id for each proposal
    """
    # 1. Cluster by mask/box IoU (same as WBF)
    clusters = []
    for i, (mask, box, score) in enumerate(zip(all_pred_masks, all_pred_boxes, all_scores)):
        matched = False
        for cl in clusters:
            # representative of the cluster
            rep = cl[0]
            iou = mask_iou(mask, rep['mask'])
            if iou >= iou_thr:
                cl.append({'mask': mask, 'box': box, 'score': score, 'model': inmodels[i]})
                matched = True
                break
        if not matched:
            clusters.append([{'mask': mask, 'box': box, 'score': score, 'model': inmodels[i]}])

    # 2. Fuse each cluster
    final_masks, final_boxes, final_scores = [], [], []
    for cl in clusters:
        # Probability fusion: average soft masks, then threshold
        if all(m.dtype in (np.float32, torch.float32) for m in [c['mask'] for c in cl]):
            fused = np.mean([c['mask'] for c in cl], axis=0)
            fused_mask = (fused >= 0.5).astype(np.uint8)
        else:
            # Fallback to best-confidence if only binary masks available
            best = max(cl, key=lambda x: x['score'])
            fused_mask = best['mask']

        # Box: weighted average by score (WBF rule)
        scores = np.array([c['score'] for c in cl])
        boxes  = np.stack([c['box'] for c in cl])
        w = scores / scores.sum()
        fused_box = (boxes * w[:, None]).sum(0)

        # Confidence: score of best * mean mask quality
        best_score = max(scores)
        mask_quality = fused_mask.mean()
        fused_score = best_score * (1.0 + mask_quality) / 2.0

        final_masks.append(fused_mask)
        final_boxes.append(fused_box)
        final_scores.append(fused_score)

    return final_masks, final_boxes, final_scores
```

For **model ensembling** specifically, the GoN recipe [^6] is the simplest proven baseline:

* WBF for boxes.
* **Mean average** of the *raw predicted masks* inside the fused box.
* Post-filter `area < 80 px`.
* Re-score: `score = conf_bbox × mean(mask[mask > 0.5])`.

The project should adopt the same logic in `moonshot_2048/ensemble_instances.py`, replacing the current `best_conf` / `majority_vote` / `intersection` options with **probability averaging** when the same-resolution models provide soft masks.

### 3.3 TTA under Kirillov PQ

**The danger:** Ultralytics’ built-in `augment=True` does **not** support segmentation inference (issues [#15763](https://github.com/ultralytics/ultralytics/issues/15763) and [#16606](https://github.com/ultralytics/ultralytics/issues/16606)) [^32].  The current `predict_pytorch_gpu.py` performs manual two-flip TTA by running `model.predict()` three times (original, H-flip, V-flip), then concatenating the already-thresholded binary masks.  This is **not metric-safe**: it can produce duplicate or merged masks and does not average probabilities.

**The safe pattern for TTA under PQ:**

For each view `v ∈ {original, H-flip, V-flip}`:
1. Run the model and obtain the raw **mask logits** (not the binarized masks).  In the current Ultralytics code the logits are `(mask_coefficients @ prototype)` before `gt_(0.0)`.
2. Apply the inverse geometric transform to the logits so all views are in the original image frame.
3. Convert to probabilities with `sigmoid`.
4. **Average the probability maps** across views.
5. Threshold **once** at 0.5.
6. Run connected-components or use the original detection boxes to extract instances.
7. Greedy carve to enforce zero overlap.

The expected result is a **sharper or unchanged boundary**, never a dilated one, because a pixel is included only if the average probability exceeds 0.5.  If one view is uncertain at a boundary, the average pulls the logit below the threshold.

**Code sketch for probability-fusion TTA with a YOLO-seg model:**

```python
import torch
import torch.nn.functional as F

def tta_probability_fusion(model, img_path, imgsz=2048, device='0'):
    """Return a single 2048x2048 probability map from H- and V-flip TTA."""
    views = [
        (img_path, None),                  # original
        (hflip_image(img_path), 'h'),      # horizontal flip
        (vflip_image(img_path), 'v'),      # vertical flip
    ]

    prob_sum = torch.zeros((2048, 2048), device=device)
    n_views = 0

    for src, flip in views:
        # Run model and extract the raw prototype/coefficient outputs.
        # This requires a forward hook or a custom predictor.
        # Simplified: assume we have logits tensor of shape (H, W)
        logits = yolo_forward_logit_map(model, src, imgsz, device)

        if flip == 'h':
            logits = torch.flip(logits, dims=[2])   # undo H-flip
        elif flip == 'v':
            logits = torch.flip(logits, dims=[1])   # undo V-flip

        prob = torch.sigmoid(logits)
        prob_sum += F.interpolate(prob[None, None], (2048, 2048), mode='bilinear')[0, 0]
        n_views += 1

    mean_prob = prob_sum / n_views
    consensus_mask = mean_prob > 0.5
    return mean_prob, consensus_mask
```

In practice, extracting the raw logit map from the Ultralytics `SegmentationPredictor` requires overriding `process_mask` to return probabilities instead of `byte()` masks, or using the raw `preds` tuple `(feats, mask_coefficients, proto)` and calling `process_mask` manually with `upsample=True` and no threshold.

If the architecture only allows access to **binarized masks** (as in the current pipeline), the safe TTA fallback is:

* Generate masks for each flip.
* Invert the flip.
* For each spatially aligned mask cluster, take the **intersection** (most conservative) or the **best-confidence mask**, never the union.
* Then run the greedy carve.

This will not improve boundary sharpness, but it will not dilate.

---

## 4. High-Precision Consensus Calibration for Multi-Annotator PQ

### 4.1 The multi-annotator ceiling

The MAGFiLO training set has **707 physical JPEGs** and **1,154 COCO `image_id`s**, i.e. 447 physical disks have 2–3 independent annotators.  Valentin Haase’s Kaggle post measured **mean Dice 0.687 / median 0.727** between annotators on the same images, with **23 % of filaments appearing in only one annotator** [^33].  The project’s own multi-annotator evaluator estimates the human-vs-human PQ ceiling at **0.36–0.37** [^31].

This means:

* A single-annotator model cannot realistically score far above 0.36 if the test ground truth is an *average* of annotators or a single hidden annotator.
* If the hidden test ground truth is **pooled / consensus-aggregated** (a possibility discussed in `master/DEVIN_DEEP_RESEARCH_REPORT.md`), the model should be encouraged to predict the **inter-annotator consensus**, not the idiosyncrasies of one annotator.

### 4.2 The PQ formula and its decomposition

Kirillov et al. define Panoptic Quality as [^34]:

```
PQ = Σ_{(p,g) ∈ TP} IoU(p, g) / (|TP| + 0.5·|FP| + 0.5·|FN|)
```

Equivalently:

```
PQ = SQ × RQ
SQ = (1 / |TP|) · Σ_{(p,g) ∈ TP} IoU(p, g)
RQ = |TP| / (|TP| + 0.5·|FP| + 0.5·|FN|)
```

**Matching rule:** a predicted segment `p` and a ground-truth segment `g` form a true positive iff `IoU(p,g) > 0.5`.  Kirillov proves that at this threshold and under the non-overlap constraint the matching is **unique** — each `p` matches at most one `g` and vice versa [^34].

This makes the 0.5 IoU boundary **razor-sharp**: a mask with `IoU = 0.49` is treated identically to one with `IoU = 0.01` (both are unmatched and become FP/FN).  It also makes **area dilation the enemy**: an 8.7 % area increase can push a 0.52 IoU down to ~0.48 and convert a TP into an FP.

### 4.3 Marginal optimization of confidence, NMS, and area thresholds

Consider a candidate instance proposal with feature vector `x` (confidence `c`, area `a`, radial position, etc.).  Let `P_match(x)` be the probability that this proposal matches a ground-truth filament with `IoU > 0.5`, and let `E[IoU | match]` be the expected IoU of those matches.

When the proposal is added to the prediction set:

* With probability `P_match(x)`, it becomes a **TP**: numerator increases by `E[IoU | match]`, denominator increases by `+1` (but removes the `0.5` FN that the matched GT would otherwise incur), net denominator increase **`+0.5`**.
* With probability `1 - P_match(x)`, it becomes an **FP**: numerator unchanged, denominator increases by **`+0.5`**.

Therefore the **expected change** in the numerator and denominator are:

```
E[ΔN] = P_match(x) · E[IoU | match]
E[ΔD] = 0.5 + 0.5 · P_match(x) = 0.5 · (1 + P_match(x))
```

At the current operating point `(N, D)` with `PQ = N / D`, the first-order expected change in PQ is:

```
E[ΔPQ] ≈ E[ΔN] / D - (N / D²) · E[ΔD]
       = (E[ΔN] - PQ · E[ΔD]) / D
```

So a proposal should be accepted if:

```
P_match(x) · E[IoU | match]  >  PQ · 0.5 · (1 + P_match(x))
```

For the project’s current `PQ ≈ 0.36`, this becomes:

```
P_match(x) · E[IoU | match]  >  0.18 · (1 + P_match(x))
```

If a candidate has `P_match = 0.6` and `E[IoU | match] = 0.65`, the left-hand side is `0.39`, the right-hand side is `0.288`; the candidate is clearly worth keeping.  If `P_match = 0.25` and `E[IoU | match] = 0.55`, the left side is `0.1375` and the right side is `0.225`; the candidate should be rejected.

This is a **calibrated acceptance rule**, not a fixed confidence threshold.

**Implementation:** the `moonshot_2048/match_and_calibrate.py` `InstanceCalibrator` already extracts `conf, log_area, perimeter, circularity, elongation, solidity, n_components, radial_dist` and fits a `LogisticRegression` or `HistGradientBoostingClassifier` to estimate `P(match)`.  The next step is to **deploy this calibrator at inference time**: replace the raw confidence sort with a **calibrated match-probability sort** and use the marginal rule above to accept/reject candidates.

### 4.4 NMS IoU and `min_area`

* **`nms_iou`**: PQ enforces non-overlapping predictions.  A high NMS IoU (e.g. 0.7) pre-suppresses many candidates, which is dangerous for thin curvilinear objects where two overlapping boxes may actually be two adjacent filaments.  A low `nms_iou` (e.g. `0.0`, the current project default) keeps candidates and lets the **greedy zero-overlap carver** resolve conflicts.  The evidence from Sartorius and GoN favors **low NMS/late NMS** followed by mask-level resolution.

* **`min_area`**: PQ treats every instance equally regardless of area [^35].  A spurious 50-px blob costs the same as a 5000-px blob in the denominator.  Therefore `min_area` should be set by validation, not by a fixed rule.  The project’s current sweep grid `min_area ∈ {50, 100, 200, 400}` is correct; the optimal point is likely in the **50–150 px** range, because many real filaments are small but small FPs are very costly.

* **Confidence threshold `conf`**: should be replaced by the calibrated `P_match` whenever possible.  If only a raw confidence sweep is available, the PQ-optimal operating point will typically be the **precision-recall sweet spot** where the marginal gain from an extra TP balances the `0.5·FP` penalty.  The project’s sweep already explored `conf ∈ {0.15, 0.20, 0.25, 0.30, 0.35}` and found a plateau near **0.30** [^31].  A finer sweep with the calibrator and a multi-annotator val split is the next step.

### 4.5 Multi-annotator consensus targets (optional but high-ceiling)

If the hidden test set is pooled or consensus-aggregated, training on a single annotator is suboptimal.  A better target is a **soft consensus mask** per physical disk:

```
M_consensus(y, x) = (1 / K) · Σ_{k=1}^{K} M_annotator_k(y, x)
```

where `K` is the number of annotators for that physical disk.  Pixels labeled by all annotators have value `1.0`; pixels labeled by one annotator have value `0.5` or `0.33`.  This can be used as a soft target for `BCEWithLogitsLoss`:

```python
# In a custom YOLO-seg loss or in a post-train refiner
loss = F.binary_cross_entropy_with_logits(pred_logit, soft_target, reduction='mean')
```

This encourages the model to learn the **inter-annotator agreement distribution** rather than the hard boundary of a single rater, which should improve both recall and boundary alignment against a pooled test set.  Note: YOLO’s `BCEDiceLoss` accepts float targets, so this is implementable by overriding the mask target construction in `v8SegmentationLoss`.

---

## 5. Recommended Next-Run Checklist (Prioritized)

### P0 — immediate, highest expected PQ return

1. **Fix inference to use `retina_masks=True` and bilinear upsampling.**
   * Verify in `predict_pytorch_gpu.py` / `predict_native.py` that `model.predict(..., retina_masks=True, imgsz=2048)` is used, and remove any manual `INTER_NEAREST` resize.  Confirm output masks are 2048×2048 and thresholded once.

2. **Calibrate `conf`, `min_area`, and NMS on the local multi-annotator PQ evaluator.**
   * Run the existing `run_holdout_validation_sweep` with a finer grid:
     * `conf ∈ [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40]`
     * `min_area ∈ [25, 50, 75, 100, 150, 200]`
     * `nms_iou ∈ [0.00, 0.10, 0.20, 0.30, 0.40]`
   * Select the operating point that maximizes `pq_mean` on the multi-annotator val split.

3. **Deploy the `InstanceCalibrator` at inference.**
   * Train the calibrator on OOF candidates from the 0.360 `best.pt` (or the current Fold-0 val).
   * Replace the greedy-carve sort key `conf` with the calibrated `P_match`.
   * Accept candidates using the marginal rule `P_match · E[IoU] > 0.18 · (1 + P_match)`.

### P1 — architecture and resolution experiments

4. **A/B `yolo11l-seg` and `yolo11x-seg` at native 2048 with `mosaic=0`.**
   * Use the same data pipeline as the 0.360 YOLOv8l run.
   * Train for **no more than 60 epochs** from COCO-pretrained weights (do not fine-tune from YOLOv8 weights; they are incompatible).
   * Compare PQ, SQ, RQ, TP/FP/FN on the same multi-annotator val split.
   * **Decision rule:** switch to YOLO11 only if it clearly beats YOLOv8l on local PQ; otherwise stay on YOLOv8l.

5. **Investigate higher mask-prototype resolution.**
   * Option A: patch the `Proto` module to use a **4× or 8× transposed convolution** instead of 2×, and retrain a small model from scratch on synthetic/MAGFiLO to see if OOM is manageable [^29].
   * Option B: add a **second-stage boundary refiner** (e.g. a lightweight U-Net or PointRend-style head) that takes the YOLO probability/box crops and refines them to 2048.  This is architecturally the V8.1 cascade lesson, but the cascade must be at **2048**, not 1024.

### P2 — ensembling, TTA, and loss

6. **Implement probability-fusion instance ensembling.**
   * Modify `moonshot_2048/ensemble_instances.py` to:
     * Cluster overlapping proposals from multiple models or TTA views by mask IoU.
     * Fuse aligned masks by **averaging probabilities** (or `np.mean` of soft masks) and thresholding once.
     * Use WBF for boxes and `score = conf_box × mean(mask_prob)`.
   * Only ensemble **native 2048 models** (never mix with 1024/1536/1792).

7. **Replace manual TTA with logit/probability TTA.**
   * If raw logits are accessible, implement the probability-fusion TTA in Section 3.3.
   * If only binary masks are accessible, use **intersection-consensus TTA** and evaluate whether it improves PQ.

8. **Test boundary/topology losses in a standalone crop refiner or auxiliary head.**
   * Train a small `smp.UnetPlusPlus(effnet-b4)` with `BCEDiceLoss + LovaszLoss` or `clDiceLoss` on 384×384 or 512×512 **crops around YOLO boxes**.
   * Ensure the crop is large enough not to truncate long filaments (at least 2× the filament bounding box, clamped to image border).

### P3 — data and test-time strategy

9. **Generate soft multi-annotator consensus targets and retrain (or post-train).**
   * Create `M_consensus` for the 447 multi-annotator disks.
   * Experiment with soft-target BCE in a custom YOLO-seg loss or in a refiner only.

10. **Avoid the 100 % full-data fine-tune trap.**
    * The full-data `mosaic=1.0` fine-tune regressed.  A clean full-data run should use **no mosaic**, no extra epochs beyond convergence, and start from COCO-pretrained weights, not from the 60-epoch checkpoint, to avoid prototype drift.

---

## 6. Conclusion

The path from **0.360 → 0.50+** is not a single silver-bullet architecture.  The highest-impact, lowest-risk moves are:

1. Use the **native 2048 resolution** with `retina_masks=True` and proper bilinear mask resampling.
2. Stop **dilating masks** — no mosaic, no threshold-then-average ensembling, no nearest-neighbor upsampling.
3. **Calibrate every candidate** with the PQ metric and the `InstanceCalibrator` rather than a fixed confidence threshold.
4. If and only if those are exhausted, experiment with **YOLO11l/x** and/or a **second-stage boundary refiner / point-rendering head**.

The human inter-annotator ceiling is ~0.36–0.37, so crossing 0.50 requires the model to be more **metric-aware, boundary-precise, and calibrated** than the average human annotator.  That is achievable with the interventions above, but only if every post-processing step is treated as a first-class optimization lever under the Kirillov PQ formula.

---

## References

[^1]: Rist, Takuoko, Tascj. *1st place solution, Sartorius Cell Instance Segmentation*. Kaggle write-up, 2022. https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/writeups/rist-takuoko-tascj-1st-place-solution

[^2]: Tascj. *kaggle-sartorius-cell-instance-segmentation-solution* GitHub, 2022. https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution

[^3]: nvnn / sheep. *2nd place solution, Sartorius Cell Instance Segmentation*. Kaggle discussion, 2021. http://kaggle.com/competitions/sartorius-cell-instance-segmentation/discussion/297988

[^4]: RobinSmits. *KaggleHuBMAP* repository, 2021. https://github.com/RobinSmits/KaggleHuBMAP

[^5]: Boorjian et al. *Robust and generalizable segmentation of human functional tissue units*. bioRxiv, 2021. https://doi.org/10.1101/2021.11.09.467810

[^6]: SDSRV.AI / GoN. *4th place solution, HuBMAP — Hacking the Human Vasculature*. Kaggle write-up, 2023. https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/sdsrv-ai-gon-4th-place-solution-sdsrv-ai-gon

[^7]: Igor Krashenyi. *4th place solution, SenNet + HOA Blood Vessel Segmentation*. Kaggle write-up, 2024. https://www.kaggle.com/competitions/blood-vessel-segmentation/writeups/igor-krashenyi-4th-place-solution-boundary-dou-los

[^8]: cns-iu. *hra-sennet-hoa-kaggle-2024* repository, 2024. https://github.com/cns-iu/hra-sennet-hoa-kaggle-2024

[^9]: Odede. *8th place solution, Sartorius Cell Instance Segmentation*. Kaggle write-up, 2022. https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/writeups/odede-8th-place-solution

[^10]: Shit et al. *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*. CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf

[^11]: Kervadec et al. *Boundary loss for highly unbalanced segmentation*. MIDL 2019. http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf

[^12]: Zhang et al. *Centerline Boundary Dice Loss for Vascular Segmentation*. MICCAI 2024. https://papers.miccai.org/miccai-2024/paper/0458_paper.pdf

[^13]: Berman, Rannen Triki, Blaschko. *The Lovász-Softmax Loss*. CVPR 2018. https://arxiv.org/abs/1705.08790

[^14]: Kirillov et al. *PointRend: Image Segmentation As Rendering*. CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf

[^15]: Krähenbühl & Koltun. *Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials*. NeurIPS 2011.

[^16]: Ultralytics. `ultralytics/utils/loss.py` — `BCEDiceLoss` and `v8SegmentationLoss`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py

[^17]: Solomon et al. *EdgeAttNet: Towards Barb-Aware Filament Segmentation*. arXiv:2509.02964, 2025. https://arxiv.org/abs/2509.02964

[^18]: Ultralytics. `yolov8-seg.yaml`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-seg.yaml

[^19]: Ultralytics. `yolo11-seg.yaml`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml

[^20]: Ultralytics. `nn/modules/block.py` — `C3k2`, `PSABlock`, `C2PSA`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py

[^21]: Ultralytics. YOLO11 model documentation and COCO benchmarks. https://docs.ultralytics.com/models/yolo11

[^22]: Ultralytics. *YOLO Architecture Explained*. https://docs.ultralytics.com/guides/yolo-architecture

[^23]: 2048 AI / CSDN. YOLO11 C2PSA and PSABlock code walkthrough. https://2048ai.net/6846909c01ee5225109b045f.html

[^24]: Ultralytics. `nn/modules/head.py` — `Segment` and `Proto`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py

[^25]: Ultralytics. `nn/modules/block.py` — `Proto`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py

[^26]: Ultralytics GitHub issue #20200, *mask_ratio not working*. https://github.com/ultralytics/ultralytics/issues/20200

[^27]: Ultralytics GitHub issue #23820, *Yolo 26 ONNX and TFLITE segmentation mask size*. https://github.com/ultralytics/ultralytics/issues/23820

[^28]: Ultralytics. `utils/ops.py` — `process_mask` and `process_mask_native`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

[^29]: Ultralytics issue #20200 maintainer comment: changing Proto’s `ConvTranspose2d` stride can increase mask resolution. https://github.com/ultralytics/ultralytics/issues/20200

[^30]: Ultralytics GitHub PR #20310, *Add Grayscale image training support*. https://github.com/ultralytics/ultralytics/pull/20310

[^31]: `master/03_MODEL_HISTORY.md`, `master/04_CURRENT_DIRECTIVE.md`, `master/DEVIN_DEEP_RESEARCH_REPORT.md` in this repository.

[^32]: Ultralytics GitHub issues #15763 and #16606: `augment=True` is not supported for segmentation. https://github.com/ultralytics/ultralytics/issues/15763, https://github.com/ultralytics/ultralytics/issues/16606

[^33]: Kaggle discussion #729809, *Some measurements on the public metric, and a possible fix* (Valentin Haase). https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809

[^34]: Kirillov et al. *Panoptic Segmentation*. CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf

[^35]: TensorFlow Models, deeplab PQ evaluation README: “PQ treats all regions of the same ‘stuff’ class as one instance, and the size of instances is not considered … PQ is sensitive to false positives with small regions.” https://github.com/tensorflow/models/blob/master/research/deeplab/evaluation/README.md

[^36]: G. Zhu et al., “Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk Hα Images,” *ApJ* 980, 176 (2025). https://iopscience.iop.org/article/10.3847/1538-4357/adadff

[^37]: Compound U-Net Zenodo artifact (MHAS / HAS solar filament segmentation). https://doi.org/10.5281/zenodo.17230604
