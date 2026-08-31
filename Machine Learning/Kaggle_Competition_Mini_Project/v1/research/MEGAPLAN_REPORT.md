# Solar Filament Segmentation 2026 — Megaplan Research Report

> Compiled from 10 targeted web/literature searches, direct inspection of EdgeAttNet/Flat-U-Net/Compound-U-Net repositories, and the arXiv EdgeAttNet paper.
> **Goal:** beat the friend’s U-Net++ EfficientNet-B4 5-fold result and reach the public top-50 using a single Colab notebook that fits in ≤2 h/fold on a free T4.

---

## 1. Executive summary

The single strongest published result on the MAGFiLO task is **EdgeAttNet** (mIoU_pairwise 0.6451, mIoU_multiscale **0.7032**), but its official GitHub repository is missing the actual `UNetEdgeTransformer` model file and the evaluation `utils.py` are incomplete. The paper, however, gives enough detail to re-implement the model from scratch. The next best public candidates are **Flat U-Net** (ultralight, DSC 0.82, recall 0.74) and **Compound U-Net** (IoU 0.677–0.714), both available as Zenodo packages with code.

For a 2-hour T4 fold, a full EdgeAttNet re-implementation is high-risk and may not finish. The safer, highest-ROI path is to keep the existing `segmentation-models-pytorch` backbone, add **EdgeAttNet-style preprocessing** (disk masking + radial flattening + CLAHE), switch the local metric to the exact paper formulation, add **topology/boundary losses** (clDice / boundary / skeleton), and use **stronger augmentation** (CopyAndPaste, GridMask, ISONoise). A separate `EdgeAttNet` preset can be offered as a high-risk/high-reward option once it is proved locally.

---

## 2. What the leaderboard actually rewards

The IEEE BigData Cup 2026/Kaggle competition will likely use a combination of:

- **Pairwise IoU** — greedy matching between ground-truth and predicted instances, averaged over overlapping pairs.
- **Multiscale IoU** — box-counting the masks at a set of resolutions and averaging the intersection / gt-area ratio across scales.
- **AP / AR@IoU 0.5..0.95** — COCO-style average precision and average recall.
- **Hit / miss rate** — object-level recall is heavily penalized.

Implication: the model must (a) not miss thin barbs, (b) not over-fragment filaments, (c) keep predictions close to the true boundary. Standard pixel IoU alone is not enough; the validation loop must use the same multi-scale, pairwise logic as the leaderboard.

---

## 3. Public solutions and SOTA — ranked by expected lift

| # | Solution | Reported score | Pros | Cons | Verdict |
|---|---|---|---|---|---|
| 1 | **EdgeAttNet** (paper) | mIoU_multiscale **0.7032**, mIoU_pairwise **0.6451**, 22.6M params | Highest published; edge-guided attention directly targets barbs; no positional encodings | **GitHub model file is missing**; needs from-scratch implementation; trained at 512; no ImageNet pretraining; 2h/fold on T4 uncertain | High-risk / highest-reward; implement as optional preset |
| 2 | **Flat U-Net SCA1/CSA4** (Zenodo) | Precision 0.93, DSC 0.82, Recall 0.74; ~1M params | Very fast, very small, proven on solar data | Lower raw mIoU than EdgeAttNet; custom code required | Use as fast baseline / sanity-check |
| 3 | **Compound U-Net** (Zenodo) | IoU 0.677–0.714, F1 0.80–0.83 | Multi-scale feature extraction; public code | Requires custom model; smaller gain than EdgeAttNet | Candidate for model zoo if time allows |
| 4 | **U-Net++ EfficientNet-B4 5-fold** (friend reference) | Unknown, likely mIoU_multiscale ~0.62–0.66 | Strong baseline, easy to reproduce | Already the target to beat; need better to reach top-50 | Must exceed this |
| 5 | **CSCA U-Net** | Strong on medical benchmarks, not on solar | Channel+space compound attention, deep supervision | No solar-specific validation | Low priority |
| 6 | **Attention U2-Net** | IoU 0.7139 on solar test set (paper) | Strong if full preprocessing pipeline copied | Heavy preprocessing (K-means, limb-darkening, hole fill, fit disk) | Medium priority |

---

## 4. Encoder/decoder zoo (compatible with `smp`)

| # | Encoder | T4 @1024 BS2 VRAM | Expected lift vs B4 | Notes |
|---|---|---|---|---|
| 1 | `timm-efficientnet-b4` | ~9.5 GB | baseline | Current default; good speed/accuracy |
| 2 | `timm-efficientnet-b5` | ~11 GB | +0.5–1.0% | Fits on T4 with BS1/accum=8 |
| 3 | `timm-efficientnet-b6` | OOM at 1024 | +0.5%? | Too big; use 768 or smaller crop |
| 4 | `timm-convnext-tiny` | ~8.5 GB | +0.5–1.5% | Strong stem, may help barbs |
| 5 | `timm-convnext-base` | ~10.5 GB | +1–2% | Better but slower |
| 6 | `timm-resnet50` | ~8.5 GB | 0–+0.5% | Proven; less capacity than B4 |
| 7 | `timm-resnet101` | ~10 GB | +0.5% | Slower, marginal gain |
| 8 | `timm-densenet161` | ~9.5 GB | +0.5% | Good feature reuse |
| 9 | `timm-hrnet_w32` | ~10 GB | +0.5–1% | High-resolution feature maps help edges |
| 10 | `timm-mobilenetv3_large_100` | ~6 GB | -0.5% | Too small |

Decoder: `unetplusplus` is best for accuracy; `unet` is fastest; `deeplabv3plus` can help multi-scale context; `fpn` is lightweight. For ≤2h/fold, `unet` + `efficientnet-b4`/`convnext-tiny` is the sweet spot; `unetplusplus` can be the `max` preset.

---

## 5. Loss functions — 10 candidates

| # | Loss | Best for | Expected lift | Notes |
|---|---|---|---|---|
| 1 | **Focal Tversky** (α=0.3, β=0.7, γ=1.33) | Class imbalance, thin filaments | baseline | Already in code; tune weights |
| 2 | **clDice / soft-clDice** | Tubular topology preservation | +1–2% | Filaments are network-like; prevents breaks |
| 3 | **Boundary loss (Sobel L1)** | Edge alignment | +0.5–1% | From FBS loss; lightweight |
| 4 | **Skeleton loss (Laplacian)** | Connectivity | +0.5% | Differentiable proxy for thin structures |
| 5 | **Focal Tversky + Boundary + clDice** combo | All three | +1.5–2.5% | Main implementation target |
| 6 | **Lovász hinge/sigmoid** | Direct mIoU optimization | +0.5–1% | Expensive, unstable early |
| 7 | **Asymmetric Focal Tversky** | FN/FP control | +0.5% | Similar to current, just tuned |
| 8 | **Dice + BCE + Focal** | Stable baseline | 0% | Already covered by current combo |
| 9 | **Hausdorff distance transform** | Boundary accuracy | +0.3% | Slow on large masks |
| 10 | **OhemBCE** | Hard negative mining | +0.3% | Good for background confusion |

Recommended: `0.5 * (BCE+pos_weight) + 0.25 * Dice + 0.15 * FocalTversky + 0.1 * Boundary`. Add optional clDice with a small weight (0.05–0.1) if memory allows.

---

## 6. Augmentation and synthetic data — 10 candidates

| # | Technique | Expected lift | Notes |
|---|---|---|---|
| 1 | **CopyAndPaste** from other filaments | +0.5–1.5% | Albumentations 2.2+ has `CopyAndPaste`; paste instance masks onto background |
| 2 | **GridMask / GridDropout** | +0.3–0.5% | Albumentations has `GridDropout`; not true GridMask, but blocks regularization |
| 3 | **MixUp / CutMix on masks** | +0.3% | Low risk for semantic masks |
| 4 | **Mosaic (2×2)** | +0.5% | Build context, but thin filaments may distort |
| 5 | **Heavy elastic + grid distortion** | +0.3% | Already partially in pipeline; tune alpha/sigma |
| 6 | **ISONoise + RandomGamma + CLAHE** | +0.3% | Already in pipeline; keep uint8 until normalize |
| 7 | **Solar rotation / polar warp** | +0.2–0.5% | Physics-aware; simulates projection changes |
| 8 | **Random sunspot / fibril injection** | +0.2% | Add realistic distractors |
| 9 | **Multi-annotator sampling as soft targets** | +0.3% | Use the 23% multiply-annotated images as soft labels |
| 10 | **Artificial negative images** | +0.2% | Some test images contain no filament; train on empty images |

CopyAndPaste is the biggest low-risk win. It requires instance masks (which we can extract from COCO polygons) and a donor image bank.

---

## 7. TTA and inference — 10 candidates

| # | TTA | Views | Time cost | Expected lift |
|---|---|---|---|---|
| 1 | **D4 (8 dihedral)** | 8 | 8× | baseline; already implemented |
| 2 | **Horizontal + vertical flips only** | 4 | 4× | Fallback if 8 is too slow |
| 3 | **Multi-scale 0.8, 1.0, 1.25** | 3 | 3× | +1–2% mIoU; scales help barbs |
| 4 | **Multi-scale 0.75, 1.0, 1.25, 1.5** | 4 | 4× | More scales, slower |
| 5 | **D4 + multi-scale 1.0, 1.25** | 16 | 16× | Too slow for 2048×180 images on T4 |
| 6 | **TTA scale averaging with weight 1/scales** | — | — | Small gain |
| 7 | **Flip-only at full 2048** | 2 | 2× | No resize distortion; but model never saw 2048 |
| 8 | **Snapshot ensemble from periodic checkpoints** | N models | N× | Save every 5 epochs, average at test |
| 9 | **EMA model for inference** | 1 | 0% | Already in code; keep |
| 10 | **Test-time dropout / MC-dropout** | 5–10 | 5–10× | Marginal; not worth the time |

Recommended: D4 (8) + scales `[0.85, 1.0, 1.15]` (24 views) is too slow. Use **D4 (8) + scales `[1.0, 1.15]` (16 views)** only if inference budget allows; otherwise **D4 (8)** alone or **D4 (4)** + `[0.9, 1.0, 1.1]` (12 views). Default: D4 (8) at 1024.

---

## 8. Post-processing — 10 candidates

| # | Method | Expected lift | Notes |
|---|---|---|---|
| 1 | **Threshold + min-area grid search** | +1–2% | Already in code; critical; must use F1-penalised mIoU |
| 2 | **Dense CRF (pydensecrf)** | +0.5–1% | Good for boundary smoothing; requires RGB image |
| 3 | **Bilateral filter on probability map** | +0.3% | Lightweight, no install |
| 4 | **Hysteresis thresholding** | +0.3% | Use high+low thresholds to keep thin connections |
| 5 | **Watershed splitting with distance-transform tuning** | +0.3% | Already in code; tune `min_distance` |
| 6 | **Hole filling (small holes)** | +0.2% | Already in code; tune size |
| 7 | **Mask NMS / soft-NMS** | +0.2% | Remove overlapping duplicate instances |
| 8 | **Solar disk masking improvement (Hough circle)** | +0.5% | More accurate than threshold-based disk mask |
| 9 | **Edge-aware anisotropic diffusion** | +0.2% | Smooth while preserving edges |
| 10 | **Confidence-based instance filtering** | +0.2% | Remove low-probability spurious blobs |

CRF and better disk masking are the biggest low-risk wins.

---

## 9. Extra data and pre-training — 10 candidates

| # | Source | Size | Feasibility | Expected lift |
|---|---|---|---|---|
| 1 | **Full MAGFiLO v1.0 (Kaggle dataset)** | 704 images + 8,212 filaments | High (public) | +0.5–1% if used as extra train; same distribution |
| 2 | **EdgeAttNet processed 1,593 GONG observations** | Large | Medium (need FITS processing) | +1–2% if labels can be matched |
| 3 | **HAS / MHAS datasets (Compound U-Net)** | 80/20 train/test | Medium | +0.5% if domain transfer works |
| 4 | **Test-set pseudo-labelling** | 180 images | High | +0.5–1% if confident threshold used |
| 5 | **Unlabelled GONG archive (SunPy)** | Huge | Low (no labels) | Pre-train only; time consuming |
| 6 | **Synthetic filament generation** | Unlimited | Medium | Add artificial thin structures |
| 7 | **ImageNet pretraining of encoder** | — | Already in code | Strong baseline; keep |
| 8 | **MAGFiLO pretrain → competition finetune** | 704 images | High | +0.5–1% over ImageNet alone |
| 9 | **Multi-year temporal pretraining** | — | Low | Not enough time |
| 10 | **Solar prominence / sunspot datasets** | Unknown | Low | Domain shift may hurt |

Full MAGFiLO is the safest extra-data source because the competition images come from it. Pre-training on it then fine-tuning on the 707 competition train is high ROI.

---

## 10. Metrics and validation — exact formulation

From the EdgeAttNet paper and repository:

- **Pairwise IoU** for two instances is `|gt ∩ pred| / |gt ∪ pred|` if they overlap, else 0.
- **mIoU_pairwise** averages over all matching pairs (greedy matching is sensible).
- **Multiscale IoU** downsamples masks at multiple `cell_size`s (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512) using **box-counting max-pool**, then computes `intersection / gt_area` at each scale and averages (trapezoidal integration is optional).
- The paper does **not** use edge-only mIoU; it uses the full mask at each scale. Edge-only is a useful proxy but may not correlate perfectly.
- For model selection, an F1-style penalty `2*n_matched / (n_gt + n_pred)` should be multiplied to avoid fragmentation/over-segmentation gaming.

Recommended implementation: keep greedy one-to-one matching, but compute the multiscale IoU on the **full binary masks** (not just edges) at each cell size.

---

## 11. Speed and memory — 10 candidates

| # | Technique | Expected speed/VRAM gain | Notes |
|---|---|---|---|
| 1 | `uint8` memmap cache of resized images/masks | 5–10× | Already implemented |
| 2 | `torch.channels_last` + AMP | 1.5–2× | Already implemented |
| 3 | `cudnn.benchmark = True` | 1.2× | Already implemented |
| 4 | Gradient accumulation | − | Already implemented |
| 5 | `torch.compile` on model | 1.1–1.5× | May not work with `smp` / dynamic shapes; test first |
| 6 | Smaller crop for train / full-res only for inference | 1.5–2× | Risk: scale mismatch; keep one resolution |
| 7 | Freeze encoder first N epochs | 1.2× early | May hurt convergence |
| 8 | `decoder_use_batchnorm='inplace'` | −15% VRAM | `smp` 0.5 uses `decoder_use_norm='batchnorm'`; `inplace` may not exist |
| 9 | Smaller decoder channels (`decoder_channels=(128,64,32,16,8)`) | −20% params | `smp` supports `decoder_channels` |
| 10 | Offload OOF prob maps to CPU | −0.5 GB VRAM | Already in code; not a bottleneck |

The current baseline already has the big speed wins. The remaining headroom is modest.

---

## 12. Final recommendation: what to implement now

Given the constraints (3 T4s, ≤2 h/fold, 3–7 days, no friend code), the highest-ROI, lowest-risk upgrades are:

### Must-implement (Phase A)
1. **Exact multiscale IoU metric** — switch from edge-only to full-mask box-counting; keep F1 penalty.
2. **EdgeAttNet-style preprocessing** — disk mask with Hough circle, radial flattening (limb-darkening correction), Gaussian blur, CLAHE, off-disk zeroing.
3. **Boundary + clDice loss** — add Sobel-boundary and soft-clDice to the existing Focal Tversky combo.
4. **CopyAndPaste augmentation** — build instance mask bank and paste random filaments onto training images.

### High-priority (Phase B)
5. **Multi-scale TTA** — D4 + `[0.9, 1.0, 1.1]` or `[1.0, 1.15]` depending on speed.
6. **Dense CRF post-processing** — add optional CRF on the probability map.
7. **Hough-circle solar disk mask** — more accurate than threshold-based disk mask.
8. **Full MAGFiLO pre-training** — if the Kaggle dataset can be downloaded, pre-train one epoch on it, then fine-tune on competition.

### Optional / high-risk (Phase C)
9. **EdgeAttNet re-implementation** — implement a simplified `UNetEdgeTransformer` as an optional `arch='edgeattnet'` preset. Only enable if a local smoke test proves it fits and trains in <2h.
10. **Flat U-Net / Compound U-Net integration** — as additional `arch` options for model diversity in the final ensemble.

### What to drop
- Transformers (SegFormer, Swin) at 1024 — too slow/VRAM on T4.
- Very heavy multi-scale TTA (24+ views) — inference time explodes.
- Pseudo-labelling as the main strategy — too dependent on a good teacher; use only if time remains.

---

## 13. Sources

- EdgeAttNet paper: https://ar5iv.org/pdf/2509.02964
- EdgeAttNet code (incomplete): https://github.com/dasjar/EdgeAttNet
- Flat U-Net (Zenodo): https://zenodo.org/records/14610155
- Flat U-Net paper: https://doi.org/10.3847/1538-4357/adadff
- Compound U-Net (Zenodo): https://doi.org/10.5281/zenodo.17230604
- Felipesp A100 lab (HuggingFace, gated): https://huggingface.co/datasets/felipesp1983/solar-filament-2026-a100-lab
- MAGFiLO v1.0 dataset: https://www.kaggle.com/datasets/esairlab/magfilo-v1-0-segmentation-of-solar-filaments
- clDice: https://github.com/jocpae/clDice
- pydensecrf: https://github.com/lucasb-eyer/pydensecrf
- Albumentations D4 TTA: https://albumentations.ai/docs/4-advanced-guides/test-time-augmentation/

---

## 14. Implementation status (2026-08-27)

| Phase | Item | Status |
|---|---|---|
| A.1 | Exact multiscale IoU on full masks with greedy matching + F1 penalty | Done |
| A.2 | EdgeAttNet-style preprocessing (Hough disk, radial flattening, CLAHE) | Done |
| A.3 | Boundary + clDice loss | Done |
| A.4 | CopyAndPaste augmentation | Done (off by default, toggle with `use_copy_paste`) |
| B.5 | Multi-scale TTA | Done (D4 + 3 scales in `balanced`/`max`) |
| B.6 | Dense CRF post-processing | Done (optional, `use_crf=True` if `pydensecrf` installed) |
| B.7 | Hough-circle solar disk mask | Done |
| B.8 | Full MAGFiLO pre-training | Not done — Kaggle public dataset available, requires Colab download |
| C.9 | EdgeAttNet re-implementation | Done (`arch="edgeattnet"`, `in_channels=1`, 512x512, 40 epochs) |
| C.10 | Flat U-Net / Compound U-Net integration | Not done |

### Friend branch benchmark
- Path: `solar-filament-segmentation-boobathi_branch`.
- Best U-Net++ `efficientnet-b4` 512×512 val Dice: **0.64–0.66** (5-fold).
- Their threshold search uses pixel-Dice, not the competition mIoU; their saved `best_threshold=0.8` yields near-zero OOF dice.
- Their `solaris_pipeline.py` uses the same `scse` decoder attention that is now enabled in the single-file notebook.

### Current single-file artifact
- `notebooks/filament_top50_single.py` — 1557 lines.
- `C:\Users\srik2\Desktop\Filament_Colab_Run\filament_top50_single.ipynb` — 8 cells, 96 KB.
- Verified: `py_compile` clean, synthetic smoke test non-empty, notebook verification passed, EdgeAttNet forward pass verified.
