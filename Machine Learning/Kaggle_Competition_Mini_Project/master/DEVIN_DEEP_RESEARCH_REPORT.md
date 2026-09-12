# Devin Deep Research Report — Solar Filament Segmentation Challenge 2026

**Date compiled:** September 11, 2026  
**Research scope:** Competition ground truth, host guidance, peer-reviewed MAGFiLO literature, and production architecture/post-processing analysis for the IEEE BigData Cup / Kaggle *Solar Filament Segmentation Challenge 2026*.

---

## Executive Summary

This report consolidates public host statements, peer-reviewed literature, and the project’s own production telemetry to validate the current strategic path: a **native 2048×2048 YOLOv8l-seg** single-stage instance-segmentation pipeline with strict zero-overlap post-processing. Key findings:

1. **The real leaderboard metric is Panoptic Quality (PQ) at `IoU > 0.50`**, updated on August 7, 2026 and re-scored on August 12, 2026. The earlier public metric was a broken per-prediction Dice average that awarded empty submissions ~0.93 and collapsed to ~0.65 for honest dense submissions.
2. **The host explicitly warns that scores above ~0.35 are already scientifically valuable** and should not be blindly compared with suspect top entries. The verified honest SOTA on the re-scored PQ board is the project’s own **0.360**; public `0.55+` notebooks were forensically shown to be precomputed payload replays (`CHAMPION_PAYLOAD`) or legacy pre-August-12 submissions.
3. **Human inter-annotator PQ on MAGFiLO multi-annotator disks is ~0.36**, making it the practical ceiling for single-annotator supervised models. The MAGFiLO paper reports Cohen’s Kappa = 0.66.
4. **The data and the metric reward native 2048 resolution** because 68.9 % of filaments are < 2,000 px and many are only 2–6 px wide. Downsampling to 1024 or 512 destroys sub-pixel filament connectivity.
5. **YOLO segmentation masks are natively 1/4 of `imgsz` (e.g., 512×512 for 2048 input) unless `retina_masks=True` is used** to resample to the original image size. `mask_ratio` only down-samples the training GT mask canvas; it does not raise output mask resolution without model surgery.
6. **Mosaic augmentation is physically harmful for full-disk solar images.** Ultralytics’ mosaic stitches four images into one training grid, cutting the circular chromosphere disk and long filaments across artificial crosshairs. The project’s own run proved `mosaic=1.0` regressed a 0.360 model to 0.330–0.340.
7. **Panoptic Quality penalizes every missed ground-truth and every spurious detection equally in the denominator.** Controlling confidence and area thresholds, and enforcing a greedy zero-overlap sanitizer, is therefore a first-class optimization lever, not just post-processing.

---

## 1. Competition Details & Host Discussions

### 1.1 Metric update: August 7 / August 12, 2026

The competition overview states that the primary evaluation is Panoptic Quality (PQ), citing Kirillov et al. (CVPR 2019, `10.1109/CVPR.2019.00963`) [^1]:

```
PQ(Y, Ŷ) = Σ_{(y, ŷ) ∈ TP} IoU(y, ŷ) / (|TP| + 0.5|FP| + 0.5|FN|)
```

with segment matching based on `IoU > 0.50` per the Kirillov uniqueness theorem [^2].

The timeline on the competition page records:

- **August 7, 2026:** “We are going to update our evaluation methodology to address the raised issues. … The new scoring metric will be in effect soon; all previous submissions will be updated as well.” [^1]
- **August 9, 2026:** A self-evaluation notebook was released.
- **August 12, 2026:** “Our leaderboard is now re-scored.” [^1]
- **August 20, 2026:** Host Azim Ahmadzadeh posted: “…some participants are spending their time and energy on hacking the system rather tackling the actual problem. We encourage honest participants to continue their work without comparing their scores with others. **Any PQ score of greater than 0.35 is of great value to us.**” [^3][^4]

This is consistent with the project’s own trajectory: the highest **honest, verified** public LB on the re-scored board is **0.360** (`master/03_MODEL_HISTORY.md`, `master/00_EXECUTIVE_SUMMARY.md`).

### 1.2 Why the old public metric was broken

Kaggle discussion `729809` (Valentin Haase) empirically demonstrated the pre-PQ failure mode [^5]:

| Predictions per image | Public score (old metric) |
|-----------------------|---------------------------|
| 0 (all empty)         | 0.93                      |
| ~2 most confident     | 0.72                      |
| 6.4 (tuned pipeline)  | 0.67                      |
| 11.0 (permissive)     | 0.65                      |
| 11.7 (more permissive)| 0.63                      |

The post explains that the old evaluator averaged Dice only over matched pairs, so missed filaments never entered the score, and an empty submission averaged over an empty set returned ~1.0 or a default high value. The suggested fix—Panoptic Quality—puts missed and spurious instances directly in the denominator.

A separate discussion (`728474`) confirmed the all-empty-mask exploit: replacing every `segmentation_rle` with `PPP8` (empty mask) produced a public score of **0.93** [^6]. The host’s response: “For future readers, we have addressed this issue. The updated text in Overview shows the changes.” [^6]

### 1.3 Empty disks, zero-overlap, and submission contract

The submission format is one row per predicted filament: `filament_id, segmentation_rle`. The `filament_id` is `{image_stem}_{i+1}`; the RLE counts string must encode a single instance mask of size 2048×2048 [^1].

Critical host-implied rules (also enforced in the project’s `master/06_TEST_RESULTS.md` and `moonshot_2048/audit_submission.py`):

- **Empty disks:** A disk with zero predicted filaments must emit **zero rows**. Dummy all-zero masks or a single `PPP8` row are non-compliant and, under PQ, score exactly 0.
- **Zero overlap:** Because PQ matching assumes non-overlapping panoptic segments, any two predicted masks on the same disk must have **zero shared pixels**. The project uses a greedy confidence-ordered pixel carve: `m = m_sub[k] & ~occupied`, then `occupied |= m`.
- **Single class:** Categories 1–4 (Left, Right, Unidentifiable, Ambiguous) all map to **Class 0** for the segmentation task [^1].

### 1.4 Multi-annotator ground truth

The training set has **707 unique physical JPEGs** but **1,154 COCO `image_id`s**, because 447 physical disks have 2 or 3 independent annotators (`master/02_DATA_AUDIT.md`).

Kaggle discussion `729487` asked the host how test-set ground truth is constructed when multiple annotators exist. The public search-indexed text of the thread does **not** contain a definitive host answer at the time of this research; the question remains open in the indexed view [^7]. However, the project’s own `match_and_calibrate.py` and `test_host_pq_compat.py` implement a multi-annotator Kirillov PQ evaluator and are reported to agree 100 % with the official host evaluator (`master/06_TEST_RESULTS.md`).

Valentin Haase’s analysis in `729809` provides a crucial calibration point:

> “296 of the 707 training observations carry two or three independent annotation batches. Scoring one annotator's labels against another's, with the same instance matching by overlap, gives a **mean Dice of 0.687 and a median of 0.727** … one annotator marked 409 filaments and the other 378 on the same images, and 95 of the first annotator's filaments (23 %) don't appear in the second annotator's labels at all. So expert against expert sits at roughly 0.69.” [^5]

The project’s internal multi-annotator PQ benchmark for human-vs-human is **0.359–0.369** (`master/00_EXECUTIVE_SUMMARY.md`). This is the practical performance ceiling for a model trained on one annotator’s labels and evaluated against another.

### 1.5 Real scores and the 0.55+ / 0.70 mystery

The project’s forensic audit (documented in `master/03_MODEL_HISTORY.md`, `docs/18_MOONSHOT_BUILD_REPORT.md`, `docs/19_CHATGPT_MASTER_REBOOT_PROMPT.md`) established that public notebooks claiming `0.55+` were **not** legitimate models. They embedded a 182 KB base64 string called `CHAMPION_PAYLOAD` containing **1,342 precomputed test RLE masks** and replayed it as the submission. The apparent `0.70` scores from the pre-August-12 period are consistent with the broken metric (few predictions → high Dice-over-matched-pairs) or with empty-mask exploits.

The honest score distribution on the re-scored PQ board is far tighter:

| Entry / Run | Public LB PQ |
|-------------|--------------|
| V1 U-Net 512 | 0.220 |
| V3/V4 1024 ensemble | 0.300 |
| V8.1 cascade 1024+384 | 0.350 |
| **Moonshot YOLOv8l-seg 2048, Fold 0, `mosaic=0.0`** | **0.360** |
| Moonshot full-data, `mosaic=1.0` | 0.330–0.340 |
| Projected human expert ceiling | 0.359–0.369 |

This table is sourced from `master/00_EXECUTIVE_SUMMARY.md` and `master/03_MODEL_HISTORY.md`.

---

## 2. MAGFiLO Literature & State-of-the-Art Scores

### 2.1 The MAGFiLO dataset

The dataset descriptor was published in *Scientific Data* (Nature) on September 27, 2024 [^8]:

- **10,244 annotated filaments** from **1,593 Hα observations** (2011–2022) captured by the GONG network.
- **1,066 person-hours** of double-blind manual annotation.
- **Cohen’s Kappa = 0.66** reported as the inter-annotator agreement.
- Annotations include polygon segmentation, bounding box, spine, and magnetic chirality (left / right / unidentifiable).

The public release at Harvard Dataverse (DOI `10.7910/DVN/J6JNVK`) contains the full 1,593 observations and 10,244 annotations [^9]. The competition training set is the same release with the 180 test observations removed, yielding **8,199 training annotations over 707 physical images** [^5][^10].

### 2.2 EdgeAttNet — the strongest published MAGFiLO benchmark

*EdgeAttNet: Towards Barb-Aware Filament Segmentation* (arXiv:2509.02964, also DOI `10.1109/ICDMW69685.2025.00230`) is the strongest peer-reviewed result on the full MAGFiLO test split [^11][^12]. It uses a U-Net backbone with Edge-Guided Multi-Head Self-Attention (EG-MHSA) and a learned edge map that modulates the Query and Key matrices at the bottleneck.

| Model | mIoU_pairwise | mIoU_multiscale | Parameters |
|-------|---------------|-----------------|------------|
| U-Net | 0.5724 | 0.5848 | ~31 M |
| U-Net + MHSA (no PE) | 0.5856 | 0.6000 | — |
| U-Net + MHSA (with PE) | 0.6200 | 0.6601 | ~35 M |
| **EdgeAttNet** | **0.6451** | **0.7032** | **22.7 M** |

These numbers come directly from the EdgeAttNet paper and GitHub repository [^11][^13].

Importantly, EdgeAttNet was evaluated with **pairwise and multiscale IoU**, not the competition’s **Panoptic Quality at IoU > 0.50**. The multiscale metric (`mIoU_multiscale`) averages the intersection ratio across downsampled scales to better capture barb geometry [^11]:

```
r(o, õ, δ) = |s(o, δ) ∩ s(õ, δ)| / |s(o, δ)|
IoU_multiscale(o, õ) = ∫_0^1 r(o, õ, δ) dδ
```

This is conceptually aligned with the project’s own `mIoU_multiscale`/`mIoU_pairwise` diagnostics in `code/metrics.py`.

### 2.3 Flat U-Net and Compound U-Net

**Flat U-Net** (*ApJ*, 2025, DOI `10.3847/1538-4357/adadff`) is a lightweight attention U-Net for full-disk Hα images. It reports Dice Similarity Coefficient (DSC) up to **0.82** and recall **0.74** with a SCA/CSA attention block design [^14]. However, the paper does not report PQ or the exact MAGFiLO test split, so the numbers are not directly comparable to the Kaggle leaderboard.

**Compound U-Net** is another recent solar-filament architecture. Its Zenodo artifact reports **Test IoU 0.677 / F1-Score 0.803 on MHAS** and **Test IoU 0.714 / F1-Score 0.831 on HAS** [^15]. These are smaller, non-MAGFiLO datasets, and the metric is semantic IoU/F1 rather than instance PQ.

### 2.4 Literature bottom line

No peer-reviewed paper has yet reported a **PQ > 0.50** on the official MAGFiLO/Kaggle instance-segmentation task. The project’s **0.360** is therefore already at or near the literature frontier for honest, single-model submissions, and it is only ~0.01 below the estimated human inter-annotator PQ ceiling.

---

## 3. Architecture & Post-Processing Analysis

### 3.1 Native 2048 YOLOv8l-seg vs. cascade vs. tiling

The project has tested three architectural paradigms (`master/03_MODEL_HISTORY.md`):

| Generation | Architecture | Resolution | Verified LB | Failure mode |
|------------|--------------|------------|-------------|--------------|
| V1 | Monolithic U-Net | 512 | 0.220 | Down-sampling destroyed thin filaments; semantic output merged instances. |
| V3/V4 | DeepLabV3+/SegFormer/U-Net++ ensemble | 1024 | 0.300 | Watershed separation brittle on curvilinear filaments. |
| V8.1 | YOLOv8 detector → U-Net++ crop refiner | 1024 / 384 crops | 0.350 | Crop boxes truncated long curving filaments. |
| Moonshot | **YOLOv8l-seg native 2048** | 2048 | **0.360** | Best honest single model. |
| Full-data | YOLOv8l-seg native 2048, `mosaic=1.0` | 2048 | 0.330–0.340 | Mosaic seams + prototype over-smoothing. |

**Why native 2048 wins:** `master/02_DATA_AUDIT.md` proves mathematically that 68.9 % of filaments are < 2,000 px, and typical widths are 2–6 px. A 1024 model shrinks a 400 px filament to 100 px; a 512 model shrinks it to 25 px, erasing sub-pixel connectivity. The `filament-segmentation-2026` test images are fixed at 2048×2048 [^1], so a native 2048 model preserves the full spatial signal.

**Tiling / SAHI:** Ultralytics recommends SAHI (Slicing Aided Hyper Inference) for very large images [^16]. For solar disks, however, the full 2048×2048 image already fits into a T4 with `batch=1` and AMP; tiling would cut filaments across tile boundaries and require overlap-aware de-duplication. The project’s empirical result shows that a single forward pass at 2048 outperforms the 1024 cascade and avoids tile-boundary artifacts.

### 3.2 Ultralytics YOLO mask resolution: `retina_masks` and `mask_ratio`

Ultralytics YOLO-seg models produce instance masks from a learned **mask prototype tensor** plus per-instance mask coefficients. A critical, frequently misunderstood behavior is the fixed output resolution of the prototype tensor:

- **The default prototype output is 1/4 of `imgsz`**. For `imgsz=640`, the proto tensor is 160×160; for `imgsz=2048`, it is 512×512. This is because the `Segment` head applies a single ×2 transposed convolution to a stride-8 feature map [^17][^18].
- **`mask_ratio` only controls GT mask downsampling during training** (default 4), not the architecture’s output resolution. Setting `mask_ratio=1` does **not** produce full-resolution masks; it only changes the canvas on which training labels are rasterized [^17][^19].
- **`retina_masks=True` (inference-only)** resamples the 1/4 masks to the original image size using `process_mask_native` [^20][^21]. Without it, `results[0].masks.data` has shape `(N, H/4, W/4)`.

Code-level evidence from Ultralytics `default.yaml` [^22]:

```yaml
overlap_mask: True   # merge instance masks into one mask during training
mask_ratio: 4        # mask downsample ratio (segment only)
retina_masks: False  # high-resolution masks (returns masks at original image size)
```

A maintainer confirms in GitHub issue `22416` [^19]:

> “`mask_ratio` is the factor the masks are downscaled to save GPU memory during training. `retina_masks` is not a training argument. It’s for prediction.”

And in issue `20200` [^17]:

> “`mask_ratio` actually just controls how much the mask is downsampled during preprocessing for training. It doesn't affect the resolution of output mask which is fixed to be 2× the resolution of the input tensor … If you want to increase that, you will have to edit the above line and change 2 to a higher value.”

**Recommendation for the competition:**
- Train at `imgsz=2048` to keep the prototype at the highest feasible resolution (512×512) for the 2048 inputs.
- Use `retina_masks=True` at inference so final masks are resampled to 2048×2048 and then converted to COCO RLE.
- Do **not** rely on `mask_ratio=1` to increase mask sharpness; it will not change the architecture output.

### 3.3 Why `mosaic=1.0` harms solar chromosphere physics

Ultralytics’ mosaic augmentation composes each training batch image from four random images placed in a 2×2 grid [^23]. This is powerful for natural scenes but physically wrong for full-disk solar images because:

1. The Sun is a **single circular disk** with radial intensity fall-off (limb darkening) and a continuous chromospheric absorption pattern. A 2×2 mosaic slices the disk into quadrants, inserts artificial black/background crosshairs, and places pieces of different solar disks next to each other.
2. Long filaments can be **bisected by the mosaic seams**, teaching the prototype masks that filaments naturally terminate at straight horizontal/vertical boundaries.
3. The photometric and scale statistics of the four corners are inconsistent: different days, different seeing, different limb-darkening profiles.

The project’s own full-data run provides direct evidence. The 60-epoch Fold-0 champion with `mosaic=0.0` scored **0.360**. A 50-epoch fine-tune from that checkpoint with `mosaic=1.0` scored **0.330–0.340**, and the post-mortem identified mosaic-induced artificial termination edges and a +8.7 % average mask dilation as root causes (`master/05_ANTIGRAVITY_REPORT.md`, `master/04_CURRENT_DIRECTIVE.md`).

The Ultralytics Academy guidance itself warns that mosaic “produces images that don't look like deployment frames” and recommends disabling it for the final 10 epochs with `close_mosaic=10` [^23]. For this competition the recommendation is stronger: **disable mosaic entirely (`mosaic=0.0`)** for solar disk training.

### 3.4 Panoptic Quality mechanics and the low-confidence penalty

The official competition formula is [^1]:

```
PQ = Σ_{(p,g) ∈ TP} IoU(p, g) / (|TP| + 0.5|FP| + 0.5|FN|)
```

This is the standard Kirillov decomposition:

```
PQ = SQ × RQ
SQ = Σ_{(p,g) ∈ TP} IoU(p, g) / |TP|
RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
```

**Matching rule:** A predicted segment `p` and ground-truth `g` match iff `IoU(p, g) > 0.50`. Kirillov et al. prove that at this threshold, the matching is unique under the non-overlap constraint [^2].

**Implications for low-confidence predictions:**
- A detection with confidence below the operating threshold is not suppressed by NMS; it becomes either a **false positive** (if it does not overlap any GT at > 0.50) or a **poor true positive** (if it overlaps just above 0.50 with low SQ).
- Every **FP** adds `0.5` to the denominator but `0` to the numerator. A single spurious mask at `conf=0.15` therefore directly reduces RQ.
- Every **FN** (missed GT) also adds `0.5` to the denominator. This is the key fix relative to the old metric: the old metric ignored misses entirely.
- Because the matching threshold is binary at `IoU > 0.50`, a mask with `IoU = 0.49` is treated identically to `IoU = 0.01`: both are **FP** for that ground-truth. This makes the 0.50 boundary razor-sharp and explains why the project observed that an 8.7 % area dilation pushed many borderline matches from ~0.52 to ~0.48 (`master/03_MODEL_HISTORY.md`).

The optimal inference policy is therefore a **precision-recall trade-off calibrated to the metric**, not to validation mAP. The project sweeps `conf ∈ {0.15, 0.20, 0.25, 0.30, 0.35}` with `min_area=50` and evaluates the actual PQ on a local multi-annotator evaluator that matches the host (`master/06_TEST_RESULTS.md`).

### 3.5 Zero-overlap sanitizer and RLE submission contract

The project enforces the zero-overlap rule with a GPU greedy carve (`moonshot_2048/ensemble_instances.py`):

```python
# Sort by descending confidence, then greedily remove already-occupied pixels
occupied = torch.zeros((2048, 2048), dtype=torch.bool, device="cuda")
for k in order:
    m = masks[k].bool()
    m = m & ~occupied
    if m.sum() < min_area:
        continue
    clean_masks.append(m)
    occupied |= m
```

After sanitization, each mask is encoded with `pycocotools.mask.encode(np.asfortranarray(mask))["counts"]` into a COCO Fortran RLE counts string. The submission CSV contains one row per mask, with `filament_id = {stem}_{i+1}`.

This matches the host’s stated contract exactly [^1] and passed the project’s `test_submission_contract.py`, `test_rle_roundtrip.py`, and `test_r3_sanitizer.py` audits.

---

## 4. Strategic Recommendations

Based on the research above, the following actions maximize the probability of reaching or exceeding the 0.50 PQ target while remaining competition-legal:

1. **Stay on native 2048 YOLOv8l-seg.** Do not revert to 1024/512 semantic models or crop refiners. The data geometry (68.9 % of filaments < 2,000 px) makes resolution the dominant lever.
2. **Keep `mosaic=0.0`.** The project’s 0.360 champion already proved this. Any mosaic, even with `close_mosaic=10`, introduces physically inconsistent disk cross-sections for the Sun.
3. **Use `retina_masks=True` at inference** to obtain full 2048×2048 masks. Do not rely on `mask_ratio` to change output resolution.
4. **Calibrate `conf` and `min_area` against the local PQ evaluator, not mAP.** Because PQ penalizes FP and FN symmetrically, the optimal confidence is likely in the 0.25–0.35 range; the project’s sweep already explored 0.15–0.35 and found a plateau near 0.30.
5. **Avoid ensembling 2048 native with 1024 cascade.** The project’s ensemble dropped 0.360 → 0.350 because the 1024 masks had dilated boundaries and added 146 uncorroborated FPs (`master/03_MODEL_HISTORY.md`).
6. **Do not train beyond convergence.** The full-data fine-tune from the 60-epoch `best.pt` for another 50 epochs caused prototype over-smoothing and +8.7 % dilation. If retraining, start from a high-quality initialization but stop when validation PQ plateaus.
7. **Treat 0.36 as near the honest ceiling.** Because human inter-annotator PQ is ~0.36, the only path to 0.50+ is likely a combination of (a) native 2048, (b) an edge/barb-aware head (inspired by EdgeAttNet), (c) metric-aware post-processing, and (d) possibly a legal ensemble of two 2048-native models with low-correlation errors.
8. **Do not use the public Harvard Dataverse MAGFiLO archive.** The host explicitly bans it as leakage; the competition training set already contains all usable training observations (`master/00_EXECUTIVE_SUMMARY.md`, `master/04_CURRENT_DIRECTIVE.md`).

---

## References

[^1]: Kaggle, *Solar Filament Segmentation Challenge 2026* overview and rules. https://www.kaggle.com/competitions/filament-segmentation-2026
[^2]: A. Kirillov et al., “Panoptic Segmentation,” *CVPR 2019*. https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf
[^3]: Kaggle discussion `737177`, “Difference between the top scores” (host comment). https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/737177
[^4]: Kaggle search-indexed host quote: “Any score higher than 0.35 is of great value for us and we will examine them carefully.”
[^5]: Kaggle discussion `729809`, “Some measurements on the public metric, and a possible fix” (Valentin Haase). https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809
[^6]: Kaggle discussion `728474`, “Possible evaluation issue: all-empty masks score > 0.9.” https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728474
[^7]: Kaggle discussion `729487`, “Clarification: how is the test-set ground truth constructed when an image has multiple annotators?” https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729487
[^8]: A. Ahmadzadeh et al., “A dataset of manually annotated filaments from H-alpha observations,” *Scientific Data* 11, 1031 (2024). https://www.nature.com/articles/s41597-024-03876-y
[^9]: Harvard Dataverse, MAGFiLO v1.0. https://doi.org/10.7910/DVN/J6JNVK
[^10]: MLEcoFi, MAGFiLO dataset description. https://www.mlecofi.net/magfilo
[^11]: V. Solomon et al., “EdgeAttNet: Towards Barb-Aware Filament Segmentation,” arXiv:2509.02964 (2025). https://arxiv.org/abs/2509.02964
[^12]: IEEE ICDM Workshops, EdgeAttNet DOI. https://doi.org/10.1109/ICDMW69685.2025.00230
[^13]: `dasjar/EdgeAttNet` GitHub repository. https://github.com/dasjar/EdgeAttNet
[^14]: G. Zhu et al., “Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk Hα Images,” *ApJ* 980, 176 (2025). https://iopscience.iop.org/article/10.3847/1538-4357/adadff
[^15]: Compound U-Net Zenodo artifact. https://doi.org/10.5281/zenodo.17230604
[^16]: Ultralytics, “Slicing Aided Hyper Inference (SAHI).” https://docs.ultralytics.com/guides/sahi-tiled-inference.md
[^17]: Ultralytics GitHub issue `20200`, “mask_ratio not working.” https://github.com/ultralytics/ultralytics/issues/20200
[^18]: Ultralytics `yolov8-seg.yaml` head definition. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-seg.yaml
[^19]: Ultralytics GitHub issue `22416`, “About mask_ratio, retina_masks, seed parameters during train.” https://github.com/ultralytics/ultralytics/issues/22416
[^20]: Ultralytics GitHub issue `18859`, “how to fix image size for yolo prediction.” https://github.com/ultralytics/ultralytics/issues/18859
[^21]: Ultralytics GitHub issue `4909`, “Upscaled masks not so accurate.” https://github.com/ultralytics/ultralytics/issues/4909
[^22]: Ultralytics `default.yaml`. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
[^23]: Ultralytics Academy, “Use Augmentation Carefully / Mosaic and close-mosaic.” https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/use-augmentation-carefully
[^24]: PyTorch-Metrics Panoptic Quality documentation. https://lightning.ai/docs/torchmetrics/stable/detection/panoptic_quality.html

---

**Internal project cross-references:** `master/00_EXECUTIVE_SUMMARY.md`, `master/01_PROJECT_STATE.md`, `master/02_DATA_AUDIT.md`, `master/03_MODEL_HISTORY.md`, `master/04_CURRENT_DIRECTIVE.md`, `master/05_ANTIGRAVITY_REPORT.md`, `master/06_TEST_RESULTS.md`, `master/07_KAGGLE_RUNBOOK.md`, `AGENTS.md`.
