# Section 3 Research Report — Why the 0.360 PQ Model Stalls and What Is Actually Broken

## Executive verdict

The project’s central diagnosis is directionally correct but needs four important corrections before another expensive training run.

First, the **stride-4 YOLO mask-prototype grid is a real spatial bottleneck**, especially for thin 2–6 px structures, but the measured 0.9196 mean IoU from the project’s downsample/upsample experiment is **not a mathematical hard ceiling on YOLO SQ or PQ**. It is a useful empirical proxy for information loss under one rasterization experiment. The pinned Ultralytics 8.3.145 implementation does assemble masks on the low-resolution prototype grid and only then upsamples them, so the bottleneck is real. However, learned real-valued prototypes and coefficients can place contours differently from a literal downsampled binary mask, and PQ also depends heavily on Recognition Quality (FP/FN), not only SQ.

Second, **`mask_ratio=1` alone is not a solution**. In Ultralytics 8.3.145, the segmentation loss explicitly resizes the ground-truth masks to the prototype tensor’s spatial dimensions before computing BCE. The `Proto` module itself receives P3/8 features and performs one ×2 transposed convolution, yielding a stride-4 prototype map. Raising the data-mask resolution without raising the prototype resolution therefore does not provide full-resolution supervision to the output basis. A real solution requires changing the mask head, or adding a high-resolution second-stage renderer/refiner.

Third, the project’s **24.3% median bbox fill-ratio is valid, but the stated mechanism is not**. YOLOv8-seg is not a Mask R-CNN-style ROIAlign system. It predicts global prototype masks plus per-instance mask coefficients and then crops the assembled masks to boxes. Thus, “75.7% empty ROI background dilutes ROIAlign features” is not an accurate description of the architecture. Low fill ratio still matters because axis-aligned boxes are a poor geometric summary of long curved filaments, because boxes constrain/crop the final mask, and because proposal assignment/localization can fail on elongated structures. It does not, by itself, justify abandoning YOLO.

Fourth, the measured **human-vs-human PQ of 0.3329 is not a proven mathematical ceiling**. It is an important empirical warning about label noise and ambiguity. The competition’s public documentation states that images can have multiple independent annotators, but as of this research the public clarification thread asking how hidden test ground truth is constructed has no indexed organizer answer. Without knowing whether hidden scoring uses one rater, pooled labels, averaged per-rater scores, or another construction, 0.33–0.37 cannot be called a theorem-level ceiling. Multi-rater segmentation literature also demonstrates that consensus/soft-label models can sometimes agree with individual raters better than raters agree with each other.

The strongest near-term conclusion is therefore:

> **Do not spend the next GPU budget on a wholesale architecture switch. First extract and exploit the information already present in the 0.360 checkpoint: raw mask logits, metric-aligned instance calibration, soft TTA, and a better zero-overlap resolver. If that saturates, add a high-resolution boundary refiner such as a PointRend-style head or a carefully designed crop refiner. Only then justify replacing YOLO.**

The project’s own red-team audit makes this prioritization especially compelling: the 0.360 champion already exists; the full-data model lost 31% of champion detections on the sampled disks; 6.9% of champion masks are fragmented after pixel carving; and the binary ensemble added 141 predictions while dropping LB from 0.360 to 0.350. Those are Recognition Quality and post-processing failures that can be attacked without retraining the entire model.

---

## 1. Evidence map: which Section 3 claims survive external validation?

| Section 3 claim | Research verdict | Confidence | Practical consequence |
|---|---|---:|---|
| YOLO stride-4 prototypes are a boundary bottleneck | **Validated, with nuance** | High | Keep as a major architectural limitation, but do not call 0.9196 a hard PQ/SQ theorem. |
| `mask_ratio=1` could raise prototype resolution | **Rejected for stock 8.3.145** | Very high | `mask_ratio=1` alone is insufficient; modify `Proto` or add a refiner. |
| `retina_masks=True` fixes prototype detail | **Rejected as a fundamental fix** | Very high | It bilinearly rescales assembled low-res logits; test it, but it cannot create new basis detail. |
| 24.3% fill ratio means ROIAlign background dilution | **Mechanism rejected** | Very high | YOLOv8-seg has no ROIAlign mask pooling; geometry still matters, but for different reasons. |
| Multi-annotator PQ 0.33–0.37 is a hard ceiling | **Not established** | High | Treat as an empirical noise range. Test consensus methods rather than assuming no headroom. |
| Greedy pixel carve leaks PQ through fragmentation | **Validated and externally consistent with GT protocol** | Very high | Replace global-confidence carving with soft per-pixel ownership; then apply conditional connectivity cleanup. |
| Binary TTA/ensemble is dangerous for PQ | **Validated for this project** | Very high | Fuse aligned logits/probabilities before thresholding; never union binary masks. |
| Soft TTA is possible with YOLO-seg | **Validated** | Very high | Keep float mask logits before `.gt_(0)`; reconstruct per view, invert transforms, cluster instances, fuse maps. |
| Coefficients can simply be averaged across TTA views | **Rejected** | High | Coefficients are defined relative to view-specific prototypes; fuse reconstructed logits/maps, not coefficient vectors. |
| A fixed confidence threshold is enough | **Rejected** | High | Optimize an expected PQ contribution/quality score using grouped OOF data. |
| Mosaic should remain banned | **Strongly supported by project evidence** | High | Keep `mosaic=0.0`. It is a physically implausible transform for intact full-disk solar observations. |
| 110 epochs alone caused the regression | **Not isolated** | High | Drift is observed, but causality is confounded by mosaic, data scope, batch, and schedule changes. Do controlled ablations. |

Internal evidence is from the uploaded `MASTER_PROMPT_SEPT13(1).md`, `10_RED_TEAM_FORENSIC_AUDIT(1).md`, `02_DATA_AUDIT(1).md`, `03_MODEL_HISTORY(1).md`, and `05_ANTIGRAVITY_REPORT(1).md`.

---

## 2. Structural barrier #1 — YOLO prototype resolution: real bottleneck, wrong “hard ceiling” framing

### 2.1 What Ultralytics 8.3.145 actually does

The project pins `ultralytics==8.3.145`, so the exact version matters. In that implementation:

1. The YOLOv8 segmentation head receives P3/P4/P5 detection features. The P3 stream is at stride 8 relative to the input.
2. The `Proto` module applies one transposed convolution with stride 2. Thus a 2048×2048 input produces a prototype grid at approximately 512×512 (stride 4).
3. Per-instance mask coefficients are linearly combined with those prototypes to create a floating mask-logit map.
4. Standard mask processing crops at prototype resolution and optionally bilinearly upsamples. `process_mask_native` bilinearly scales the assembled mask and then crops. Both paths eventually threshold the floating result at zero.
5. During training, if the ground-truth mask tensor does not match `mask_h × mask_w`, Ultralytics explicitly interpolates it down to the prototype dimensions before computing binary cross-entropy with the assembled logits.

This validates the main architectural concern: thin boundaries are supervised and represented on a stride-4 grid before the final image-size raster is produced.

### 2.2 Why the project’s 0.9196 experiment matters

The red-team experiment took 60 true masks, shrank them to 512×512, and restored them to 2048×2048. Bilinear recovery achieved mean IoU 0.9196, while nearest-neighbor recovery achieved 0.8071. That is a strong empirical demonstration that a coarse representation erodes thin filament masks, and that interpolation choice matters.

However, the phrase **“hard architectural SQ ceiling = 0.92”** goes too far. The experiment measures a particular binary-mask resampling pipeline. A learned prototype representation is a collection of real-valued basis functions. Their linear combination can position threshold crossings differently from a directly downsampled binary target. More importantly, PQ is not equal to mean mask IoU: unmatched instances dominate RQ, and any prediction under the 0.5 matching threshold contributes zero numerator while becoming an FP/FN pair.

A more accurate statement is:

> **Stride-4 prototypes impose a strong spatial-frequency and boundary-localization bottleneck whose magnitude is empirically large on MAGFiLO-scale thin masks; 0.9196 is a useful resampling reference, not a formal upper bound on model SQ or PQ.**

### 2.3 Why `mask_ratio=1` alone will not fix it

This is the clearest code-level answer to Section 3. Stock 8.3.145 uses `mask_ratio=4` as a data/training setting, but the loss checks whether the mask dimensions match the prototype dimensions and resizes the ground truth to `(mask_h, mask_w)` when they do not. Therefore:

- `mask_ratio=1` may keep a higher-resolution target tensor earlier in the data path.
- But if the prototype is still 512×512, the loss downsamples the target to 512×512 immediately before mask loss.
- The basis itself remains stride 4.

Thus **a `mask_ratio=1` experiment without changing `Proto` is not the high-resolution-head experiment the project wants**.

### 2.4 Why `retina_masks=True` is useful but insufficient

The pinned predictor selects `process_mask_native` when `retina_masks=True`. That routine first assembles masks on the prototype grid, bilinearly scales them to the original shape, crops, and thresholds. This can change box-edge behavior and may be worth an OOF A/B test. It does not add high-frequency information that was absent from the prototype basis.

For the same reason, setting `retina_masks=True` should be classified as **better rendering of existing logits**, not **higher-resolution prediction**.

### 2.5 Which high-resolution modification is actually sensible?

A brute-force full-resolution prototype head is expensive. A 32-channel prototype tensor alone is roughly:

- 512²: ~8.4M elements = ~16 MiB FP16 / ~32 MiB FP32.
- 1024²: ~33.6M elements = ~64 MiB FP16 / ~128 MiB FP32.
- 2048²: ~134.2M elements = ~256 MiB FP16 / ~512 MiB FP32.

Those numbers exclude the much larger intermediate activation maps, gradients, optimizer state, and backbone features. In the stock `Proto` module, the intermediate channel count is 256, so naively moving that tensor to 2048² would be prohibitive on a 16 GB T4.

The best architectural escalation order is therefore:

1. **No-training rendering upgrade:** keep raw logits, bilinear-native rendering, tune mask threshold.
2. **PointRend-style boundary head:** refine only uncertain boundary points at high resolution. PointRend was explicitly designed to avoid the memory cost of dense high-resolution prediction and produces sharper instance/semantic boundaries.
3. **ROI/crop refiner at native source resolution:** detector proposes an instance; a small high-res network sees a padded crop and predicts a refined mask using BCE + IoU/topology-aware losses. The crop must be substantially larger than the predicted box to avoid the V8.1 truncation failure.
4. **Stride-2 prototype experiment:** add another carefully designed upsampling/refinement stage, ideally after reducing channels before upsampling. This is plausible but requires retraining and memory profiling.
5. **Full-resolution dense prototype basis:** lowest priority on T4; poor compute/memory trade-off.

**Decision:** do not start by forking the model to `mask_ratio=1`. If architecture work becomes necessary, PointRend/high-res refinement is the more rational first experiment.

---

## 3. Structural barrier #2 — bbox fill ratio: the statistic is useful, the ROIAlign story is not

The project measured a median filament-area / bbox-area ratio of 24.3% across all 8,199 training annotations. That is a meaningful geometric property: filaments are curved, sparse shapes poorly summarized by axis-aligned rectangles.

But YOLOv8-seg does not feed the rectangular ROI through an ROIAlign mask head. The segmentation head follows the YOLACT pattern: global prototypes are generated for the image, per-instance coefficient vectors are predicted from detection features, and the final mask is a linear combination of the prototypes followed by box cropping. YOLACT’s original design explicitly emphasizes that it avoids re-pooling.

So what does low bbox fill actually imply?

- **Box localization is an imperfect proxy for filament geometry.** A long curved filament can require a large box even when the mask occupies little of it.
- **Box cropping can truncate a correct mask** if localization is too tight or wrong near an extremity/barb.
- **Object assignment/detection remains box-driven**, so tiny or thin instances may be hard to assign confidently even when a mask branch could represent them.
- **A second-stage crop refiner can indeed suffer background dilution** if it literally consumes the rectangular crop; this was relevant to the V8.1 architecture, but not as a description of the YOLO prototype head itself.

### Should YOLO be abandoned because of 24.3% fill?

No. The best verified score came from YOLO at native 2048, and older semantic+watershed systems performed worse. A wholesale switch throws away a proven detector without first fixing lower-risk leaks.

Alternative architectures remain legitimate A/B candidates:

- **CondInst** removes ROI cropping and uses dynamic instance-conditioned mask heads. This is conceptually attractive for curved instances, but it is not automatically a full-resolution solution and would require a new training stack.
- **Mask2Former** is a powerful universal segmentation architecture and has excellent panoptic results on standard benchmarks, but it is a much heavier engineering and training change. Its generic benchmark superiority is not evidence that it will beat a tuned 2048 YOLO on this very small, noisy, multi-annotator solar dataset.
- **EdgeAttNet** is highly domain-relevant because it explicitly incorporates edge information for MAGFiLO and improves fine barb segmentation, but it is a semantic segmentation architecture evaluated under different metrics. It is better viewed as a source of boundary-refiner ideas than as an immediate replacement for instance proposal logic.

**Decision:** retain YOLO as the instance proposal engine. If an architecture extension is needed, decouple detection from mask rendering: use YOLO for instance hypotheses and a domain-specific high-resolution refiner for boundaries/topology.

---

## 4. Structural barrier #3 — multi-annotator disagreement is a noise floor, not a proven ceiling

### 4.1 What is empirically known

The project has 707 physical training images represented by 1,154 COCO observation records because many disks are independently annotated more than once. The red-team audit scored 40 human-vs-human pairs with the host PQ implementation and measured:

- SQ = 0.6310
- RQ = 0.5276
- PQ = 0.3329
- 47.2% of filament pairs unmatched at the >0.5 IoU criterion

That is strong evidence of major label ambiguity. MAGFiLO’s publication reports Cohen’s kappa of 0.66, independently confirming nontrivial inter-rater disagreement.

### 4.2 What is *not* publicly established

The competition data page says a physical image may have multiple independent sets of annotations and tells participants they may treat the corresponding records as different images. The competition’s public overview says submissions are evaluated against MAGFiLO ground truth using PQ. But the public discussion currently contains an explicit unanswered clarification titled “how is the test-set ground truth constructed when an image has multiple annotators?”

Therefore the public evidence does not establish whether the hidden evaluator:

- chooses one annotator,
- scores against each annotator and averages,
- pools annotations,
- forms a consensus,
- or applies another rule.

Without that information, the numerical range 0.33–0.37 is **not a fundamental upper bound**.

### 4.3 Why models can exceed pairwise human agreement

Pairwise inter-rater performance is not generally a maximum possible model-to-rater performance. A model can learn systematic common structure across many raters and approximate the central tendency of the labeling distribution. In multi-rater segmentation research, soft-label methods formed by averaging multiple expert masks can yield well-calibrated predictions and can, on average, match physician annotations better than other individual physicians. STAPLE likewise models an unknown latent segmentation and rater reliability rather than treating any one annotation as perfect truth.

This is directly relevant to the project: contradictory annotations are not only noise; they are information about uncertainty.

### 4.4 Do not use hard consensus intersection as the default

For 2–6 px structures, pixelwise intersection is especially dangerous. A 1–2 px boundary displacement between raters can erase a barb or sever a narrow bridge entirely. Hard intersection maximizes precision at the cost of recall and can create targets that resemble no individual expert’s mask.

Prefer one of these approaches:

1. **Random-rater sampling per physical image per epoch.** Sample each physical disk once, then choose one available annotator set. This avoids overweighting multi-annotated disks while preserving the observed labeling distribution.
2. **Soft consensus masks.** Match corresponding instances across annotators, then average their binary masks into probabilities. Train a refiner to predict these probabilities; threshold can later be calibrated for PQ.
3. **STAPLE/weighted consensus as an ablation, not dogma.** Estimate a latent mask while modeling rater performance. Validate against held-out individual annotators.
4. **Confidence-weight uncertain instances.** Instances seen by all raters can receive higher target weight; one-rater-only instances should not automatically be deleted.

### 4.5 The experiment that resolves the ceiling question locally

Use only physical images with multiple annotations and perform leave-one-annotator-out evaluation:

- Method A: train on annotation records independently (current style).
- Method B: random-rater-per-epoch.
- Method C: soft consensus.
- Method D: hard intersection.

For each held-out physical disk, evaluate the predicted instance set against **each individual rater separately** with the exact host PQ and macro-average the result. Measure PQ, SQ, RQ, and count bias. If soft consensus beats the best single-rater/random-rater strategy, the project has empirically demonstrated that human-human PQ is not its learning ceiling.

**Decision:** treat 0.3329 as a warning about label uncertainty, not a reason to stop. The test-ground-truth construction remains a critical unknown.

---

## 5. Fixable leak #1 — greedy zero-overlap carving is throwing away morphology

The red-team audit found that 27 of 390 champion predictions (6.9%) became multi-component masks after the confidence-ordered carve, with as many as six components. This is highly actionable. It is also inconsistent with the MAGFiLO annotation protocol: the dataset paper explicitly requires an acceptable filament segmentation to be one piece, contain no islands, contain no holes, and tightly fit the filament body.

### 5.1 Why “largest connected component only” is directionally right but incomplete

Keeping only the largest component will remove islands and can improve metric alignment. But if greedy carving split a genuinely correct long filament into two substantial components, deleting the second component can destroy true area and reduce IoU further.

The deeper problem occurs **before** connected-component cleanup: overlapping pixels are assigned solely by global instance confidence, not by local pixel evidence.

### 5.2 Replace global carving with soft pixel ownership

Keep a floating logit/probability map `L_i(x)` for every candidate mask. In pixels claimed by multiple candidates, assign ownership to the candidate with the strongest local support, optionally regularized by instance confidence:

`owner(x) = argmax_i [ L_i(x) + alpha * logit(conf_i) ]`

Then:

1. Threshold each candidate probability map at its calibrated mask threshold.
2. For every overlapping pixel, select exactly one owner by the local score above.
3. Remove secondary components only when they are clearly insignificant (e.g., below an OOF-calibrated absolute area or relative fraction of the main component).
4. Never blindly fill holes that are owned by another instance; only restore holes in unclaimed pixels, then rerun the zero-overlap audit.

This preserves the host contract while using the model’s pixel-level evidence rather than a single scalar confidence for the entire mask.

### 5.3 Connectivity cleanup should be tuned to PQ, not aesthetics

The project’s internal estimate of +0.01–0.02 PQ from component cleanup should not be treated as guaranteed. The correct local experiment is an OOF sweep over:

- secondary-component absolute area,
- secondary/main area ratio,
- minimum bridge probability,
- ownership confidence weight `alpha`,
- mask threshold.

Measure the exact host PQ after each variant. Because the ground-truth protocol strongly prefers connected, hole-free masks, this is one of the safest post-processing directions available.

**Decision:** implement soft overlap ownership before largest-component cleanup. This is higher priority than retraining.

---

## 6. Fixable leak #2 — soft TTA is feasible, but fuse reconstructed maps, not coefficient vectors

The project’s 2048+1024 binary ensemble increased the submission from 1,182 to 1,323 rows and dropped LB from 0.360 to 0.350. That result is fully consistent with PQ: unmatched extra instances increase the denominator by 0.5 each.

Competition practice on analogous thin-object tasks also supports probability-level fusion: the Sartorius 1st-place solution averaged model mask probabilities and thresholded the fused result, rather than unioning binary masks.

### 6.1 The pinned Ultralytics code exposes the needed signal

In `process_mask`, Ultralytics computes:

`masks = masks_in @ protos`

and only at the end applies `.gt_(0.0)` to create binary masks. Thus the pipeline can retain the pre-threshold floating mask logits without retraining.

### 6.2 Why coefficient averaging across TTA views is wrong

For view `v`, an instance mask is approximately:

`M_v = C_v P_v`

where `C_v` is the coefficient vector and `P_v` is the set of prototypes generated from that transformed image. The basis `P_v` is image/view-dependent. A coefficient dimension in one transformed view is not guaranteed to mean the same spatial basis function in another view.

Therefore:

> **Do not average coefficient vectors across views. Reconstruct each instance’s full logit/probability map in its own prototype basis first, invert the image transform, spatially align the result, cluster corresponding instances, then fuse the maps.**

### 6.3 Recommended custom soft-TTA pipeline

Use conservative geometry-preserving views first: original, horizontal flip, vertical flip. For each view:

1. Run the model once at native 2048.
2. Keep detections, confidence, boxes, prototype tensors, and coefficients.
3. Reconstruct floating mask logits before thresholding.
4. Bilinearly map logits to the original 2048 coordinate system.
5. Undo the flip.
6. Match instances between views using box IoU plus soft-mask IoU/centroid distance.
7. For matched clusters, average probabilities or calibrated logits.
8. Compute a TTA-consistency feature: number of supporting views, pairwise mask IoU, and confidence variance.
9. Threshold only once after fusion.
10. Send fused masks into the soft zero-overlap resolver.

Unmatched one-view detections should be heavily down-weighted rather than automatically retained. For PQ, TTA agreement is useful as an estimate of whether a candidate is a true object, not merely as a way to increase detections.

**Decision:** soft TTA is a high-value no-training experiment. The correct fusion object is the aligned float mask map, not the coefficient vector and not the binary mask.

---

## 7. Fixable leak #3 — confidence should be an expected-PQ decision, not a global scalar threshold

The full-data run demonstrates that threshold changes matter but cannot repair a bad model: conf 0.15 produced 1,281 rows and LB 0.340; conf 0.30 produced 911 rows and LB 0.330. Lowering confidence added 370 rows for only +0.010 PQ. This argues against blindly lowering the champion threshold. It does **not** prove the champion is calibrated optimally, because its score distribution and errors are different.

### 7.1 A useful marginal PQ rule

Let current PQ be:

`Q = S / D`

where `S` is the sum of IoUs for true-positive matches and `D = TP + 0.5 FP + 0.5 FN`.

Consider admitting one candidate instance. Let:

- `p` = calibrated probability that the candidate will uniquely match a ground-truth instance at IoU > 0.5,
- `q` = expected IoU conditional on a match.

Relative to rejecting the candidate, accepting it increases the denominator by 0.5 whether it becomes a TP (a previous FN is replaced by a TP) or an FP. Its expected numerator gain is approximately `p*q`. Under the local independence/no-duplicate approximation, accepting the candidate improves PQ when:

`p * q > Q / 2`

At `Q ≈ 0.36`, the candidate should contribute more than roughly **0.18 expected IoU**. If a matched candidate is expected to have IoU 0.75, this corresponds to match probability above ~0.24. This is **not** the same thing as a YOLO confidence threshold of 0.24.

The rule is an analytical guide; real candidate interactions, duplicate detections, and overlap competition mean the final decision must still be validated by exact OOF PQ.

### 7.2 Build an instance-quality calibrator

For every OOF candidate, compute features such as:

- YOLO object confidence,
- box area and aspect ratio,
- predicted mask area,
- mask/box fill ratio,
- mean and lower-quantile mask probability inside the mask,
- probability margin near the boundary,
- radial position on the solar disk,
- TTA support count and agreement,
- overlap with stronger candidates,
- connected-component count before/after ownership resolution.

Label the candidate with:

- whether it matches a GT instance at IoU > 0.5,
- actual matched IoU if it matches.

Estimate `u(x) = E[ IoU * 1{IoU>0.5} | x ]`. Rank candidates by `u`, and optimize the admission threshold, NMS IoU, mask threshold, and min area on the exact grouped OOF PQ evaluator.

This is more aligned with the target metric than object confidence alone.

### 7.3 Calibration dimensions that should be swept jointly

At minimum:

- detection confidence,
- NMS IoU,
- mask-logit threshold,
- minimum area,
- radial cutoff / limb prior,
- overlap ownership `alpha`,
- component retention rule,
- TTA support requirement.

**Decision:** the “optimal confidence” is not a single universal number derivable from PQ. The metric suggests an expected-contribution rule; the actual threshold must be learned on physical-image-grouped OOF predictions.

---

## 8. Training failure #1 — mosaic: keep the permanent ban

The 0.360 champion used native 2048 images and `mosaic=0.0`. The full-data run used `mosaic=1.0` for most of training and regressed to 0.330–0.340. Cross-matching showed 31% of sampled champion detections did not match the full-data model above IoU 0.5, while the degraded model produced fewer predictions with larger mean mask area.

Mosaic is a generic object-detection augmentation. For full-disk solar imagery it constructs training scenes that cannot occur physically: multiple disk quadrants, artificial crosshair boundaries, discontinuous chromosphere texture, and severed filaments. This transformation directly attacks the task’s most difficult property, structural continuity.

Strictly speaking, the run is not a single-variable A/B because the data scope, batch size, training duration, and initialization/schedule also differed. Nevertheless, the physical argument and empirical result point in the same direction.

**Decision:** keep `mosaic=0.0` as architectural law unless a future controlled experiment demonstrates otherwise. Do not spend more GPU budget retesting ordinary four-image mosaic.

### Related augmentations

- **MixUp:** avoid. Blending two complete solar disks is physically implausible and creates contradictory textures/labels.
- **Copy-paste:** avoid as a default. A filament’s contrast and morphology depend on local chromospheric context and radial geometry; naive pasting produces unrealistic boundaries.
- **Horizontal/vertical flips:** plausible for single-class morphology, but validate because solar hemispheric/chirality priors are not perfectly symmetric.
- **Rotation:** centered rotations preserve disk geometry, but large rotations can erase real latitude-related priors. Tune on grouped OOF rather than assuming 180° invariance.
- **Photometric normalization/contrast perturbation:** more physically defensible than multi-image composition, especially for site/seeing variation, but should be mild.

---

## 9. Training failure #2 — “110 epochs caused over-training” is plausible but not proven

The full-data fine-tune exhibits real drift: predicted masks became larger, the submission count fell, and public PQ regressed. Those are facts. What is not isolated is the cause.

Between the champion and degraded model, the project changed several things at once:

- training data: 82% fold → 100%,
- initialization: already-trained 60-epoch checkpoint → further fine-tuning,
- mosaic: 0 → 1.0,
- batch: 2 → 1,
- total optimization steps and schedule,
- validation setup.

Thus “110 total epochs caused the regression” should be restated as:

> **The additional training run produced mask/detection drift consistent with over-training or catastrophic forgetting, but the experiment is confounded and cannot attribute the regression to epoch count alone.**

### 9.1 Do not choose an epoch count from the leaderboard run

The correct next training experiment needs a real physical-image-grouped validation set and should log host PQ every epoch or at frequent checkpoints. COCO mask mAP is demonstrably insufficient: the degraded full-data run improved its reported mask mAP while worsening public PQ.

### 9.2 If a training experiment is approved, use controlled branches

The most informative design is not “80–100 epochs on all data and hope.” It is:

- same physical-image split,
- same native 2048 resolution,
- `mosaic=0`,
- same inference/post-processing,
- exact host PQ on validation,
- checkpoint frequently.

Compare two controlled branches:

**Branch A — conservative champion adaptation**
- initialize from 0.360 checkpoint,
- very small learning rate,
- short schedule (e.g., 10–20 epochs, determined by validation PQ rather than a fixed total),
- optionally freeze most of the backbone initially,
- stop immediately when PQ/RQ or mask area begins to drift.

**Branch B — clean pretrained restart**
- initialize from the original generic pretrained YOLOv8l-seg weights,
- train on the same grouped split with no mosaic,
- stop by local PQ.

This isolates whether the champion checkpoint is a beneficial domain initialization or a trap. Only after that comparison should the winning schedule be expanded to all training images for a final model.

**Decision:** do not infer “optimal total epochs” from 60 vs 110. Optimize checkpoint selection by grouped validation PQ.

---

## 10. Should the project abandon YOLO-seg?

### Answer: not yet

The evidence supports **extending** the 0.360 system before replacing it.

Why:

1. It is the only architecture in the project proven at 0.360 under the current PQ leaderboard.
2. The red-team identified fixable post-processing and recognition losses in the champion itself.
3. The most serious “bbox fill” architectural argument was partly based on an incorrect ROIAlign analogy.
4. Alternative models introduce new instance-separation and training risks. EdgeAttNet is semantic; Mask2Former is heavier; CondInst requires a new stack; semantic connected components revive the merge/separation problem that hurt earlier versions.
5. High-resolution boundary rendering can be attached to the existing detector without losing the detector’s learned domain representation.

The architecture-switch trigger should be empirical:

> **Abandon YOLO only if an OOF-controlled alternative beats the champion pipeline after raw-logit calibration and improved overlap resolution, not because a generic architecture is theoretically more modern.**

A strong next-generation architecture would likely be hybrid:

`YOLO 2048 instance proposal -> padded native-resolution crop -> edge/topology-aware high-res mask refiner -> soft ownership / PQ calibrator`

This combines the project’s proven instance localization with the strongest lesson from thin-structure literature: allocate high-resolution computation to the boundary and topology rather than densifying the entire 2048 feature stack.

---

## 11. Highest-impact next action: a no-training “Champion Re-render & Recalibrate” experiment

This is the recommended next production experiment before any new training run.

### Phase A — expose float masks

Patch the pinned 8.3.145 inference path so `process_mask` / `process_mask_native` returns floating assembled logits (or sigmoid probabilities) before `.gt_(0.0)`. Preserve:

- box confidence,
- boxes,
- mask coefficients/prototypes if needed for debugging,
- full aligned probability map,
- instance metadata.

No model weights change.

### Phase B — use the champion’s untouched validation fold

The 0.360 checkpoint was trained on ~82% of the data and retained a physical-image holdout. That is extremely valuable. Generate predictions on that holdout and evaluate with the exact host-compatible PQ code already verified by the test suite.

Tune all post-processing on this holdout, not on Kaggle LB.

### Phase C — calibrate mask rendering

Sweep:

- mask threshold in logit/probability space,
- `retina_masks`/native vs default rendering,
- small-area cutoff,
- limb/radial priors.

The objective is host PQ, with SQ and RQ recorded separately.

### Phase D — replace overlap carving

Compare:

1. current confidence-ordered carve,
2. soft per-pixel ownership,
3. soft ownership + conditional connected-component cleanup.

Audit:

- zero overlap,
- connected-component count,
- hole count,
- matched IoU distribution,
- TP/FP/FN.

### Phase E — custom soft TTA

Use original + H flip + V flip first. Reconstruct/invert masks, cluster matching instances, probability-average, compute TTA agreement, threshold once. Compare to no TTA. Do not use binary union.

### Phase F — learn instance utility

Fit/calibrate expected PQ contribution `u(x)` using the validation fold. Jointly optimize candidate threshold and post-processing parameters against exact PQ.

### Go/no-go rule

- If these changes produce a clear OOF PQ improvement with stable gains across physical-image subsets, make a small set of test submissions.
- If they do not, do not repeatedly tune the public LB. Escalate to a PointRend/high-res refiner experiment.

This experiment is the best risk-adjusted use of the project’s current assets because it attacks three measured leaks without exposing the proven 0.360 weights to training drift.

---

## 12. Priority roadmap

### P0 — immediately, no retraining

1. Expose raw float mask logits/probabilities from the 0.360 champion.
2. Re-run exact grouped holdout PQ with mask-threshold, conf, NMS, min-area, and radial sweeps.
3. Replace global greedy carve with soft per-pixel ownership.
4. Add conditional one-piece cleanup consistent with MAGFiLO annotation rules.
5. Implement soft TTA through aligned probability-map fusion.
6. Learn an instance utility/calibration score based on probability of IoU>0.5 and expected matched IoU.

### P1 — small targeted training

7. Build a PointRend-style or lightweight edge/topology refiner on padded YOLO crops.
8. Train it using BCE/IoU-oriented loss plus a topology/boundary term such as soft-clDice; evaluate exact instance PQ, not semantic Dice alone.
9. Experiment with soft multi-rater consensus in the refiner, where probabilistic targets are natural.

### P2 — controlled core-model experiments

10. Test a stride-2 prototype head only after profiling memory and after P0/P1 saturate.
11. Compare short low-LR champion adaptation versus a clean pretrained restart on the same grouped validation split, both with `mosaic=0`.

### P3 — architecture replacement

12. Only then A/B CondInst/Mask2Former or another instance-native system at sufficient resolution. Require a direct local PQ win before migration.

---

## 13. Answers to the key Section 3 / Section 4 research questions

### Is the multi-annotator 0.33–0.37 ceiling fundamental?

**No proof supports calling it fundamental.** The project’s 0.3329 human-human PQ is a valuable empirical noise estimate. Public competition documentation does not currently specify how multiple hidden annotators are aggregated. Consensus/soft-label learning can exceed pairwise human agreement in other multi-rater segmentation tasks. Test random-rater and soft-consensus strategies locally.

### Can `mask_ratio=1` solve the prototype bottleneck?

**No, not in stock Ultralytics 8.3.145.** The loss downsamples GT to the prototype dimensions. You must increase the prototype/head resolution or attach a high-res refiner.

### Can `retina_masks=True` solve it?

**No, not fundamentally.** It changes the order/native rendering of bilinear scaling and cropping; it does not create a higher-resolution learned prototype basis. Still worth an OOF A/B because boundary rasterization can affect thin masks.

### Should YOLO-seg be abandoned?

**No.** Keep it as the proposal engine until the champion has been re-rendered/recalibrated with soft masks and a better overlap resolver. Add a high-res boundary refiner before a wholesale architecture change.

### Can soft TTA work?

**Yes.** The raw assembled logits exist immediately before Ultralytics binarizes the masks. Reconstruct each view’s map, invert the transform, match instances, fuse probabilities/logits, then threshold once.

### Can coefficients be averaged across TTA views?

**Do not do that.** Prototype bases are view-specific. Fuse reconstructed aligned masks, not coefficient vectors.

### What is the mathematically optimal confidence threshold?

There is no universal scalar threshold. Under a local independent-candidate approximation, admit a candidate when `P(match) × E[IoU|match] > PQ/2`. At PQ 0.36 the expected IoU contribution threshold is ~0.18. Learn that quantity from grouped OOF candidate features, then validate exact PQ.

### Is largest-connected-component cleanup correct?

**Useful but secondary.** First replace global confidence carving with soft per-pixel ownership. Then drop only clearly insignificant secondary components. The MAGFiLO protocol explicitly favors one-piece, hole-free masks.

### Is mosaic permanently banned?

**Yes, reasonably.** The physical transformation is wrong for intact solar disks and the project’s full-data run regressed badly with it. The evidence is not a perfectly controlled one-variable experiment, but there is no rational reason to spend scarce GPU budget retesting ordinary mosaic now.

### Did 110 epochs cause the regression?

**Not proven.** The model drift is real, but mosaic, dataset composition, batch and schedule changed simultaneously. Use grouped PQ checkpoint selection and controlled training branches.

### What is the one-shot highest-impact next action?

**Re-render and recalibrate the existing 0.360 champion using raw logits, soft overlap ownership, connectedness cleanup, soft TTA, and metric-aligned instance utility — all validated on its held-out physical-image fold.** Do this before retraining.

---

## 14. What would change the strategic verdict?

The recommendation would change only if one of these occurs:

1. The organizer publicly specifies that hidden evaluation uses a consensus construction that materially changes the ideal target policy.
2. Champion OOF experiments show post-processing/TTA improvements are negligible and errors are overwhelmingly mask-resolution-limited.
3. A controlled high-resolution alternative (PointRend/refiner/CondInst/Mask2Former) produces a reproducible PQ improvement on the same grouped split.
4. A clean no-mosaic retraining study demonstrates that the 0.360 checkpoint itself is overfit and a fresh model has clearly better local PQ.

Until then, the evidence favors **exploiting the champion’s soft information and fixing metric alignment**, not resetting the project around a new backbone.

---

# Sources

## Project sources supplied in this conversation

1. `MASTER_PROMPT_SEPT13(1).md` — Section 3/4 research questions and proposed failure modes.
2. `10_RED_TEAM_FORENSIC_AUDIT(1).md` — human-human PQ, fill-ratio audit, prototype resampling experiment, champion-vs-full-data cross-match, fragmentation, and ensemble FP analysis.
3. `02_DATA_AUDIT(1).md` — 8,199 annotation size distribution and physical-image structure.
4. `03_MODEL_HISTORY(1).md` — champion/full-data training configurations and verified LB trajectory.
5. `05_ANTIGRAVITY_REPORT(1).md` — full-data confidence sweep and Kaggle scores.
6. `06_TEST_RESULTS(1).md` — host-PQ compatibility and sanitizer/RLE tests.
7. `08_DECISION_LOG(1).md` — architectural decisions and current bans.
8. `DEVIN_COMPETITIVE_ANALYSIS(1).md` and `DEVIN_DEEP_RESEARCH_REPORT(1).md` — prior literature/architecture research; treated as secondary leads and independently checked where material.

## External primary / authoritative sources

9. Kaggle, *Solar Filament Segmentation Challenge 2026* overview/evaluation/announcements: https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements
10. Kaggle, competition data page (multi-annotator records): https://www.kaggle.com/competitions/filament-segmentation-2026/data
11. Kaggle discussion index containing the unanswered test-ground-truth construction question: https://www.kaggle.com/competitions/filament-segmentation-2026/discussion
12. Kirillov et al., *Panoptic Segmentation*, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html
13. Ahmadzadeh et al., *A dataset of manually annotated filaments from H-alpha observations*, Scientific Data 2024: https://www.nature.com/articles/s41597-024-03876-y
14. Ultralytics 8.3.145 `Proto` implementation: https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.145/ultralytics/nn/modules/block.py
15. Ultralytics 8.3.145 segmentation head: https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.145/ultralytics/nn/modules/head.py
16. Ultralytics 8.3.145 segmentation loss: https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.145/ultralytics/utils/loss.py
17. Ultralytics 8.3.145 mask processing: https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.145/ultralytics/utils/ops.py
18. Ultralytics 8.3.145 segmentation predictor: https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.145/ultralytics/models/yolo/segment/predict.py
19. Ultralytics 8.3.145 default segmentation configuration: https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.145/ultralytics/cfg/default.yaml
20. Bolya et al., *YOLACT: Real-Time Instance Segmentation*, ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html
21. Kirillov et al., *PointRend: Image Segmentation as Rendering*, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html
22. Warfield et al., *STAPLE*, IEEE TMI 2004 / PubMed: https://pubmed.ncbi.nlm.nih.gov/15250643/
23. Silva & Oliveira, *Using Soft Labels to Model Uncertainty in Medical Image Segmentation*: https://arxiv.org/abs/2109.12622
24. Solomon et al., *EdgeAttNet: Towards Barb-Aware Filament Segmentation*: https://arxiv.org/abs/2509.02964
25. Shit et al., *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html
26. Berman et al., *The Lovász-Softmax Loss*, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html
27. Tian et al., *Conditional Convolutions for Instance Segmentation (CondInst)*, ECCV 2020: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1105_ECCV_2020_paper.php
28. Cheng et al., *Mask2Former*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html
29. Kaggle, Sartorius 1st-place solution (probability-mask averaging): https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/discussion/298869

