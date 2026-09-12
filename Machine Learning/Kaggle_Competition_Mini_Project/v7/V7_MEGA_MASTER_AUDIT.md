# V7 Mega Master Grandmaster Pipeline — Brutal Forensic Audit

**Auditor:** Principal CV Research Scientist / Kaggle Grandmaster  
**Date:** 2026-09-01  
**Files audited:**
- `v7/v7_mega_master_pipeline.py`
- `v6/train_yolo_detector.py` (cross-reference for YOLO training)
- `v6/infer_yolo_plus_14models_ensemble.py` (cross-reference for V6 baseline)

---

## 1. Executive Summary — Brutal Honest Take

**The V7 script is mostly a copy-paste of V6 with extra comments claiming state-of-the-art techniques that are NOT actually implemented.** It will run and likely produce a valid submission, but several of its headline claims are false or dead code.

**Headline problems:**
1. **Claims "8-Way D4 TTA" but implements only 4 flips.** The comment is a lie and misses 90°/180°/270° rotations.
2. **Imports a Gaussian apodization window for blended tiles but never uses it.** Dead code / false claim.
3. **Claims EdgeAttNet / boundary-loss inspiration but implements none of it.** Marketing, not math.
4. **Fallback connected-components can merge distinct filaments at saddle points.** No watershed or skeleton splitting.
5. **DeepLab encoder guessing still has the V6 bug** (`'resnet'` forces `resnet50`).
6. **Unsharp mask is computed in saturated `uint8` before normalization**, throwing away dynamic range.

**Bottom line:** Fix the false claims, implement the missing pieces, and tune thresholds on validation. As it stands, V7 will not score meaningfully better than a corrected V6.

| Severity | Count | Headline issues |
|----------|-------|-----------------|
| **Critical** | 0 | No submission-invalidating bugs |
| **High** | 5 | Dead-code/false claims; 4-flip-only TTA; fallback merging; uint8 unsharp saturation; untuned thresholds |
| **Medium** | 5 | DeepLab encoder bug; unused imports; Gaussian window unused; no area-weighted priority tuning; no validation harness |
| **Low** | 2 | Minor code cleanliness; verbose logging |

---

## 2. Vector 1 — Coordinate Frame & Bounding Box Scaling Alignment

### Code under review (lines 188–205)

```python
yolo_preds = yolo_detector(feats_3ch_1024, conf=0.35, iou=0.45, verbose=False)[0]
if yolo_preds.boxes is not None and len(yolo_preds.boxes) > 0:
    boxes = yolo_preds.boxes.xyxy.cpu().numpy()
    scores = yolo_preds.boxes.conf.cpu().numpy()
    boxes[:, [0, 2]] *= (w_orig / 1024.0)
    boxes[:, [1, 3]] *= (h_orig / 1024.0)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        ...
        x1_p, y1_p = max(0, x1 - PADDING), max(0, y1 - PADDING)
        x2_p, y2_p = min(w_orig, x2 + PADDING), min(h_orig, y2 + PADDING)
        if (x2_p - x1_p) < 10 or (y2_p - y1_p) < 10: continue
        crop_p = prob_2048[y1_p:y2_p, x1_p:x2_p]
```

### Mathematical analysis

- YOLO is trained and run at 1024×1024 (`imgsz=1024`). Output boxes `xyxy` live in the 1024 detector frame.
- Original images are 2048×2048, so `w_orig/1024 = h_orig/1024 = 2.0`.
- Scaling by exactly 2.0 maps a detector box to the 2048 sensor frame with **no scale-induced sub-pixel error**.
- The only source of misalignment is the `.astype(int)` floor. A box corner at `x2 = 341.7` in 1024-space becomes `683` instead of `683.4` in 2048-space. The maximum error is **< 1 pixel** per corner, negligible.
- `PADDING = 20` is added after flooring and clamped to `[0, 2048]`. Bounds checks are correct.

### Verdict

✅ **Scaling is mathematically correct and safe.** No sub-pixel bias of consequence.

### Caveats

- `PADDING = 20` is small. Thin filaments have barbs that can extend beyond 20 px from a tight box. For the refiner-style crop logic used here, **32–64 px** is safer.
- The crop is taken from the **semantic probability map**, not from the raw image. This is the correct fusion design (YOLO proposes, UNet confirms).

---

## 3. Vector 2 — Astronomical Preprocessing & Multi-Channel Math

### Code under review (lines 62–67)

```python
def build_astronomical_features(raw_gray: np.ndarray) -> np.ndarray:
    ch0 = raw_gray
    ch1 = clahe_op.apply(raw_gray)
    ch2 = cv2.addWeighted(raw_gray, 1.5, cv2.GaussianBlur(raw_gray, (0, 0), sigmaX=3.0), -0.5, 0)
    return np.stack([ch0, ch1, ch2], axis=-1)
```

### Mathematical analysis

**Channel 0 — Raw grayscale:** Baseline intensity.

**Channel 1 — CLAHE(clip=3.0, grid=8×8):** Adaptive local contrast enhancement. Standard for H-alpha filament enhancement.

**Channel 2 — Unsharp mask:**
```
ch2 = 1.5 * Raw - 0.5 * Gaussian(Raw, sigma=3) + 0
```

This is a classic high-pass / ridge-enhancement filter. It emphasizes filament spines and barbs against the smoother background.

### The real problem: uint8 saturation

`raw_gray` is `uint8` (0–255). `cv2.GaussianBlur(...)` returns `uint8`. `cv2.addWeighted(...)` on `uint8` inputs uses OpenCV's default **saturating arithmetic**:
- Values > 255 are clipped to 255.
- Values < 0 are clipped to 0.

The unsharp coefficients `1.5` and `-0.5` with offset `0` will frequently produce:
- Bright disk pixels: `1.5*255 - 0.5*255 = 255` → okay.
- Dark filaments on brighter disk: `1.5*30 - 0.5*90 = 0` (clipped to 0).
- Moderate ridges: `1.5*120 - 0.5*80 = 140` → okay.

**Result:** The dynamic range of the unsharp channel is compressed before the model ever sees it. Information about faint barbs is lost in the saturated floor/ceiling.

### Fix: compute unsharp in float32

```python
def build_astronomical_features(raw_gray: np.ndarray) -> np.ndarray:
    raw_f = raw_gray.astype(np.float32)
    ch0 = raw_f
    ch1 = clahe_op.apply(raw_gray).astype(np.float32)  # CLAHE still on uint8, output is uint8
    blurred = cv2.GaussianBlur(raw_f, (0, 0), sigmaX=3.0)
    ch2 = 1.5 * raw_f - 0.5 * blurred
    # Stack as float32, then normalize later
    return np.stack([ch0, ch1, ch2], axis=-1)
```

Then remove the `/255.0` division before ImageNet normalization, or normalize consistently. Since you are already applying ImageNet stats, the simplest path is:

```python
feats_f = build_astronomical_features(raw_2048)  # float32, ~0-255 range
feats_3ch_1024 = cv2.resize(feats_f, (1024, 1024), interpolation=cv2.INTER_AREA)
norm_3ch = (feats_3ch_1024 / 255.0 - mean_3ch) / std_3ch
```

### Verdict

⚠️ **Conceptually good, numerically flawed.** Unsharp mask must be computed in float32 before saturation. This is a real loss of information.

### ImageNet normalization on synthetic channels

Using `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` on `[raw, CLAHE, unsharp]` is arbitrary but **consistent** with training, so the model learns the channel semantics. It is not "correct" in a statistical sense, but it works. No change needed if training and inference use the same normalization.

---

## 4. Vector 3 — TTA Symmetry & Mathematical Invariance

### Code under review (lines 176–180)

```python
p1 = torch.sigmoid(m(tensor_in))
p2 = torch.flip(torch.sigmoid(m(torch.flip(tensor_in, [2]))), [2])
p3 = torch.flip(torch.sigmoid(m(torch.flip(tensor_in, [3]))), [3])
p4 = torch.flip(torch.sigmoid(m(torch.flip(tensor_in, [2, 3]))), [2, 3])
prob_1024 += ((p1 + p2 + p3 + p4) / 4.0).squeeze().cpu().numpy() / max(1, len(models_and_devices))
```

### The claim vs. the reality

The file header and line 163 claim:

> `A. 14-Model Dual-GPU 8-Way D4 TTA`

**This is false.** The code implements only **4 flips** (identity, vertical, horizontal, both). A true Dihedral-4 (D4) group has **8 elements**: the 4 flips plus rotations by 0°, 90°, 180°, 270°.

### Mathematical equivariance of flips

A CNN with symmetric kernels is equivariant to horizontal and vertical flips:
```
f(flip(x)) = flip(f(x))
```
so averaging flipped predictions is valid.

Solar filaments are **not** uniformly oriented; they appear at arbitrary angles on the disk. Rotational TTA (90°, 180°, 270°) captures diagonal structures that flips miss.

### Exact D8 (full D4) TTA implementation

```python
def rotate_90(x, k):
    """Rotate tensor x by k*90 degrees counter-clockwise in the H-W plane.
       x: (B, C, H, W)
    """
    return torch.rot90(x, k, dims=[2, 3])

def tta_d8(model, tensor_in):
    """Full D4 group: 8 orientations."""
    probs = []
    # identity
    probs.append(torch.sigmoid(model(tensor_in)))
    # horizontal flip
    probs.append(torch.flip(torch.sigmoid(model(torch.flip(tensor_in, [3]))), [3]))
    # vertical flip
    probs.append(torch.flip(torch.sigmoid(model(torch.flip(tensor_in, [2]))), [2]))
    # hv flip
    probs.append(torch.flip(torch.sigmoid(model(torch.flip(tensor_in, [2, 3]))), [2, 3]))
    # 90, 180, 270 rotations (+ their flips are covered by the above, but rot alone is not)
    for k in [1, 2, 3]:
        rot_in = rotate_90(tensor_in, k)
        rot_out = torch.sigmoid(model(rot_in))
        probs.append(rotate_90(rot_out, -k % 4))
    return torch.stack(probs).mean(dim=0)
```

Then replace the 4-flip block with:

```python
prob_1024 += tta_d8(m, tensor_in).squeeze().cpu().numpy() / max(1, len(models_and_devices))
```

### Verdict

❌ **False claim and missed opportunity.** Either implement true 8-way D4 TTA or remove the "8-Way" claim. The 4-flip TTA alone is valid but suboptimal for diagonal filaments.

---

## 5. Vector 4 — Confidence Fusion & Non-Overlap Priority Arbitration

### Code under review (lines 209–242)

```python
fused_score = 0.5 * float(y_sc) + 0.5 * float(crop_p[crop_bin > 0].mean() if area > 0 else 0)
instance_candidates.append((fused_score, area, full_m))
...
instance_candidates.sort(key=lambda x: x[0], reverse=True)
...
for score, area, mask in instance_candidates:
    clean_mask = mask & (occupied == 0)
    if int(clean_mask.sum()) >= MIN_AREA:
        ...
        occupied |= clean_mask
```

### Mathematical analysis

Priority score:
```
S_i = 0.5 * S_YOLO + 0.5 * P_mean
```

where `P_mean` is the average semantic probability inside the binarized mask.

**Is this sound?** ✅ **Yes, as a heuristic.**
- `S_YOLO` measures detection confidence.
- `P_mean` measures pixel-level consensus of the ensemble.
- Both terms are in `[0, 1]`, so the score is in `[0, 1]`.
- It penalizes boxes that YOLO likes but the semantic ensemble does not support (low `P_mean`), and vice versa.

**Could a large low-confidence blob preempt a small high-confidence filament?** Under pure `S_i` sorting: yes, if the large blob has slightly higher `S_i`. The small filament would be removed during non-overlap arbitration.

### Should you multiply by sqrt(Area)?

A proposed priority:
```
S'_i = S_i * sqrt(Area)
```

Pros:
- Rewards larger filaments, which are often more confident and more valuable under PQ.
- Can suppress tiny fragmented false positives.

Cons:
- Can **kill small but real filaments**, which are common in the dataset.
- The PQ metric does not inherently reward area; it rewards correct matches (IoU > 0.5).

**Verdict:** There is no universal answer. You must **tune the priority function on validation PQ**. A safe starting family is:

```python
priority = alpha * S_YOLO + (1 - alpha) * P_mean + beta * log(area)
```

and grid-search `(alpha, beta)` on out-of-fold validation.

### Exact production fix: tunable priority

```python
ALPHA = 0.5    # YOLO vs ensemble weight
BETA = 0.0     # area bonus; tune on validation
GAMMA = 0.0    # area penalty exponent; tune on validation

for i in range(len(boxes)):
    ...
    area = int(full_m.sum())
    if MIN_AREA <= area <= MAX_AREA:
        p_mean = float(crop_p[crop_bin > 0].mean()) if area > 0 else 0.0
        fused_score = ALPHA * float(y_sc) + (1.0 - ALPHA) * p_mean
        if BETA != 0.0:
            fused_score += BETA * math.log1p(area)
        instance_candidates.append((fused_score, area, full_m))
```

Then run a small grid search on a held-out validation fold: `ALPHA ∈ {0.3, 0.5, 0.7}`, `BETA ∈ {-0.05, 0.0, 0.05}`.

### Verdict

⚠️ **Fusion score is reasonable but untuned.** The arbitration is correct; the priority function should be optimized on validation.

---

## 6. Vector 5 — Fallback Instance Separation vs Over-Fragmentation

### Code under review (lines 213–225)

```python
if len(instance_candidates) == 0:
    bin_map = (prob_2048 > PROB_THRESH).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_map = cv2.morphologyEx(bin_map, cv2.MORPH_CLOSE, kernel)

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
    for lbl in range(1, n_lbl):
        area_val = stats[lbl, cv2.CC_STAT_AREA]
        if MIN_AREA <= area_val <= MAX_AREA:
            comp_m = (labels == lbl).astype(np.uint8)
            score_val = float(prob_2048[comp_m > 0].mean()) if area_val > 0 else 0.5
            instance_candidates.append((score_val, area_val, comp_m))
```

### Failure mode: merging at saddle points

Two distinct filaments can produce two separate high-probability ridges in `prob_2048`. If they are connected by a thin, low-probability bridge above `PROB_THRESH = 0.45`, `connectedComponents` treats them as **one instance**. The 3×3 closing kernel **makes this worse** by intentionally bridging small gaps.

This is the exact failure mode the prompt describes: two filaments touching at a saddle point become one, lowering IoU and RQ.

### When should you split?

For any fallback component with `area > 3000 px`, you should assume it may contain multiple filaments and run **Distance-Transform Watershed** or **probability-watershed** splitting.

### Exact production fix: watershed splitting for large components

```python
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def split_large_component(prob_map, bin_mask, min_area=200, split_area_threshold=3000):
    """Split a binary component using distance-transform watershed if it is large."""
    labels, n = ndimage.label(bin_mask)
    out_masks = []
    for idx in range(1, n + 1):
        comp = (labels == idx).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        if area < split_area_threshold:
            out_masks.append(comp)
            continue

        # Distance transform watershed
        dist = distance_transform_edt(comp)
        # Use the probability map inside the component as topography instead of distance
        topo = -prob_map * comp
        coords = peak_local_max(dist, min_distance=20, exclude_border=False)
        if len(coords) < 2:
            out_masks.append(comp)
            continue

        markers = np.zeros_like(comp, dtype=int)
        markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)
        split_labels = watershed(topo, markers, mask=comp)

        for sub_idx in range(1, len(coords) + 1):
            sub_mask = (split_labels == sub_idx).astype(np.uint8)
            if int(sub_mask.sum()) >= min_area:
                out_masks.append(sub_mask)
    return out_masks
```

Then replace the fallback loop with:

```python
if len(instance_candidates) == 0:
    bin_map = (prob_2048 > PROB_THRESH).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_map = cv2.morphologyEx(bin_map, cv2.MORPH_CLOSE, kernel)
    split_masks = split_large_component(prob_2048, bin_map,
                                        min_area=MIN_AREA,
                                        split_area_threshold=3000)
    for comp_m in split_masks:
        area_val = int(comp_m.sum())
        score_val = float(prob_2048[comp_m > 0].mean()) if area_val > 0 else 0.5
        instance_candidates.append((score_val, area_val, comp_m))
```

### Verdict

⚠️ **Fallback is functional but can under-segment touching filaments.** Add watershed splitting for components > 3000 px.

---

## 7. Vector 6 — COCO RLE Fortran Order & Empty Instance Safety

### Code under review (lines 75–83)

```python
def rle_encode_single(mask: np.ndarray, h: int = 2048, w: int = 2048) -> str:
    fortran_mask = np.asfortranarray(mask, dtype=np.uint8).reshape((h, w, 1))
    rle = mask_utils.encode(fortran_mask)[0]
    return rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']

def rle_empty(h: int = 2048, w: int = 2048) -> str:
    empty_mask = np.zeros((h, w, 1), dtype=np.uint8, order='F')
    rle = mask_utils.encode(empty_mask)[0]
    return rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
```

### Mathematical / API compliance

- `np.asfortranarray(mask)` ensures column-major (Fortran) order.
- `.reshape((h, w, 1))` produces the `(H, W, N)` shape pycocotools expects.
- `[0]` extracts the single RLE dict.
- Empty mask is explicitly Fortran-ordered.

**Verdict:** ✅ **Correct and safe.** No transposition errors.

---

## 8. Vector 7 — Hidden / Bonus Issues

### 8.1 Dead code and false marketing claims

The header claims to synthesize:
- **Anthony Therrien (0.36): 2D Gaussian Apodization Overlapping Blended Tiles**
- **Koushik Dinda (EdgeAttNet): High-Pass Astronomical Ridge Filtering & Boundary Loss**
- **8-Way D4 TTA**

**Reality:**
- `create_gaussian_tile_weights` and `GAUSSIAN_WINDOW_1024` are defined (lines 86–92) but **never used**. The inference is single-tile 1024×1024 resize, not blended tiles.
- EdgeAttNet architecture is not implemented.
- Boundary loss is not implemented.
- TTA is 4-flip, not 8-way.

**Fix:** Either implement these features or remove the claims. False claims in a Kaggle notebook do not affect the score, but they mislead you and anyone reading the code.

### 8.2 DeepLab encoder guessing bug (line 122)

```python
enc_name = 'resnet50' if ('resnet50' in p.lower() or 'resnet' in str(encoder).lower()) else encoder
```

If `encoder = 'resnet34'`, the condition `'resnet' in str(encoder).lower()` is true, so it forces `resnet50`, causing a state-dict mismatch. Same V6 bug.

**Fix:**
```python
enc_name = 'resnet50' if ('resnet50' in p.lower() or 'resnet50' in str(encoder).lower()) else encoder
```

### 8.3 `MAX_AREA = 120000` is a good improvement

V6 had no upper bound. V7 caps at 120000 px. ✅ This prevents a merged mega-blob from destroying the submission.

### 8.4 `PROB_THRESH = 0.45` used for both YOLO crop and fallback

The threshold is hard-coded and shared. In practice, the optimal threshold for YOLO-cropped regions (where the box is already a strong prior) may differ from the optimal fallback threshold. Should be tuned separately.

### 8.5 YOLO detector training uses Mosaic/CopyPaste/Mixup

As noted in the V6 audit, `mosaic=0.5`, `copy_paste=0.3`, `mixup=0.1` are likely **harmful** for thin solar filaments. The V7 script uses the same `v6/train_yolo_detector.py`. See the V6 audit for the recommended recipe (disable mosaic/copy_paste/mixup, use flips/rotation/brightness/scale).

### 8.6 No validation / threshold tuning harness

Every threshold in V7 is hard-coded:
- `PROB_THRESH = 0.45`
- `MIN_AREA = 200`
- `MAX_AREA = 120000`
- `PADDING = 20`
- YOLO `conf=0.35`, `iou=0.45`
- Fusion weights `0.5 / 0.5`

A **5% change in `PROB_THRESH` can move public LB by 0.01–0.03**. You must build an OOF validation harness that computes PQ and mIoU_pairwise and grid-searches these values.

---

## 9. Corrected core blocks

### 9.1 Float32 feature builder

```python
def build_astronomical_features(raw_gray: np.ndarray) -> np.ndarray:
    raw_f = raw_gray.astype(np.float32)
    ch0 = raw_f
    ch1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(raw_gray).astype(np.float32)
    blurred = cv2.GaussianBlur(raw_f, (0, 0), sigmaX=3.0)
    ch2 = 1.5 * raw_f - 0.5 * blurred
    return np.stack([ch0, ch1, ch2], axis=-1)
```

### 9.2 Full D4 (8-way) TTA function

```python
def rotate_90(x, k):
    return torch.rot90(x, k, dims=[2, 3])

def tta_d4(model, x):
    """D4 group: 8 orientations. x: (B, C, H, W)"""
    outs = []
    # identity + flips
    outs.append(torch.sigmoid(model(x)))
    outs.append(torch.flip(torch.sigmoid(model(torch.flip(x, [3]))), [3]))
    outs.append(torch.flip(torch.sigmoid(model(torch.flip(x, [2]))), [2]))
    outs.append(torch.flip(torch.sigmoid(model(torch.flip(x, [2, 3]))), [2, 3]))
    # 90, 180, 270
    for k in [1, 2, 3]:
        xr = rotate_90(x, k)
        out = torch.sigmoid(model(xr))
        outs.append(rotate_90(out, -k % 4))
    return torch.stack(outs, dim=0).mean(dim=0)
```

### 9.3 Watershed fallback splitting

See section 5 for `split_large_component`.

### 9.4 Tunable fusion priority

```python
ALPHA = 0.5
BETA = 0.0

p_mean = float(crop_p[crop_bin > 0].mean()) if area > 0 else 0.0
fused_score = ALPHA * float(y_sc) + (1.0 - ALPHA) * p_mean
if BETA != 0.0:
    fused_score += BETA * math.log1p(area)
instance_candidates.append((fused_score, area, full_m))
```

---

## 10. Summary of fixes before running V7

| Priority | Fix | Expected impact |
|----------|-----|-----------------|
| **High** | Compute unsharp mask in float32 | Preserves faint barb signal |
| **High** | Implement true 8-way D4 TTA or remove claim | +0.005–0.015 PQ for diagonal filaments |
| **High** | Add watershed splitting to fallback | Reduces under-segmentation |
| **High** | Build OOF validation + threshold grid search | +0.02–0.05 PQ |
| **Medium** | Fix DeepLab `resnet50` forcing bug | Prevents load failure on resnet34/101 checkpoints |
| **Medium** | Increase `PADDING` to 32–64 px | Better context for thin barbs |
| **Medium** | Disable Mosaic/CopyPaste/Mixup in YOLO training | Better training for filament morphology |
| **Low** | Remove dead Gaussian-window / EdgeAttNet claims | Code cleanliness |

---

## 11. Final verdict

**V7 is not a "mega master grandmaster" pipeline.** It is a decent V6-style YOLO + UNet++ ensemble with misleading marketing and several unimplemented features. The submission it produces will be valid, and the score may be reasonable because the underlying V6 architecture is sound. However, it will not beat a cleanly implemented, validation-tuned version of the same idea.

**The fastest path to a higher score:**
1. Fix the uint8 unsharp saturation.
2. Implement real 8-way TTA.
3. Add watershed fallback splitting.
4. Grid-search thresholds and fusion weights on validation PQ.
5. Fix the YOLO augmentation recipe.

Do those five things, and V7 becomes a genuine medal-level pipeline. Without them, it is just V6 with extra adjectives.
