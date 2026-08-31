# Solar Filament Segmentation Challenge 2026 — Competition Brief

> compiled from the Kaggle competition page, the IEEE BigData Cup 2026 proposal, and related public sources

---

## 1. What is this competition?

The **Solar Filament Segmentation Challenge 2026** is a Kaggle community prediction competition run by the **Earth-Space AI Research (ESAIR) Lab**. It is also one of the IEEE BigData Cup 2026 tracks.

**Goal:** Build an algorithm that can automatically draw pixel-precise segmentation masks around every solar filament in a full-disk H-alpha image.

- **Kaggle URL:** https://www.kaggle.com/competitions/filament-segmentation-2026
- **IEEE BigData Cup page:** https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/
- **Competition host:** ESAIR Lab
- **Sponsors:** U.S. National Science Foundation (NSF) and the National Solar Observatory (NSO)

---

## 2. Why solar filaments matter

Solar filaments are dense clouds of cooler plasma suspended above the Sun's surface by magnetic field lines. They are important because they are directly tied to space-weather events:

- **Coronal Mass Ejections (CMEs)**
- **Solar flares**
- **Solar Energetic Particle (SEP) storms**

An Earth-directed CME can damage power grids, disrupt GPS, create radiation hazards for flights near the poles, and be dangerous to astronauts. Because of this, tracking filaments is a key task in space-weather research.

---

## 3. The data

The competition uses the **MAGFiLO** dataset (Manually Annotated GONG Filaments from H-Alpha Observations). MAGFiLO is described in a 2024 *Scientific Data* paper (doi: 10.1038/s41597-024-03876-y).

Key facts about MAGFiLO:

- **10,244 annotated filaments** from **1,593 H-alpha observations** taken by the **Global Oscillation Network Group (GONG)** between 2011 and 2022.
- Each image is a **full-disk H-alpha observation** of the Sun.
- Competition images are fixed at **2048 x 2048 pixels**.
- Each annotation has:
  - a **segmentation polygon** (the mask we need to predict)
  - a **minimum bounding box**
  - a **filament spine** (a polyline)
  - a **chirality label** (left, right, or unidentifiable)
- Annotations are stored in **COCO-style JSON** format with polygons instead of binary masks to save space.
- The public MAGFiLO v1.0 dataset on Kaggle Datasets has 704 JPEG images and 8,212 annotated filaments: https://www.kaggle.com/datasets/esairlab/magfilo-v1-0-segmentation-of-solar-filaments

> Important: the competition says participants may use external data sources, but for training the model may **not** use any other ground-truth metadata. Only the competition's own H-alpha images and their ground-truth masks should be used for supervised training. Using the full public MAGFiLO ground truth to leak test-set labels would violate the spirit of rule (2).

---

## 4. The task in plain words

For every test image, your model has to:

1. Look at the 2048 x 2048 H-alpha image.
2. Find every filament (one image can contain many filaments).
3. Produce a binary mask for **each separate filament**.
4. Encode every mask as an **RLE count string**.
5. Write one row per predicted filament to a CSV file.

The number of predicted filaments does **not** have to match the number of ground-truth filaments. The evaluation matches predictions to ground truth by actual overlap, not by index.

---

## 5. Submission format

You must upload a single CSV. Each row is one predicted filament.

```text
filament_id,segmentation_rle
20150125172714Mh_1,f8uSDds...VQNC
20150125172714Mh_2,KHT%$HD...9>km
20170501024112Bh_1,HBy4d6D...97*D
```

Rules:

- `filament_id` = `image_id + "_" + a unique tail` (e.g. `_1`, `_2`, `_3`). The tail is only there to make rows unique; the `image_id` part must stay unchanged.
- `segmentation_rle` = **RLE counts only** (no RLE size). The size is fixed at 2048 x 2048.
- Do not wrap the RLE string in quotes.
- Use `pycocotools` (`annToMask`, `encode`, `decode`) to convert masks to the required RLE count format.

Reference: the Kaggle overview explicitly points to `pycocotools.mask.encode` and the COCO API.

---

## 6. Evaluation

The competition uses a mix of **quantitative** and **qualitative** criteria.

### Quantitative (70%)

- **Mean Dice score** across test images (`torchmetrics.segmentation.DiceScore`)
- **Panoptic Quality (PQ)**
- Distribution of Dice scores
- Distribution of IoU scores
- Distribution of **one-to-many** and **many-to-one** relations between predicted and ground-truth masks

### Qualitative (30%)

- Detailed description of the full pipeline (preprocessing → model → post-processing)
- Visual quality of predicted masks on H-alpha images
- Quality of code (modularity and documentation)

### Panoptic Quality formula

PQ is defined as:

```text
PQ = sum_of_IoU_of_true_positives / (|TP| + 0.5 * |FP| + 0.5 * |FN|)
```

Where:

- A predicted segment and a ground-truth segment are matched if their **IoU > 0.5**.
- **TP** = true positives (one-to-one matches)
- **FP** = predicted segments that did not match any ground-truth segment
- **FN** = ground-truth segments that were not matched
- One-to-many and many-to-one cases are penalized.

The original PQ paper is Kirillov et al., "Panoptic Segmentation", CVPR 2019 (doi: 10.1109/CVPR.2019.00963).

---

## 7. Important dates

| Date | Event |
|------|-------|
| 10 July 2026 | Competition launched |
| 15 Nov 2026 | Final report + code submission deadline |
| 30 Nov 2026 | Winners announced |
| 14-17 Dec 2026 | IEEE BigData 2026 conference, Phoenix, AZ |

---

## 8. Prizes and final submission

- Total prize pool: **up to $3,000** distributed among 1st, 2nd, 3rd place.
- Winners may be announced at IEEE BigData 2026.
- Finalists must submit:
  1. A **4-page technical report** (PDF) using the provided Overleaf template.
  2. A **public Git repository** with:
     - `requirements.txt`
     - a Jupyter notebook showing the whole pipeline
     - source code that can reproduce the submitted predictions
- Prize winners must provide banking info and a license to use their code for research/educational purposes.

---

## 9. Open-access and code policy

- The repository must be public by the competition close and stay public until winners are announced.
- Code quality (modularity, reproducibility, documentation) is part of the score.
- Submissions must be reproducible without extra files.

---

## 10. Main challenges

The overview highlights three hard parts:

1. **Fine-scale structures (barbs):** thin, thread-like features that stick out of a filament. Their direction tells us about the magnetic chirality.
2. **Background noise and image quality:** ground-based images have noise, atmospheric artifacts, and low contrast, so dark regions are not always filaments.
3. **Structural continuity:** models often split one filament into many small fragments or merge nearby filaments into one blob. The competition penalizes both fragmentation and over-merging.

---

## 11. Organizers and citation

**Organizers:**

- Azim Ahmadzadeh, Ph.D., University of Missouri–St. Louis
- Dustin J. Kempton, Ph.D., Georgia State University
- Qin Li, Ph.D., New Jersey Institute of Technology
- Alexei A. Pevtsov, Ph.D., National Solar Observatory

**Citation:**

Azim Ahmadzadeh, Dustin J. Kempton, Qin Li, and Alexei A. Pevtsov. Solar Filament Segmentation Challenge 2026. https://kaggle.com/competitions/filament-segmentation-2026, 2026. Kaggle.

---

## 12. Quick links for the next steps

- Competition page: https://www.kaggle.com/competitions/filament-segmentation-2026
- Self-evaluation notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- MAGFiLO dataset: https://www.kaggle.com/datasets/esairlab/magfilo-v1-0-segmentation-of-solar-filaments
- MAGFiLO website: https://www.mlecofi.net/magfilo
- EdgeAttNet paper: https://arxiv.org/abs/2509.02964
- Flat U-Net paper: https://iopscience.iop.org/article/10.3847/1538-4357/adadff
- PQ paper: https://arxiv.org/abs/1801.00868
