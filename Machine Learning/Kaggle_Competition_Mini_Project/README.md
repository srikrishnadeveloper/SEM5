# Solar Filament Segmentation Challenge 2026

End-to-End Deep Learning Pipeline for the **Solar Filament Segmentation Challenge 2026** (Kaggle & IEEE BigData Cup).

Predict pixel-precise binary instance masks for solar filaments in $2048 \times 2048$ full-disk H-alpha solar imagery from the MAGFiLO v1.0 dataset.

---

## 🌟 Key Features

- **Multi-Architecture Support**: U-Net, U-Net++ (with scSE attention), DeepLabV3+, FPN, and EdgeAttNet (`UNetEdgeTransformer` with MHSA bottleneck).
- **Competition-Aligned Evaluation**: Validation and threshold search optimized directly against the true **Multiscale $m\text{IoU}$** instance metric (10-scale Sobel edge box-counting with trapezoidal integration and greedy 1-to-1 matching).
- **Robust Pipeline**:
  - Memory-mapped training cache for rapid iteration.
  - Filament-centric cropping & heavy Albumentations augmentation (Copy-Paste, CLAHE, Elastic Transform, Grid Distortion).
  - Test-Time Augmentation (D4 dihedral flip/rotations).
  - Out-of-fold probability caching & automated threshold / minimum area search.
  - Multi-fold probability ensembling & watershed instance splitting.
  - COCO RLE encoding.
- **Google Colab & Kaggle Ready**: Self-contained verified notebooks with hardware diagnostics, interactive Kaggle token fallback, and live step-by-step progress logging.

---

## 📁 Repository Structure

```
.
├── code/                                # Modular pipeline source files
│   ├── config.py                        # Central hyperparameter & env configurations
│   ├── dataset.py                       # COCO dataset loader & GroupKFold by year
│   ├── infer.py                         # Full-resolution TTA inference & postprocessing
│   ├── losses.py                        # Combined Dice + Focal + BCE loss
│   ├── metrics.py                       # Pairwise and Multiscale mIoU implementation
│   ├── model.py                         # Model builder (smp + EdgeAttNet)
│   ├── postprocess.py                   # Solar disk masking, watershed & RLE encoding
│   ├── submit.py                        # OOF and test submission generator
│   └── train.py                         # Single-fold training loop
├── notebooks/                           # Standalone Colab & Kaggle notebooks
│   ├── colab_filament_top50_verified.ipynb # Single verified Colab notebook with live logs
│   ├── filament_top50_single.py         # Complete end-to-end single-file engine
│   ├── build_single_notebook.py         # Single notebook generator
│   └── colab_payloads/                  # Multi-account payload generator
├── research/                            # Technical notes & top-50 battle plans
│   ├── TOP50_PLAN.md                    # Literature review & architecture guide
│   └── TOP50_ACTION_PLAN.md             # Code audit & phase-by-phase implementation
├── tests/                               # Verification test suite
│   ├── test_unit_fixes.py               # Pure-function unit test suite (<1s)
│   └── smoke_synthetic.py               # End-to-end synthetic dataset smoke test
├── requirements.txt
├── AGENTS.md                            # Competition specifications & runbook
└── WORKLOG.md                           # Session log & experiment tracking
```

---

## ⚡ Quickstart

### 1. Running on Google Colab (Recommended)
1. Open [`notebooks/colab_filament_top50_verified.ipynb`](notebooks/colab_filament_top50_verified.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → GPU (T4)**.
3. Provide your Kaggle API token (via Colab Secrets `KAGGLE_API_TOKEN` or interactive upload in Cell 3).
4. Select your preset in Cell 5 (`fast`, `balanced`, `max`, or `edgeattnet`) and click **Runtime → Run all**.

### 2. Running Locally

```bash
# Clone repository
git clone https://github.com/srikrishnadeveloper/SolarFilamentSegmentationChallenge-2026.git
cd SolarFilamentSegmentationChallenge-2026

# Install dependencies
pip install -r requirements.txt

# Run unit and smoke tests
python tests/test_unit_fixes.py
python tests/smoke_synthetic.py

# Train Fold 0
cd code
python train.py
```

---

## 🎯 Speed & Quality Presets

| Preset | Model Architecture | Resolution | Batch / Accum | Est. Time (T4) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fast`** | U-Net (`timm-efficientnet-b2`) | $768 \times 768$ | Batch 4 / Accum 2 | ~35 min/fold | Fast, lightweight baseline with D4 TTA. |
| **`balanced`** | U-Net (`timm-efficientnet-b4`) | $1024 \times 1024$ | Batch 2 / Accum 4 | ~70 min/fold | **Recommended:** High accuracy and edge fidelity. |
| **`max`** | U-Net++ (`timm-efficientnet-b4` + scSE) | $1024 \times 1024$ | Batch 1 / Accum 8 | ~3 hours/fold | Maximum capacity for fine boundary capture. |
| **`edgeattnet`** | `UNetEdgeTransformer` | $512 \times 512$ | Batch 4 / Accum 4 | ~2 hours/fold | Specialized solar edge-attention architecture. |

---

## 📜 License
MIT License.
