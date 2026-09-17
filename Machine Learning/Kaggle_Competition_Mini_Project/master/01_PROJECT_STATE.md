# 01_PROJECT_STATE.md — Current Codebase State & Environment
**Project:** Solar Filament Segmentation Challenge 2026  
**Status Date:** September 14, 2026  
**Timestamp:** `2026-09-14T18:42:00+05:30`  
**Branch:** `cascade_2_crop_zoom` (Cascade 2.0 High-Resolution Crop & Zoom Paradigm)  

---

## 1. Local Codebase Architecture

```
c:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\
├── master/                               # Permanent authoritative source documents
│   ├── 00_EXECUTIVE_SUMMARY.md
│   ├── 01_PROJECT_STATE.md
│   ├── 02_DATA_AUDIT.md
│   ├── 03_MODEL_HISTORY.md
│   ├── 04_CURRENT_DIRECTIVE.md
│   ├── 05_ANTIGRAVITY_REPORT.md
│   ├── 06_TEST_RESULTS.md
│   ├── 07_KAGGLE_RUNBOOK.md
│   ├── 08_DECISION_LOG.md
│   └── 09_ARTIFACT_HASHES.md
├── cascade/                              # Cascade 2.0 (Crop & Zoom) Core Engine
│   ├── geometry.py                       # Non-truncating adaptive crop geometry
│   ├── model_refiner.py                  # Dedicated U-Net++ patch segmenter & composite loss
│   ├── dataset.py                        # Physical crop dataset builder & augmentations
│   ├── train_refiner.py                  # Fast GPU refiner training script
│   └── infer_cascade.py                  # Dual-GPU high-throughput inference engine
├── moonshot_2048/                        # Native 2048 YOLO proposal modules (Stage 1)
│   ├── config.py                         # Environment variables and tunable thresholds
│   ├── data.py                           # COCO parser and YOLO polygon normalizer
│   ├── dataset_pytorch.py                # Pure PyTorch Dataset & DataLoader
│   ├── train_yolov8l.py                  # YOLOv8l-seg native 2048 training orchestrator
│   ├── train_yolo11l.py                  # YOLO11l-seg experimental trainer
│   ├── predict_native.py                 # Baseline inference module
│   ├── predict_pytorch_gpu.py            # Pure PyTorch GPU tensor inference engine
│   ├── match_and_calibrate.py            # Kirillov PQ multi-annotator evaluator
│   ├── ensemble_instances.py             # Instance clustering and greedy zero-overlap carver
│   ├── audit_submission.py               # Submission CSV integrity auditor
│   ├── gated_refiner.py                  # Gated boundary refinement module
│   └── build_fulldata_notebook.py        # Automated Kaggle notebook generator
├── v8_1/                                 # Legacy baseline anchor (0.350 LB)
│   ├── 1_train_yolo.py                   # Stage 1 YOLOv8 detector
│   ├── 2_train_crop_refiner.py           # Stage 2 U-Net++ crop refiner
│   ├── 3_infer_cascade.py                # 2-stage inference pipeline
│   ├── 4_sweep_inference.py              # Inference sweep module
│   ├── geometry.py                       # Adaptive crop bounding box utilities
│   └── config.py                         # Baseline cascade configurations
├── models/                               # Local model checkpoint storage
│   ├── best_crop_refiner.pth             # 39.8 MB U-Net++ Crop Refiner (0.8772 Val IoU)
│   ├── moonshot_2048/best.pt             # 92.8 MB native YOLOv8l-seg (0.360 LB Champion)
│   ├── moonshot_2048/best_fulldata.pt    # 92.8 MB full-data YOLOv8l-seg (0.340 LB)
│   ├── best_deeplabv3p_res50d_fold_0.pth # 49.4 MB DeepLabV3+
│   ├── best_segformer_mitb3_fold_0.pth   # 99.0 MB SegFormer MiT-B3
│   └── best_unetpp_effb4_fold_0.pth      # 64.9 MB U-Net++ EfficientNet-B4
├── submissions/                          # Production submission CSV storage
│   ├── submission_cascade_2_0.csv        # 1,309 filaments (Cascade 2.0 Results 9)
│   └── ...
├── notebooks/kaggle/                     # Production Kaggle notebooks
│   ├── Cascade_P1_CropZoom_DualGPU.ipynb # Ready-to-run 2-stage Kaggle notebook (v1.0.1)
│   └── moonshot_fulldata.ipynb           # Ready-to-run 100% full-data notebook
├── tests/                                # 60 unit and contract tests (98.4% passing)
└── data/MAGFiLO_1.0_Kaggle_2026/         # 48.6 MB COCO dataset + 707 physical images
```

---

## 2. Hardware & Execution Profiles

| Dimension | Local Development | Production Target (Kaggle) |
| :--- | :--- | :--- |
| **OS** | Windows 11 (PowerShell) | Ubuntu 22.04 LTS Container |
| **Python** | 3.13.3 (Local standard) | 3.10 / 3.11 (Kaggle base) |
| **Hardware** | Intel Core i-series / Local GPU | Dual Tesla T4 (16 GB × 2) or P100 (16 GB) |
| **System Memory** | 16–32 GB RAM | 30.0 GB RAM (permits `cache="ram"`) |
| **Session Budget** | Uncapped | Strict 12-hour session limit |
| **Internet Access** | Full connectivity | Enabled (for ultralytics & pycocotools) |

---

## 3. Desktop Mirror Synchronization

For frictionless Kaggle deployment without command-line dependencies, production artifacts are mirrored to:  
`C:\Users\srik2\Desktop\Filament_Colab_Run\`

1. [`moonshot_fulldata.ipynb`](file:///C:/Users/srik2/Desktop/Filament_Colab_Run/moonshot_fulldata.ipynb)  
   - SHA256: `fab5dc00a7180671809c8bbfa2e4f164f26ee483894e030043b6bd9ed1150c9c`  
   - Synchronized byte-for-byte with `notebooks/kaggle/moonshot_fulldata.ipynb`.
2. [`best.pt`](file:///C:/Users/srik2/Desktop/Filament_Colab_Run/best.pt)  
   - Size: 92,767,812 bytes (~92.8 MB).  
   - SHA256: `f444e87b39433881ae39608e9740c44c9bab0161590212047e224b6730d6b6f9`.  
   - The verified 60-epoch Fold-0 checkpoint ready to be uploaded as a private Kaggle dataset.

---

## 4. Software Dependencies

- `ultralytics == 8.3.145` (Strictly pinned for reproducible loss, mosaic, and segmentation heads)
- `pycocotools >= 2.0.7` (Fortran RLE encoding and decoding)
- `torch >= 2.1.0` (with CUDA support and AMP mixed precision)
- `numpy >= 1.24.0`, `scipy >= 1.10.0`, `pandas >= 2.0.0`, `opencv-python >= 4.8.0`
