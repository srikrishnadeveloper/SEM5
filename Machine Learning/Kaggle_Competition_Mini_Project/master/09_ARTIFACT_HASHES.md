# 09_ARTIFACT_HASHES.md — Cryptographic Artifact Hashes & Inventory
**Project:** Solar Filament Segmentation Challenge 2026  
**Algorithm:** SHA-256  
**Audit Date:** September 8, 2026  

---

## 1. Primary Model Checkpoints

| Artifact Name | Size (Bytes) | Size (MB) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| `models/moonshot_2048/best.pt` | 92,767,812 | 92.77 MB | `f444e87b39433881ae39608e9740c44c9bab0161590212047e224b6730d6b6f9` |
| `plots/moonshot_2048/best.pt` | 92,767,812 | 92.77 MB | `f444e87b39433881ae39608e9740c44c9bab0161590212047e224b6730d6b6f9` |
| `plots/moonshot_2048/last.pt` | 92,767,812 | 92.77 MB | `6a8e105800ab784a27e1c842b3fb339e4332a1672eece5528206c4c6280bfaea` |
| `Desktop/Filament_Colab_Run/best.pt` | 92,767,812 | 92.77 MB | `f444e87b39433881ae39608e9740c44c9bab0161590212047e224b6730d6b6f9` |
| `models/best_deeplabv3p_res50d_fold_0.pth` | 49,438,208 | 49.44 MB | `55459d13091f8adfda1269a267f7652e34e4aedb54a3e097a4367325fffee805` |
| `models/best_segformer_mitb3_fold_0.pth` | 99,020,800 | 99.02 MB | `868be55107f6d2637b2e649277efd3b761a9dcd9398a914dd456c2e5631e38c0` |
| `models/best_unetpp_effb4_fold_0.pth` | 64,897,024 | 64.90 MB | `472a6c2d97839a0044b537a5f0b9dd2004e0187be9ae46a3a59b96abc20e4aed` |

---

## 2. Production Notebooks

| Artifact Name | Size (Bytes) | Size (KB) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| `notebooks/moonshot_fulldata.ipynb` | 20,744 | 20.26 KB | `fab5dc00a7180671809c8bbfa2e4f164f26ee483894e030043b6bd9ed1150c9c` |
| `notebooks/kaggle/moonshot_fulldata.ipynb` | 20,744 | 20.26 KB | `fab5dc00a7180671809c8bbfa2e4f164f26ee483894e030043b6bd9ed1150c9c` |
| `Desktop/Filament_Colab_Run/moonshot_fulldata.ipynb` | 20,744 | 20.26 KB | `fab5dc00a7180671809c8bbfa2e4f164f26ee483894e030043b6bd9ed1150c9c` |
| `notebooks/Moonshot_2048_Inference.ipynb` | 38,912 | 38.00 KB | `309f079d5c7b851440b736862d5f88d916df1d0184f29ab28037191ec3c78b24` |
| `notebooks/Moonshot_2048_Train_Fold0.ipynb` | 87,424 | 85.38 KB | `15344ca5cb35700e793f2b471efe338dd20162fd4a77b60f71f2a5d1fd34653f` |
| `notebooks/V8_1_Kaggle_Production.ipynb` | 126,976 | 124.00 KB | `29a7120d69e83bd2b7bf678e3236c23f6aa1725bc0e64575804d152a15ea61ed` |

---

## 3. Core PyTorch & Inference Source Code

| Artifact Name | Size (Bytes) | Size (KB) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| `moonshot_2048/predict_pytorch_gpu.py` | 10,240 | 10.00 KB | `08107f29c939ed6670ec1ca9b4e3e3a9a929d7c3979128416bb9a5fc461fd1ec` |
| `moonshot_2048/dataset_pytorch.py` | 9,812 | 9.58 KB | `dfada425ae17b15e5e5f40bd45b61de811682bfd7138b0ed10826090f17c8fe9` |
| `moonshot_2048/train_yolov8l.py` | 11,264 | 11.00 KB | `979eebdda70cf5114fd7cbea4eca98ac5245e41df782242202afa5c99c725c43` |
| `v8_1/3_infer_cascade.py` | 20,480 | 20.00 KB | `a043600510fcfc3c89b0a38e5a18ea21ff46670cffbac2f441136b80bde1ca27` |

---

## 4. Competition Datasets

| Artifact Name | Size (Bytes) | Size (MB) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| `MAGFiLO_1.0_Annotations_kaggle2026_train.json` | 48,616,700 | 48.62 MB | `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1` |

---

## 5. Verification Commands

To verify the integrity of any artifact locally in PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 "models/moonshot_2048/best.pt"
Get-FileHash -Algorithm SHA256 "C:\Users\srik2\Desktop\Filament_Colab_Run\best.pt"
Get-FileHash -Algorithm SHA256 "C:\Users\srik2\Desktop\Filament_Colab_Run\moonshot_fulldata.ipynb"
```
Or via Python:
```python
import hashlib
from pathlib import Path
h = hashlib.sha256(Path("models/moonshot_2048/best.pt").read_bytes()).hexdigest()
print(h)
```
