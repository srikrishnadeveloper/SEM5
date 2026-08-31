"""
Build Standalone Jupyter Notebook for V3 Beast Mode Pipeline
"""

import json
import os
import argparse
from pathlib import Path

def build_v3_notebook(out_path: str):
    pipeline_py = Path(__file__).parent / "pipeline.py"
    with open(pipeline_py, "r", encoding="utf-8") as f:
        pipeline_code = f.read()

    cells = []

    # Cell 1: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌞 Solar Filament Segmentation Challenge 2026 — V3 Beast Mode Pipeline\n",
            "\n",
            "**Key Innovations & Hardcoded Best Configuration:**\n",
            "- **High-Resolution 1024×1024 Training**: 4× native pixel detail preserves thin thread filaments\n",
            "- **Heterogeneous 10-Model Ensemble**: `UnetPlusPlus (EffNet-B4 + SCSE)` + `DeepLabV3Plus (ResNet50d + ASPP)` across 5 folds\n",
            "- **3-Channel Astronomical Feature Map**: Raw H-alpha + CLAHE + Solar High-Pass Unsharp Ridge Mask\n",
            "- **Quad-Hybrid Loss with Lovász-Hinge**: `0.25*BCE + 0.30*SoftDice + 0.25*Focal + 0.20*Lovasz`\n",
            "- **Inference**: 40 forward passes with 4-Flip Test-Time Augmentation (TTA) + 2048px Bilinear Upsampling\n",
            "- **Post-Processing**: Solar Limb Boundary Masking + Morphological Closing + Connected Components + Area Filtering\n",
            "- **Estimated Kaggle Execution Time**: ~6 hours on GPU T4 x2 / P100 (fits comfortably in 12h limit)"
        ]
    })

    # Cell 2: Dependencies Installation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [1] Install Dependencies Safely & Quickly\n",
            "import os, subprocess, sys, importlib, socket\n",
            "socket.setdefaulttimeout(15.0)\n",
            "\n",
            "def run_pip(*args):\n",
            "    print(f\"[pip] Installing: {' '.join(args)}\", flush=True)\n",
            "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", *args], check=False)\n",
            "\n",
            "try:\n",
            "    import segmentation_models_pytorch\n",
            "except ImportError:\n",
            "    run_pip(\"segmentation-models-pytorch\", \"albumentations\", \"timm\", \"pycocotools\", \"scikit-image\", \"scipy\", \"opencv-python-headless\", \"pandas\", \"tqdm\", \"kagglehub\")\n",
            "\n",
            "# Verify GPU & P100 sm_60 compatibility\n",
            "try:\n",
            "    import torch\n",
            "    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:\n",
            "        print(\"[GPU] Detected P100 (sm_60) architecture; installing cu118 compatibility wheel...\")\n",
            "        run_pip(\"torch==2.5.1+cu118\", \"torchvision==0.20.1+cu118\", \"--index-url\", \"https://download.pytorch.org/whl/cu118\")\n",
            "        importlib.invalidate_caches()\n",
            "except Exception:\n",
            "    pass\n",
            "\n",
            "import torch\n",
            "try:\n",
            "    import segmentation_models_pytorch as smp\n",
            "    import albumentations as A\n",
            "    import pycocotools\n",
            "    print(\"\\n[OK] All core dependencies (smp, albumentations, pycocotools) installed successfully!\")\n",
            "except ImportError as e:\n",
            "    print(f\"\\n[ERROR] Package import failed: {e}\")\n",
            "    print(\"\\n--> IF ON KAGGLE: Internet access is turned OFF by default!\")\n",
            "    print(\"--> FIX: On the right panel, click 'Session options' -> toggle 'Internet' to ON -> re-run this cell.\")\n",
            "\n",
            "print(f\"\\nPyTorch Version : {torch.__version__}\")\n",
            "print(f\"CUDA Available  : {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU Device      : {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")"
        ]
    })

    # Cell 3: Dataset Setup (Google Drive / Kaggle Input auto-detect)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [2] Locate / Mount Google Drive / Setup Competition Dataset\n",
            "import os, glob, shutil, zipfile\n",
            "from pathlib import Path\n",
            "\n",
            "DATA_PATH = None\n",
            "\n",
            "# Step 1: If on Google Colab, mount Google Drive and unpack your uploaded zip\n",
            "if os.path.exists(\"/content\"):\n",
            "    print(\"[Colab] Connecting to Google Drive...\")\n",
            "    try:\n",
            "        from google.colab import drive\n",
            "        if not os.path.exists(\"/content/drive/MyDrive\"):\n",
            "            drive.mount(\"/content/drive\")\n",
            "            \n",
            "        drive_zip = Path(\"/content/drive/MyDrive/filament-segmentation-2026.zip\")\n",
            "        if not drive_zip.exists():\n",
            "            zips = list(Path(\"/content/drive/MyDrive\").glob(\"**/filament-segmentation-2026.zip\"))\n",
            "            if zips:\n",
            "                drive_zip = zips[0]\n",
            "\n",
            "        if drive_zip.exists():\n",
            "            print(f\"[Drive] Found dataset zip in Google Drive: {drive_zip}\")\n",
            "            dest_dir = Path(\"/content/MAGFiLO_1.0_Kaggle_2026\")\n",
            "            if not (dest_dir.exists() and (dest_dir / \"train\" / \"train_images\").exists()):\n",
            "                print(\"[Drive] Unpacking zip to high-speed local SSD (/content/)...\", flush=True)\n",
            "                with zipfile.ZipFile(drive_zip, \"r\") as z:\n",
            "                    z.extractall(\"/content\")\n",
            "                print(\"[Drive] Unpack complete!\")\n",
            "            DATA_PATH = dest_dir if dest_dir.exists() else Path(\"/content\")\n",
            "    except Exception as e:\n",
            "        print(f\"[Drive] Google Drive note: {e}\")\n",
            "\n",
            "# Step 2: Check standard Kaggle / local paths if not on Colab or already extracted\n",
            "if DATA_PATH is None or not DATA_PATH.exists():\n",
            "    candidate_paths = [\n",
            "        Path(\"/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"/kaggle/input/filament-segmentation-2026\"),\n",
            "        Path(\"/content/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"data/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"../data/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "    ]\n",
            "    for p in candidate_paths:\n",
            "        if p.exists():\n",
            "            DATA_PATH = p\n",
            "            break\n",
            "\n",
            "# Step 3: Fallback via kagglehub if needed\n",
            "if DATA_PATH is None or not DATA_PATH.exists():\n",
            "    print(\"[Data] Dataset not found. Downloading via kagglehub...\")\n",
            "    try:\n",
            "        import kagglehub\n",
            "        downloaded = kagglehub.competition_download(\"filament-segmentation-2026\")\n",
            "        DATA_PATH = Path(downloaded)\n",
            "        subdirs = list(DATA_PATH.glob(\"**/MAGFiLO_1.0_Kaggle_2026\"))\n",
            "        if subdirs:\n",
            "            DATA_PATH = subdirs[0]\n",
            "        print(f\"[Data] Downloaded to: {DATA_PATH}\")\n",
            "    except Exception as e:\n",
            "        print(f\"[Warning] kagglehub download failed: {e}. Fallback to data directory.\")\n",
            "        DATA_PATH = Path(\"data/MAGFiLO_1.0_Kaggle_2026\")\n",
            "\n",
            "print(f\"\\n[Data] Active dataset directory: {DATA_PATH}\")"
        ]
    })

    # Cell 4: Full Production Pipeline Engine
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [3] V3 Beast Mode Pipeline Source Code\n",
            pipeline_code
        ]
    })

    # Cell 5: Execution Trigger
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [4] Run V3 Beast Mode Training & Inference\n",
            "cfg = Config(\n",
            "    base_dir=str(DATA_PATH) if DATA_PATH else \"\",\n",
            "    img_size=1024,\n",
            "    batch_size=2,\n",
            "    accum_steps=4,\n",
            "    epochs=25,\n",
            "    train_all_folds=True,\n",
            "    train_both_architectures=True,\n",
            ")\n",
            "\n",
            "# Launch execution\n",
            "run(cfg)"
        ]
    })

    # Cell 6: Verification & Download
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [5] Inspect Final Submission & Verification Check\n",
            "import pandas as pd, os\n",
            "\n",
            "sub_path = os.path.join(cfg.output_dir, \"submission.csv\")\n",
            "if os.path.exists(sub_path):\n",
            "    df = pd.read_csv(sub_path)\n",
            "    print(f\"\\nSubmission CSV: {sub_path}\")\n",
            "    print(f\"Total Rows        : {len(df):,}\")\n",
            "    print(f\"Unique IDs        : {df['filament_id'].nunique():,}\")\n",
            "    print(f\"Empty (PPP2)      : {(df['segmentation_rle'] == 'PPP2').sum():,}\")\n",
            "    print(f\"Detected          : {(df['segmentation_rle'] != 'PPP2').sum():,}\")\n",
            "    print(f\"\\nFirst 10 Rows:\")\n",
            "    print(df.head(10).to_string(index=False))\n",
            "    \n",
            "    try:\n",
            "        from google.colab import files\n",
            "        files.download(sub_path)\n",
            "        print(\"\\n[Colab] Triggered download for submission.csv\")\n",
            "    except Exception:\n",
            "        pass\n",
            "else:\n",
            "    print(f\"[Error] Submission file not found at {sub_path}\")"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[OK] Standalone V3 Beast Notebook built successfully -> {out_path} (cells: {len(cells)}, size: {os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="v3/v3_beast_solar_filament_pipeline.ipynb")
    args = parser.parse_args()
    build_v3_notebook(args.out)
