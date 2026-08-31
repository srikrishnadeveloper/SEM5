"""
Build Standalone Kaggle / Colab Notebook for V2 Pipeline (Bullet-Proof Production Edition)
Creates a single, self-contained .ipynb file that embeds the entire V2 codebase,
handles GPU compatibility (P100 / T4), dependencies, data paths, training, and submission.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def generate_notebook(output_path: str = "v2_solar_filament_pipeline.ipynb"):
    pipeline_file = Path(__file__).parent / "pipeline.py"
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Cannot find pipeline.py at {pipeline_file}")
        
    with open(pipeline_file, "r", encoding="utf-8") as f:
        pipeline_code = f.read()

    cells = []

    # ── Cell 1: Markdown Header ──────────────────────────────────────────────
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌞 Solar Filament Segmentation Challenge 2026 — V2 Production Pipeline\n",
            "\n",
            "**Hardcoded Best Architecture & Settings:**\n",
            "- **Model:** `smp.UnetPlusPlus` + `efficientnet-b4` (ImageNet pre-trained 3-channel RGB)\n",
            "- **Attention:** SCSE (Spatial & Channel Squeeze & Excitation)\n",
            "- **Loss:** Balanced `0.30 * BCE + 0.40 * Dice + 0.30 * Focal` (zero gradient fighting)\n",
            "- **Resolution:** 512×512 fast training → 2048×2048 bilinear probability upsampling\n",
            "- **TTA:** 4-Flip Test-Time Augmentation (Original, Horizontal, Vertical, HV)\n",
            "- **Validation:** GroupKFold by Year + Fast OOF Threshold/Area Grid Search\n",
            "- **Post-Processing:** Morphological Closing + Connected Components + Area Filter + COCO RLE\n",
            "- **GPU Support:** NVIDIA P100 (sm_60 compatible) & T4 / A100 / L4 / V100"
        ]
    })

    # ── Cell 2: Dependencies Installation ────────────────────────────────────
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
            "# Install required libraries\n",
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
            "# Verify installation\n",
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

    # ── Cell 3: Dataset Setup ────────────────────────────────────────────────
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
            "        # Check for your uploaded zip file in My Drive\n",
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

    # ── Cell 4: Embedded V2 Pipeline ─────────────────────────────────────────
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [3] V2 Production Pipeline Source Code\n",
            pipeline_code
        ]
    })

    # ── Cell 5: Production Execution ─────────────────────────────────────────
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [4] Run V2 Production Training & Inference\n",
            "# Hardcoded with the best hyperparameters — no user configuration needed.\n",
            "cfg = Config(\n",
            "    base_dir=str(DATA_PATH) if DATA_PATH else \"\",\n",
            "    train_all_folds=True,  # 5-fold ensemble for highest score (>0.29)\n",
            "    val_fold=0\n",
            ")\n",
            "\n",
            "# Launch execution\n",
            "run(cfg)"
        ]
    })

    # ── Cell 6: Submission Verification & Inspection ─────────────────────────
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
            "    print(f\"Total Rows    : {len(df):,}\")\n",
            "    print(f\"Unique IDs    : {df['filament_id'].nunique():,}\")\n",
            "    print(f\"Empty (PPP2)  : {(df['segmentation_rle'] == 'PPP2').sum()}\")\n",
            "    print(f\"Detected      : {(df['segmentation_rle'] != 'PPP2').sum()}\")\n",
            "    print(\"\\nFirst 10 Rows:\")\n",
            "    print(df.head(10).to_string(index=False))\n",
            "    \n",
            "    # Auto-download on Google Colab if available\n",
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

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[OK] Standalone Notebook built successfully -> {output_path} (cells: {len(cells)}, size: {os.path.getsize(output_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="v2_solar_filament_pipeline.ipynb", help="Output .ipynb path")
    args = parser.parse_args()
    generate_notebook(args.out)
