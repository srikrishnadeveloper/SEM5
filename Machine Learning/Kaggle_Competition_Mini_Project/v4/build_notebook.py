"""
Build Standalone Jupyter Notebook for V4 Grandmaster Edition Pipeline
"""

import json
import os
import argparse
from pathlib import Path

def build_v4_notebook(out_path: str):
    pipeline_py = Path(__file__).parent / "pipeline.py"
    with open(pipeline_py, "r", encoding="utf-8") as f:
        pipeline_code = f.read()

    cells = []

    # Cell 1: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌞 Solar Filament Segmentation Challenge 2026 — V4 Grandmaster Edition\n",
            "\n",
            "**Grandmaster Innovations & Maximum-Performance Pipeline:**\n",
            "- **Native 2048×2048 Gaussian Overlapping Tile Inference**: 3×3 overlapping 1024px grid with 2D Gaussian apodization blending (100% native camera resolution, zero edge seams)\n",
            "- **Tri-Architecture Grandmaster Ensemble**: `UnetPlusPlus (EffNet-B4 + SCSE)` + `SegFormer (mit_b3 Vision Transformer)` + `DeepLabV3Plus (ResNet50 + ASPP)` across 5 folds\n",
            "- **3-Channel Solar Feature Map**: Raw H-alpha + CLAHE + High-Pass Unsharp Ridge Mask\n",
            "- **Quad-Hybrid Loss with Lovász-Hinge**: `0.25*BCE + 0.30*SoftDice + 0.25*Focal + 0.20*Lovasz`\n",
            "- **Skeleton & Distance-Transform Watershed Splitter**: Separates intertwined crossing filament arches into distinct COCO instances\n",
            "- **Hardware & Memory Tuning**: Optimized for Dual Tesla T4 GPUs on Kaggle (~7.5GB VRAM per GPU, runs in ~7 hours, zero OOM risk)"
        ]
    })

    # Cell 2: Dependency Installation
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
            "import torch\n",
            "try:\n",
            "    import segmentation_models_pytorch as smp\n",
            "    import albumentations as A\n",
            "    import pycocotools\n",
            "    print(\"\\n[OK] All core dependencies (smp, albumentations, pycocotools) installed successfully!\")\n",
            "except ImportError as e:\n",
            "    print(f\"\\n[ERROR] Package import failed: {e}\")\n",
            "    print(\"--> IF ON KAGGLE: Make sure Internet is toggled to ON in Session Options!\")\n",
            "\n",
            "print(f\"\\nPyTorch Version : {torch.__version__}\")\n",
            "print(f\"CUDA Available  : {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU Device      : {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")"
        ]
    })

    # Cell 3: Google Drive & Kagglehub Fallback
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [2] Environment Setup & Storage Path Detection\n",
            "import os, glob\n",
            "from pathlib import Path\n",
            "\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    print(\"[Colab] Connecting to Google Drive...\")\n",
            "    drive.mount('/content/drive')\n",
            "except Exception as e:\n",
            "    print(f\"[Drive] Google Drive note: {e}\")\n",
            "\n",
            "def find_data_root():\n",
            "    cands = [\n",
            "        Path(\"/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"/kaggle/input/filament-segmentation-2026\"),\n",
            "        Path(\"/content/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"/content/filament_data/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"/content/drive/MyDrive/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "        Path(\"/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026\"),\n",
            "    ]\n",
            "    for c in cands:\n",
            "        if c.exists():\n",
            "            return str(c)\n",
            "    return None\n",
            "\n",
            "base_root = find_data_root()\n",
            "if base_root is None:\n",
            "    try:\n",
            "        import kagglehub\n",
            "        print(\"[Download] Downloading dataset via kagglehub...\")\n",
            "        download_path = kagglehub.competition_download('filament-segmentation-2026')\n",
            "        sub_cands = list(Path(download_path).glob('**/MAGFiLO_1.0_Kaggle_2026'))\n",
            "        base_root = str(sub_cands[0]) if sub_cands else str(download_path)\n",
            "    except Exception as e:\n",
            "        print(f\"[Download] Note on download: {e}\")\n",
            "\n",
            "print(f\"[Data] Active dataset directory: {base_root}\")"
        ]
    })

    # Cell 4: Production Pipeline Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [3] V4 Grandmaster Production Engine\n",
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
            "# [4] Execute Full V4 Grandmaster Multi-Architecture Pipeline\n",
            "cfg = Config()\n",
            "if base_root:\n",
            "    cfg.base_dir = base_root\n",
            "cfg.resolve_paths()\n",
            "\n",
            "print(f\"Starting V4 Grandmaster Training on {cfg.device}...\")\n",
            "run(cfg)"
        ]
    })

    # Cell 6: Submission Verification & Download
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [5] Submission Verification Check & Output Summary\n",
            "import pandas as pd, os\n",
            "from pathlib import Path\n",
            "\n",
            "sub_path = \"submissions/submission.csv\" if os.path.exists(\"submissions/submission.csv\") else \"/kaggle/working/submission.csv\"\n",
            "if os.path.exists(sub_path):\n",
            "    df = pd.read_csv(sub_path)\n",
            "    print(f\"=== SUBMISSION VERIFICATION [{sub_path}] ===\")\n",
            "    print(f\"Rows                 : {len(df)}\")\n",
            "    print(f\"Unique filament_ids  : {df['filament_id'].nunique()}\")\n",
            "    print(f\"Empty (PPP2) count   : {(df['segmentation_rle'] == 'PPP2').sum()}\")\n",
            "    print(f\"Predicted filaments  : {(df['segmentation_rle'] != 'PPP2').sum()}\")\n",
            "    print(\"\\nFirst 10 rows:\")\n",
            "    print(df.head(10).to_string(index=False))\n",
            "else:\n",
            "    print(f\"[WARN] Submission file not found at {sub_path}\")"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[OK] Standalone V4 Grandmaster Notebook built successfully -> {out_path} (cells: {len(cells)}, size: {os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="v4/v4_grandmaster_solar_filament_pipeline.ipynb")
    args = parser.parse_args()
    build_v4_notebook(args.out)
