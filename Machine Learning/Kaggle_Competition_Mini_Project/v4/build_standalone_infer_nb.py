"""
Builds a 1-click Standalone Inference Notebook for Kaggle.
Loads saved checkpoints and generates submission.csv in <90 seconds without training!
"""

import json
import os

NOTEBOOK_PATH = r"C:\Users\srik2\Desktop\Filament_Colab_Run\standalone_inference_pipeline.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 1-Click Fast Inference & Submission Engine\n",
            "### Solar Filament Segmentation Challenge 2026\n",
            "**Purpose:** Loads any trained checkpoint(s) (`.pth`) and generates `submission.csv` in **<90 seconds** without waiting for training!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [1] Install dependencies\n",
            "!pip install -q segmentation-models-pytorch albumentations pycocotools scikit-image scipy pandas\n",
            "print('Dependencies ready!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [2] Configuration & Checkpoint Selection\n",
            "import os, glob\n",
            "\n",
            "# Auto-detect test images\n",
            "candidates = [\n",
            "    '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images',\n",
            "    '/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images',\n",
            "    '../data/MAGFiLO_1.0_Kaggle_2026/test/test_images'\n",
            "]\n",
            "TEST_DIR = next((p for p in candidates if os.path.exists(p)), None)\n",
            "print('Test Images Directory:', TEST_DIR)\n",
            "\n",
            "# Search for all saved checkpoints in /kaggle/working/models or attached datasets\n",
            "CKPT_PATHS = glob.glob('/kaggle/working/models/*.pth') + glob.glob('/kaggle/working/*.pth') + glob.glob('/kaggle/input/**/*.pth', recursive=True)\n",
            "print(f'Found {len(CKPT_PATHS)} checkpoint(s):', CKPT_PATHS)\n",
            "\n",
            "IMG_SIZE = 1024       # Resolution\n",
            "THRESHOLD = 0.45      # Probability threshold\n",
            "MIN_AREA = 200        # Minimum filament area in pixels\n",
            "USE_TTA = True        # 4-flip Test-Time Augmentation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [3] Run Inference & Produce submission.csv\n",
            "import time, cv2, torch, numpy as np, pandas as pd\n",
            "from tqdm import tqdm\n",
            "from scipy.ndimage import distance_transform_edt\n",
            "from skimage.feature import peak_local_max\n",
            "from skimage.segmentation import watershed\n",
            "import pycocotools.mask as mask_utils\n",
            "import segmentation_models_pytorch as smp\n",
            "\n",
            "def create_solar_features(img_raw):\n",
            "    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))\n",
            "    img_clahe = clahe.apply(img_raw)\n",
            "    blurred = cv2.GaussianBlur(img_raw, (0, 0), sigmaX=3.0)\n",
            "    unsharp = cv2.addWeighted(img_raw, 1.5, blurred, -0.5, 0)\n",
            "    return np.stack([img_raw, img_clahe, unsharp], axis=-1)\n",
            "\n",
            "def generate_solar_limb_mask(h=2048, w=2048, radius_ratio=0.93):\n",
            "    cy, cx = h / 2.0, w / 2.0\n",
            "    r = min(h, w) / 2.0 * radius_ratio\n",
            "    y, x = np.ogrid[:h, :w]\n",
            "    return ((x - cx)**2 + (y - cy)**2 <= r**2).astype(np.uint8)\n",
            "\n",
            "def encode_coco_rle(binary_mask):\n",
            "    if binary_mask.sum() == 0:\n",
            "        return 'PPP2'\n",
            "    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))\n",
            "    encoded = mask_utils.encode(fortran_mask)\n",
            "    return encoded['counts'].decode('utf-8') if isinstance(encoded['counts'], bytes) else encoded['counts']\n",
            "\n",
            "def split_filament_instances(binary_mask, min_area=200, min_distance=15):\n",
            "    if binary_mask.sum() == 0:\n",
            "        return []\n",
            "    dist = distance_transform_edt(binary_mask)\n",
            "    coords = peak_local_max(dist, min_distance=min_distance, labels=binary_mask)\n",
            "    if len(coords) <= 1:\n",
            "        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)\n",
            "        return [(labels == lbl).astype(np.uint8) for lbl in range(1, n_lbl) if stats[lbl, cv2.CC_STAT_AREA] >= min_area]\n",
            "    markers = np.zeros(dist.shape, dtype=np.int32)\n",
            "    for i, pt in enumerate(coords):\n",
            "        markers[pt[0], pt[1]] = i + 1\n",
            "    labels = watershed(-dist, markers, mask=binary_mask)\n",
            "    instances = [(labels == lbl).astype(np.uint8) for lbl in range(1, labels.max() + 1) if (labels == lbl).sum() >= min_area]\n",
            "    return instances if instances else [(binary_mask > 0).astype(np.uint8)]\n",
            "\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print('Using Device:', device)\n",
            "\n",
            "models = []\n",
            "for p in CKPT_PATHS:\n",
            "    ckpt = torch.load(p, map_location=device)\n",
            "    arch = ckpt.get('arch', 'UnetPlusPlus')\n",
            "    encoder = ckpt.get('encoder', 'efficientnet-b4')\n",
            "    attention = ckpt.get('attention', 'scse')\n",
            "    if arch.lower() == 'unetplusplus':\n",
            "        m = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1, decoder_attention_type=attention if attention else None)\n",
            "    elif arch.lower() == 'deeplabv3plus':\n",
            "        m = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)\n",
            "    else:\n",
            "        m = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)\n",
            "    clean_sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}\n",
            "    m.load_state_dict(clean_sd)\n",
            "    m.to(device).eval()\n",
            "    models.append(m)\n",
            "    print(f'Loaded checkpoint: {p} (Val Dice: {ckpt.get(\"val_dice\", 0.0):.4f})')\n",
            "\n",
            "test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpeg')) + glob.glob(os.path.join(TEST_DIR, '*.jpg')))\n",
            "print(f'Starting inference on {len(test_files)} test images across {len(models)} model(s)...')\n",
            "\n",
            "solar_mask = generate_solar_limb_mask(2048, 2048, radius_ratio=0.93)\n",
            "mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)\n",
            "std = np.array([0.229, 0.224, 0.225], dtype=np.float32)\n",
            "\n",
            "records = []\n",
            "t0 = time.time()\n",
            "for fpath in tqdm(test_files, desc='Predicting'):\n",
            "    fn = os.path.basename(fpath)\n",
            "    stem = os.path.splitext(fn)[0]\n",
            "    img_raw = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)\n",
            "    if img_raw is None: continue\n",
            "    h, w = img_raw.shape[:2]\n",
            "    feats = create_solar_features(img_raw)\n",
            "    feats_resized = cv2.resize(feats, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)\n",
            "    norm_img = (feats_resized.astype(np.float32) / 255.0 - mean) / std\n",
            "    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)\n",
            "    ensemble_prob = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)\n",
            "    for m in models:\n",
            "        with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):\n",
            "            if USE_TTA:\n",
            "                p1 = torch.sigmoid(m(tensor))\n",
            "                p2 = torch.flip(torch.sigmoid(m(torch.flip(tensor, dims=[2]))), dims=[2])\n",
            "                p3 = torch.flip(torch.sigmoid(m(torch.flip(tensor, dims=[3]))), dims=[3])\n",
            "                p4 = torch.flip(torch.sigmoid(m(torch.flip(tensor, dims=[2, 3]))), dims=[2, 3])\n",
            "                prob = (p1 + p2 + p3 + p4) / 4.0\n",
            "            else:\n",
            "                prob = torch.sigmoid(m(tensor))\n",
            "        ensemble_prob += prob.squeeze().cpu().numpy() / len(models)\n",
            "    prob_2048 = cv2.resize(ensemble_prob, (w, h), interpolation=cv2.INTER_LINEAR) * solar_mask\n",
            "    bin_mask = (prob_2048 > THRESHOLD).astype(np.uint8)\n",
            "    instances = split_filament_instances(bin_mask, min_area=MIN_AREA)\n",
            "    if not instances:\n",
            "        records.append({'filament_id': f'{stem}_1', 'segmentation_rle': 'PPP2'})\n",
            "    else:\n",
            "        for idx, inst in enumerate(instances, 1):\n",
            "            records.append({'filament_id': f'{stem}_{idx}', 'segmentation_rle': encode_coco_rle(inst)})\n",
            "\n",
            "df = pd.DataFrame(records)\n",
            "df.to_csv('/kaggle/working/submission.csv', index=False)\n",
            "print(f'✅ Generated /kaggle/working/submission.csv with {len(df)} predictions in {time.time() - t0:.1f}s!')\n",
            "print(df.head(10))"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "language_info": {"name": "python", "version": "3.10"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"[OK] Standalone Inference Notebook created at: {NOTEBOOK_PATH}")
