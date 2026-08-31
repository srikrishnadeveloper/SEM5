# build_kaggle_nb.py
# creates a self-contained Kaggle notebook from the local code/ modules.

import json
import os
import re

import nbformat

project_root = r"C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project"
code_dir = os.path.join(project_root, "code")
out_dir = os.path.join(project_root, "notebooks", "kaggle")
out_nb = os.path.join(out_dir, "filament_training.ipynb")

os.makedirs(out_dir, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

# title
nb.cells.append(nbformat.v4.new_markdown_cell("# Solar Filament Segmentation - Kaggle Training"))

# 1. setup: paths and packages
setup = """# Kaggle free GPU is a P100 (sm_60). The preinstalled PyTorch 2.13 only supports sm_70+,
# so install the cu118 build of PyTorch 2.5.1 which still includes sm_60.
!pip install -q --upgrade torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
!pip install -q segmentation-models-pytorch==0.3.3 albumentations==2.0.8

import os, sys
# kaggle puts competition data in /kaggle/input/<competition> or /kaggle/input/competitions/<competition>
comp_root = None
for base in ['/kaggle/input/competitions/filament-segmentation-2026', '/kaggle/input/filament-segmentation-2026']:
    if os.path.isdir(base):
        comp_root = base
        break
else:
    # fallback: look for any folder that contains 'filament'
    for d in os.listdir('/kaggle/input'):
        if 'filament' in d.lower():
            comp_root = os.path.join('/kaggle/input', d)
            break

# the unzipped competition data is usually under a subfolder named MAGFiLO_1.0_Kaggle_2026
if comp_root and os.path.isdir(os.path.join(comp_root, 'MAGFiLO_1.0_Kaggle_2026')):
    os.environ['FILAMENT_BASE_PATH'] = os.path.join(comp_root, 'MAGFiLO_1.0_Kaggle_2026')
else:
    os.environ['FILAMENT_BASE_PATH'] = comp_root

os.makedirs('/kaggle/working/code', exist_ok=True)
"""
nb.cells.append(nbformat.v4.new_code_cell(setup))

# 2. write each code file into the kaggle working directory
for fname in sorted(os.listdir(code_dir)):
    if not fname.endswith(".py"):
        continue
    with open(os.path.join(code_dir, fname), "r", encoding="utf-8") as f:
        src = f.read()
    # escape triple backticks if any
    cell = f"%%writefile /kaggle/working/code/{fname}\n" + src
    nb.cells.append(nbformat.v4.new_code_cell(cell))

# 3. run training and generate submission
train_cell = """import sys, torch
sys.path.append('/kaggle/working/code')

print('torch version', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device name', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))

# set training config for a Kaggle P100/T4
os.environ['FILAMENT_TRAIN_RES'] = '1024'
os.environ['FILAMENT_INFERENCE_RES'] = '2048'
os.environ['FILAMENT_ENCODER'] = 'tu-efficientnet_b3'
os.environ['FILAMENT_BATCH_SIZE'] = '1'
os.environ['FILAMENT_ACCUMULATION'] = '2'
os.environ['FILAMENT_EPOCHS'] = '30'
os.environ['FILAMENT_LR'] = '2e-4'
os.environ['FILAMENT_MIN_LR'] = '1e-6'
os.environ['FILAMENT_PATIENCE'] = '10'
# keep 5-fold group split by year (config default)
# os.environ['FILAMENT_N_FOLDS'] = '5'
os.environ['FILAMENT_VAL_FOLD'] = '0'

import config, train, infer

best = train.train_fold(fold=0, epochs=config.EPOCHS, save_dir='/kaggle/working')
print('best checkpoint', best)

# create submission on the test set using the best checkpoint
sub = infer.run_test_submission(
    checkpoint_path=best,
    output_csv='/kaggle/working/submission.csv',
    image_size=config.TRAIN_RES,
    threshold=0.45,
    use_tta=True,
)
print('submission head:')
print(sub.head())
"""
nb.cells.append(nbformat.v4.new_code_cell(train_cell))

# 4. ensure the csv has the correct header and show shapes
check_cell = """import pandas as pd
sub = pd.read_csv('/kaggle/working/submission.csv')
print(sub.shape)
print(sub.head())
assert sub.columns.tolist() == ['filament_id', 'segmentation_rle'], 'bad header'
"""
nb.cells.append(nbformat.v4.new_code_cell(check_cell))

nbformat.write(nb, out_nb)
print("wrote", out_nb, "with", len(nb.cells), "cells")
