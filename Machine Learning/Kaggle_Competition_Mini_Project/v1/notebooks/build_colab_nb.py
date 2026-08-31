# build_colab_nb.py
# creates a self-contained Colab notebook from the local code/ modules.

import os

import nbformat

project_root = r"C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project"
code_dir = os.path.join(project_root, "code")
out_dir = os.path.join(project_root, "notebooks")
out_nb = os.path.join(out_dir, "colab_filament_training.ipynb")

os.makedirs(out_dir, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "colab_filament_training.ipynb", "provenance": []},
}

nb.cells.append(nbformat.v4.new_markdown_cell("# Solar Filament Segmentation - Colab Training"))

# 1. mount drive and install
cell0 = """from google.colab import drive
drive.mount('/content/drive')

# create project folder in Drive
project_drive = '/content/drive/MyDrive/filament_kaggle'
os.makedirs(project_drive, exist_ok=True)

!pip install -q segmentation-models-pytorch==0.3.3 albumentations==2.0.8 kaggle kagglehub
"""
nb.cells.append(nbformat.v4.new_code_cell(cell0))

# 2. download data with kaggle token
cell1 = """import os, kagglehub, shutil

# set your Kaggle API access token from userdata, or paste it below
from google.colab import userdata
try:
    os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
except Exception as e:
    # fallback: paste token here manually
    os.environ['KAGGLE_API_TOKEN'] = 'YOUR_TOKEN_HERE'

# download competition data via kagglehub
path = kagglehub.competition_download('filament-segmentation-2026')
print('downloaded to', path)

# locate the folder that contains train/test
base = path
if os.path.isdir(os.path.join(path, 'MAGFiLO_1.0_Kaggle_2026', 'train')):
    base = os.path.join(path, 'MAGFiLO_1.0_Kaggle_2026')
elif not os.path.isdir(os.path.join(path, 'train')):
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d, 'train')):
            base = os.path.join(path, d)
            break

os.environ['FILAMENT_BASE_PATH'] = base
"""
nb.cells.append(nbformat.v4.new_code_cell(cell1))

# 3. write local code modules
nb.cells.append(nbformat.v4.new_code_cell("""import os
os.makedirs(os.path.join(project_drive, 'code'), exist_ok=True)"""))

for fname in sorted(os.listdir(code_dir)):
    if not fname.endswith(".py"):
        continue
    with open(os.path.join(code_dir, fname), "r", encoding="utf-8") as f:
        src = f.read()
    cell = f"%%writefile /content/drive/MyDrive/filament_kaggle/code/{fname}\n" + src
    nb.cells.append(nbformat.v4.new_code_cell(cell))

# 4. run training and create submission
train_cell = """import sys, os
sys.path.append(os.path.join(project_drive, 'code'))

os.environ['FILAMENT_TRAIN_RES'] = '1024'
os.environ['FILAMENT_INFERENCE_RES'] = '2048'
os.environ['FILAMENT_ENCODER'] = 'tu-efficientnet_b3'
os.environ['FILAMENT_BATCH_SIZE'] = '1'
os.environ['FILAMENT_ACCUMULATION'] = '2'
os.environ['FILAMENT_EPOCHS'] = '30'
os.environ['FILAMENT_LR'] = '2e-4'
os.environ['FILAMENT_MIN_LR'] = '1e-6'
os.environ['FILAMENT_PATIENCE'] = '10'
os.environ['FILAMENT_VAL_FOLD'] = '0'

import config, train, infer

best = train.train_fold(fold=0, epochs=config.EPOCHS, save_dir=project_drive)

sub = infer.run_test_submission(
    checkpoint_path=best,
    output_csv=os.path.join(project_drive, 'submission.csv'),
    image_size=config.TRAIN_RES,
    threshold=0.45,
    use_tta=True,
)
print(sub.head())
"""
nb.cells.append(nbformat.v4.new_code_cell(train_cell))

# 5. validate header
check_cell = """import pandas as pd
sub = pd.read_csv(os.path.join(project_drive, 'submission.csv'))
print(sub.shape)
print(sub.head())
assert sub.columns.tolist() == ['filament_id', 'segmentation_rle']
"""
nb.cells.append(nbformat.v4.new_code_cell(check_cell))

nbformat.write(nb, out_nb)
print("wrote", out_nb, "with", len(nb.cells), "cells")
