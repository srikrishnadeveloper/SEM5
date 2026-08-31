# build_single_notebook.py
# Wraps filament_top50_single.py into ONE self-contained, error-proof Colab notebook
# with extensive real-time logging at every single step.
# Usage:
#   python build_single_notebook.py --out-dir "C:\Users\srik2\Desktop\Filament_Colab_Run"

import argparse
import json
import os
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "filament_top50_single.py"

MD_HEADER = """# ☀️ Solar Filament Segmentation Challenge 2026
## 🚀 TOP-50 Single-File End-to-End Pipeline (Google Colab Verified)

**One notebook. Complete transparency. Extensive live logging. Zero errors.**

---

### 📋 Quickstart (3 Simple Steps)
1. **Enable GPU:** Go to `Runtime -> Change runtime type -> Hardware accelerator = GPU` (T4 GPU is completely free and sufficient).
2. **Kaggle Credentials:**
   * *Option A (Recommended):* Add a secret named `KAGGLE_API_TOKEN` in the **Secrets** panel (key icon on the left) containing your Kaggle API token (`KGAT_...` or full `kaggle.json` from [kaggle.com/settings/api](https://www.kaggle.com/settings/api)). Toggle **Notebook access** ON.
   * *Option B:* If not using Secrets, Cell 2 will automatically provide an interactive prompt to enter your token or upload `kaggle.json`.
3. **Execute:** Click `Runtime -> Run all` (or run cell by cell).

---

### ⚡ Available Speed & Quality Presets
| Preset | Model Architecture | Resolution | Batch / Accum | Est. Time (T4) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fast`** | U-Net (`timm-efficientnet-b2`) | $768 \\times 768$ | Batch 4 / Accum 2 | ~35 min/fold | Fast, lightweight baseline with D4 TTA. |
| **`balanced`** | U-Net (`timm-efficientnet-b4`) | $1024 \\times 1024$ | Batch 2 / Accum 4 | ~70 min/fold | **Recommended:** Excellent accuracy and detail. |
| **`max`** | U-Net++ (`timm-efficientnet-b4` + scSE) | $1024 \\times 1024$ | Batch 1 / Accum 8 | ~3 hours/fold | Maximum capacity for fine boundary capture. |
| **`edgeattnet`** | `UNetEdgeTransformer` (MHSA bottleneck) | $512 \\times 512$ | Batch 4 / Accum 4 | ~2 hours/fold | Specialized solar edge-attention architecture. |

---

### 🏆 Key Advantages of this Pipeline
* **Memory-Mapped Caching:** High-res $2048\\times 2048$ JPEGs and COCO polygons are decoded **once** into fast uint8 memmaps with max-pooling (preserving ultra-thin filaments).
* **True Metric Optimization:** Direct instance multi-scale $m\\text{IoU}_{\\text{multiscale}}$ evaluation with greedy 1-to-1 matching and F1 penalty (matches official IEEE BigData Cup criteria).
* **Solar-Domain Preprocessing:** Hough circle solar disk masking, limb-darkening radial flattening, and CLAHE contrast enhancement.
* **Loss Function Mixture:** Combined `BCE(pos_weight) + Dice + Focal Tversky + Boundary Loss + clDice` for topological connectivity.
* **Inference & Ensembling:** D4 (8-way) test-time augmentation $\\rightarrow$ full-resolution probability maps $\\rightarrow$ OOF threshold/min-area grid search $\\rightarrow$ COCO RLE encoding.
"""

CELL_GPU_CHECK = """#@title 1. Environment & GPU Diagnostics
# Verify Python environment, CUDA availability, and GPU hardware
import sys, os, time, platform

print("=" * 70)
print(f" Python Version : {platform.python_version()} ({platform.system()} {platform.release()})")
print(f" Working Dir    : {os.getcwd()}")

try:
    import torch
    print(f" PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f" CUDA Available : {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        print(f" GPU Device     : {device_name} ({vram_gb:.2f} GB VRAM, Compute {compute_cap[0]}.{compute_cap[1]})")
        print(f" cuDNN Version  : {torch.backends.cudnn.version()}")
    else:
        print("\\n⚠️ WARNING: GPU acceleration is NOT active!")
        print("Please click 'Runtime' -> 'Change runtime type' -> select 'T4 GPU' and re-run.")
except ImportError:
    print("PyTorch is not yet installed (will be installed in Cell 2).")

print("=" * 70)
"""

CELL_INSTALL = """#@title 2. Install Required Dependencies
# Install segmentation-models-pytorch, albumentations, timm, pycocotools, etc.
import subprocess, sys, os

# Suppress warnings and optimize download requests
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_pip(*packages):
    print(f"[pip] Installing: {' '.join(packages)} ...", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages]
    subprocess.run(cmd, check=False)

# Colab ships CUDA torch. We only install the required machine learning packages:
run_pip("segmentation-models-pytorch", "albumentations", "timm")
run_pip("pycocotools", "scikit-image", "scipy", "opencv-python-headless", "pandas", "tqdm")
run_pip("kaggle", "kagglehub")

import torch
import segmentation_models_pytorch as smp
import albumentations as A
import pycocotools

print("\\n✅ All dependencies successfully installed and verified!", flush=True)
print(f"   • smp version: {smp.__version__}")
print(f"   • albumentations: {A.__version__}")
"""

CELL_DATA = """#@title 3. Kaggle Authentication & Data Download
# Secure authentication with fallback support and dataset verification
import json, os, subprocess, sys, glob

COMPETITION = "filament-segmentation-2026"
DATA_ROOT = "/content/data"
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs("/root/.kaggle", exist_ok=True)

# 1. Attempt reading from Colab Secrets
token = ""
try:
    from google.colab import userdata
    token = (userdata.get("KAGGLE_API_TOKEN") or "").strip()
    if token:
        print("[Auth] Found KAGGLE_API_TOKEN in Colab Secrets.", flush=True)
except Exception:
    pass

# 2. Fallback interactive prompt if secret is not set
if not token:
    print("\\n[Auth Notice] KAGGLE_API_TOKEN secret not detected.")
    token_input = input("Enter your Kaggle API token (KGAT_... or JSON contents), or press Enter to upload kaggle.json: ").strip()
    if token_input:
        token = token_input
    else:
        try:
            from google.colab import files
            print("Please upload your 'kaggle.json' file now:")
            uploaded = files.upload()
            for fn, content in uploaded.items():
                if fn.endswith(".json"):
                    token = content.decode("utf-8")
                    break
        except Exception as e:
            print("Upload failed:", e)

if not token:
    raise RuntimeError(
        "❌ Missing Kaggle API Credentials!\\n"
        "Please add a secret named 'KAGGLE_API_TOKEN' in the Secrets panel (key icon) "
        "with your token from https://www.kaggle.com/settings/api and toggle Notebook access ON."
    )

# 3. Configure Kaggle credentials on disk
if token.lstrip().startswith("{"):
    creds = json.loads(token)
    open("/root/.kaggle/kaggle.json", "w").write(json.dumps(creds))
    os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
    os.environ["KAGGLE_KEY"] = creds.get("key", "")
    print(f"[Auth] Authenticated as user: {creds.get('username')}", flush=True)
else:
    os.environ["KAGGLE_API_TOKEN"] = token
    open("/root/.kaggle/access_token", "w").write(token)
    print("[Auth] Authenticated using Kaggle Bearer Token (KGAT_...).", flush=True)

os.chmod("/root/.kaggle", 0o700)

# 4. Download and extract dataset
def find_base(root):
    hits = glob.glob(os.path.join(root, "**", "MAGFiLO_1.0_Kaggle_2026"), recursive=True)
    if hits:
        return hits[0]
    for cand in glob.glob(os.path.join(root, "**", "train", "train_images"), recursive=True):
        return os.path.dirname(os.path.dirname(cand))
    return None

BASE = find_base(DATA_ROOT)
if BASE is None:
    print(f"\\n[Data] Downloading competition dataset '{COMPETITION}' (~671 MB) ...", flush=True)
    rc = subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION,
                         "-p", DATA_ROOT, "-o"]).returncode
    if rc != 0:
        print("[Data] Kaggle CLI download failed, trying kagglehub fallback ...", flush=True)
        import kagglehub
        p = kagglehub.competition_download(COMPETITION)
        BASE = find_base(p) or p
    else:
        for z in glob.glob(os.path.join(DATA_ROOT, "*.zip")):
            print(f"[Data] Unzipping {os.path.basename(z)} ...", flush=True)
            subprocess.run(["unzip", "-qo", z, "-d", DATA_ROOT], check=False)
        BASE = find_base(DATA_ROOT)

if BASE is None:
    raise RuntimeError(f"❌ Could not locate MAGFiLO dataset under {DATA_ROOT}")

os.environ["FILAMENT_BASE_PATH"] = BASE
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
n_train = sum(len(glob.glob(os.path.join(BASE, "train", "train_images", e))) for e in IMG_EXTS)
n_test = sum(len(glob.glob(os.path.join(BASE, "test", "test_images", e))) for e in IMG_EXTS)

print("\\n" + "=" * 70)
print(f"✅ DATASET READY AT: {BASE}")
print(f"   • Training Images : {n_train} files (MAGFiLO v1.0)")
print(f"   • Test Images     : {n_test} files")
print(f"   • Annotations JSON: {os.path.exists(os.path.join(BASE, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'))}")
print("=" * 70, flush=True)

assert n_train > 0 and n_test > 0, "Dataset directory appears empty. Check download logs."
"""

CELL_CONFIG = """#@title 4. Configure & Execute Pipeline
# ---------------------------------------------------------------------------
# PRESET: "max" (TOP-50 Quality, ~3h) | "balanced" (~70m) | "fast" (~35m)
# FOLD  : 0..4 (use different folds across different accounts to ensemble)
# ---------------------------------------------------------------------------
PRESET = "max"        #@param ["max", "balanced", "fast", "edgeattnet"]
FOLD    = 0           #@param {type:"integer"}

# Initialize configuration
cfg = apply_preset(CFG, PRESET)
cfg.base = os.environ["FILAMENT_BASE_PATH"]
cfg.out_dir = "/content/filament_out"
cfg.fold = int(FOLD)
cfg.n_folds = 5

# Validate fold — MUST be 0, 1, 2, 3, or 4
assert 0 <= cfg.fold < cfg.n_folds, f"ERROR: FOLD must be 0..{cfg.n_folds-1}, got {cfg.fold}"

# Path derivation
cfg = resolve(cfg)

print("=" * 80)
print(f" PIPELINE CONFIGURATION: {cfg.preset.upper()}")
print(f"   • Model Architecture : {cfg.arch.upper()} (Encoder: {cfg.encoder})")
print(f"   • Input Resolution   : {cfg.img_size} x {cfg.img_size} (Inference: {cfg.full_res} x {cfg.full_res})")
print(f"   • Training Fold      : Fold {cfg.fold} of {cfg.n_folds}")
print(f"   • Batch & Accum      : Batch {cfg.batch_size} | Gradient Accum {cfg.accum} (Effective: {cfg.batch_size*cfg.accum})")
print(f"   • Epochs & Warmup    : {cfg.epochs} Epochs (Warmup: {cfg.warmup_epochs})")
print(f"   • Precision & EMA    : AMP fp16: {cfg.use_amp} | Model EMA: {cfg.use_ema}")
print(f"   • Test Inference     : D4 8-way TTA: {cfg.tta_d4} | Multi-scale: {cfg.tta_scales}")
print("=" * 80, flush=True)

# Run full pipeline: Cache -> Train -> OOF Search -> Test Submission
result = run(cfg, do_train=True, do_search=True, do_submit=True)

sub = result["submission"]
print("\\n" + "=" * 80)
print(" 🏁 RUN COMPLETE — SUMMARY OF RESULTS")
print("=" * 80)
print(f"   • Best Model Checkpoint : {result['best_path']}")
print(f"   • Optimal Threshold     : {result['threshold']:.2f}")
print(f"   • Optimal Min Area      : {result['min_area']} px")
print(f"   • Total Filaments Found : {0 if sub is None else len(sub):,}")
print("=" * 80, flush=True)
"""

CELL_VERIFY_SUBMISSION = """#@title 5. Inspect Submission & Validate Mask Integrity
# Validates CSV structure, row counts, and verifies that RLE strings decode into valid masks
import pandas as pd
import numpy as np
import pycocotools.mask as mask_util
import os

sub_path = os.path.join(result["cfg"].sub_dir, "submission.csv")
assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

df = pd.read_csv(sub_path)
print("=" * 70)
print(" SUBMISSION INTEGRITY REPORT")
print("=" * 70)
print(f" File Path       : {sub_path}")
print(f" File Size       : {os.path.getsize(sub_path)/1e3:.2f} KB")
print(f" Total Rows      : {len(df):,} filament predictions")
print(f" Column Names    : {list(df.columns)}")
assert list(df.columns) == ["filament_id", "segmentation_rle"], "Incorrect column headers!"

if len(df) == 0:
    print("\\n⚠️ WARNING: Submission CSV is EMPTY! Consider lowering threshold / min_area.")
else:
    print(f"\\nFirst 5 Predictions Preview:")
    display_df = df.head(5).copy()
    display_df["segmentation_rle"] = display_df["segmentation_rle"].str[:40] + "..."
    print(display_df.to_string(index=False))

    # Verify RLE decoding on random samples
    sample_indices = np.random.choice(len(df), size=min(3, len(df)), replace=False)
    print("\\nValidating RLE mask decode on sample filaments:")
    for idx in sample_indices:
        row = df.iloc[idx]
        rle_dict = {"size": [2048, 2048], "counts": row["segmentation_rle"].encode()}
        decoded = mask_util.decode(rle_dict)
        area = int(decoded.sum())
        assert decoded.shape == (2048, 2048), f"Shape mismatch: {decoded.shape}"
        assert area > 0, "Decoded mask has 0 positive pixels!"
        print(f"  ✓ {row['filament_id']}: Shape {decoded.shape}, Area = {area:,} px (Valid Binary Mask)")

print("=" * 70, flush=True)
"""

CELL_DOWNLOAD = """#@title 6. Package & Download Artifacts (Model + Submission + OOF Probs)
import os, glob, shutil

cfg = result["cfg"]
stage = f"/content/fold_{cfg.fold}_results"
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage, exist_ok=True)

# Copy trained weights, submission CSV, and metrics log
for src in glob.glob(os.path.join(cfg.model_dir, "*")):
    shutil.copy(src, stage)
for src in glob.glob(os.path.join(cfg.sub_dir, "*")):
    shutil.copy(src, stage)

# Copy test probability maps (.npy) for later ensembling
prob_src = os.path.join(cfg.prob_dir, f"test_fold_{cfg.fold}")
if os.path.isdir(prob_src):
    shutil.copytree(prob_src, os.path.join(stage, f"test_probs_fold_{cfg.fold}"))

files = glob.glob(os.path.join(stage, "**", "*"), recursive=True)
file_list = [f for f in files if os.path.isfile(f)]
print(f"[Package] Staging {len(file_list)} files in {stage} ...")

zip_path = f"/content/fold_{cfg.fold}_results"
archive = shutil.make_archive(zip_path, "zip", stage)
archive_size_mb = os.path.getsize(archive) / 1e6
print(f"\\n✅ Output Archive Created: {archive} ({archive_size_mb:.1f} MB)", flush=True)

try:
    from google.colab import files as colab_files
    print("[Download] Triggering automatic browser download ...")
    colab_files.download(archive)
except Exception as e:
    print("Auto-download unavailable (or blocked by browser pop-up settings):", e)
    print(f"You can manually download the file from: {archive}")
"""

CELL_ENSEMBLE = """#@title 7. (Optional) Multi-Fold Probability Ensemble
# Run this cell after collecting probability maps from 2 or more folds (e.g. from multiple accounts)
# Upload your `test_probs_fold_*` folders to /content/probs/
import glob, os

prob_dirs = sorted(glob.glob("/content/probs/test_probs_fold_*"))
print(f"[Ensemble] Found {len(prob_dirs)} fold probability directories:")
for d in prob_dirs:
    print(f"   • {d}")

if len(prob_dirs) >= 2:
    cfg = result["cfg"]
    mean_dir = "/content/probs/mean"
    merged = ensemble_probs(prob_dirs, mean_dir)
    ensemble_csv = os.path.join(cfg.sub_dir, "submission_ensemble.csv")
    df = submission_from_prob_dir(
        cfg, merged,
        threshold=result["threshold"],
        min_area=result["min_area"],
        out_csv=ensemble_csv,
    )
    print(f"\\n🎉 Ensemble Submission Complete: {len(df):,} filaments -> {ensemble_csv}")
    try:
        from google.colab import files as colab_files
        colab_files.download(ensemble_csv)
    except Exception as e:
        print(f"Download manually from {ensemble_csv}")
else:
    print("\\n[Ensemble Notice] At least 2 fold directories are required to perform an ensemble. Skipping.")
"""


def build(out_dir: Path):
    src = PIPELINE.read_text(encoding="utf-8")
    # Strip main execution block when embedding inside Colab notebook so Cell 4 only imports/defines
    src = src.replace('if __name__ == "__main__":\n    run(apply_preset(CFG))', '# Single-file pipeline classes & functions loaded successfully.\nprint("[Init] Pipeline loaded and ready for configuration.")')

    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"name": "colab_filament_top50_verified.ipynb", "provenance": [],
                  "toc_visible": True},
    }

    nb.cells = [
        nbformat.v4.new_markdown_cell(MD_HEADER),
        nbformat.v4.new_code_cell(CELL_GPU_CHECK),
        nbformat.v4.new_code_cell(CELL_INSTALL),
        nbformat.v4.new_code_cell(CELL_DATA),
        nbformat.v4.new_markdown_cell(
            "## 📦 4. Single-File SOTA Pipeline\n"
            "The cell below contains the complete, self-contained architecture, data loader, "
            "EdgeAttNet preprocessing, loss functions, D4 TTA inference, and threshold optimizer. "
            "Execute this cell once before running the training configuration."
        ),
        nbformat.v4.new_code_cell("#@title 4. Pipeline Engine Source (Execute Once)\n" + src),
        nbformat.v4.new_code_cell(CELL_CONFIG),
        nbformat.v4.new_code_cell(CELL_VERIFY_SUBMISSION),
        nbformat.v4.new_code_cell(CELL_DOWNLOAD),
        nbformat.v4.new_code_cell(CELL_ENSEMBLE),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save both canonical names for user convenience
    path_verified = out_dir / "colab_filament_top50_verified.ipynb"
    path_single = out_dir / "filament_top50_single.ipynb"
    
    for p in (path_verified, path_single):
        nbformat.write(nb, p)
        nbformat.validate(nbformat.read(p, as_version=4))
        json.load(open(p, encoding="utf-8"))
        print(f"wrote {p} (size: {p.stat().st_size/1e3:.0f} KB, cells: {len(nb.cells)})")
    
    return path_verified


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=HERE,
                    help="Directory to save the generated notebooks (defaults to notebooks/)")
    build(ap.parse_args().out_dir)
