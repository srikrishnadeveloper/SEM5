# build_colab_nodrive_payloads.py
# generates self-contained, no-Drive Colab notebooks for multi-account runs.
# each account gets its own payload JSON + .ipynb; no Google Drive mount is needed.

import argparse
import json
import os
from pathlib import Path

import nbformat


def build_payload(account_id: int, n_folds: int = 5) -> dict:
    """one payload per Colab account.  each account gets a different
    config (fast/medium/strong) and a different validation fold so the
    three runs produce a cheap ensemble."""
    configs = {
        i: {"name": f"filament_top50_b4_fold_{i-1}", "encoder": "tu-efficientnet_b4",
            "train_res": 1024, "batch_size": 1, "accumulation": 8,
            "epochs": 20, "tta": True} for i in (1, 2, 3)
    }
    cfg = configs.get(account_id, configs[3]).copy()
    cfg.update({
        "account_id": account_id,
        "model_name": "unetplusplus",
        "inference_res": 2048,
        "lr": "1e-4",
        "min_lr": "1e-6",
        "patience": 7,
        "n_folds": n_folds,
        "val_fold": (account_id - 1) % n_folds,
        "threshold": 0.45,
        "download_results": True,
        "competition": "filament-segmentation-2026",
        "seed_offset": account_id,
    })
    return cfg


def _write_modules_cells(nb: nbformat.NotebookNode, code_dir: Path, project_drive: str):
    for fname in sorted(os.listdir(code_dir)):
        if not fname.endswith(".py"):
            continue
        src = (code_dir / fname).read_text(encoding="utf-8")
        cell = f"%%writefile {project_drive}/code/{fname}\n" + src
        nb.cells.append(nbformat.v4.new_code_cell(cell))
    return nb


def build_notebook(
    payload: dict,
    code_dir: Path,
    kaggle_username: str = "",
    kaggle_token: str = "",
) -> nbformat.NotebookNode:
    project_drive = "/content/filament_kaggle"
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {
            "name": f"account_{payload['account_id']}_payload_nodrive.ipynb",
            "provenance": [],
        },
        "accelerator": "GPU",
    }

    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            f"# Solar Filament Segmentation — Colab no-Drive payload (account {payload['account_id']})\n\n"
            "Run with a T4 GPU runtime. No Google Drive mount is required."
        )
    )

    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            "## Runtime & secret setup\n\n"
            "1. Runtime → Change runtime type → Hardware accelerator = **GPU**, Runtime shape = **T4**.\n"
            "2. Open the Colab **Secrets** panel (key icon on the left), add a secret named `KAGGLE_API_TOKEN`, paste your Kaggle API access token (the `KGAT_...` string from https://www.kaggle.com/settings/api), and toggle **Notebook access** on.\n"
            "3. Do **not** paste the old `kaggle.json` contents here — the new API uses the raw `KGAT_...` token.\n"
            "4. Do **not** share the notebook with the secret set."
        )
    )

    install_cell = """# Colab already ships a GPU torch. Re-downloading the 800 MB +cu118 wheel often
# hits a hash mismatch on flaky session networks, so we only install the rest and
# verify the pre-installed torch can see the GPU. If torch is missing, we fall
# back to installing the CUDA 11.8 build.
import subprocess, sys

try:
    import torch
    print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())
except Exception as e:
    print('torch not pre-installed, installing cu118 build:', e)
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
        'torch==2.5.1+cu118', 'torchvision==0.20.1+cu118',
        '--index-url', 'https://download.pytorch.org/whl/cu118',
    ])

!pip install -q --no-cache-dir segmentation-models-pytorch==0.3.3 albumentations==2.0.8
!pip install -q --no-cache-dir pycocotools scikit-image scipy scikit-learn pandas numpy matplotlib opencv-python-headless tqdm kaggle kagglehub
"""
    nb.cells.append(nbformat.v4.new_code_cell(install_cell))

    payload_json = json.dumps(payload, indent=2)
    username = kaggle_username or "srikrishnaos"
    token = kaggle_token or "YOUR_TOKEN_HERE"
    setup_cell = f"""import json, os, pathlib, shutil, sys
from google.colab import userdata

PAYLOAD = json.loads(r'''
{payload_json}
''')

# Kaggle credentials: read the new-style API access token from the Colab secret
# KAGGLE_API_TOKEN. The secret must be the raw KGAT_... string from
# https://www.kaggle.com/settings/api, not the old kaggle.json contents.
# If you passed --kaggle-token when building, that value is used as a fallback.
kaggle_token = {repr(token)}

try:
    token = userdata.get('KAGGLE_API_TOKEN')
    print('read KAGGLE_API_TOKEN from Colab secret')
except Exception as e:
    print('WARNING: could not read KAGGLE_API_TOKEN secret:', e)
    token = kaggle_token

if not token or token == 'YOUR_TOKEN_HERE':
    raise ValueError(
        "No Kaggle API token found. "
        "Open the Colab Secrets panel, add a secret named KAGGLE_API_TOKEN, "
        "and paste your Kaggle API access token (the KGAT_... string)."
    )

# Accept a raw KGAT token or a legacy {{\"username\": ..., \"key\": ...}} JSON
if token.strip().startswith(chr(123)):
    try:
        token = json.loads(token)['key']
        print('parsed token from legacy kaggle.json')
    except Exception as e:
        print('WARNING: token looks like JSON but could not parse:', e)
elif not token.startswith('KGAT_'):
    print('WARNING: token does not start with KGAT_; new Kaggle API tokens should.')

# kagglehub uses the new bearer token via KAGGLE_API_TOKEN or ~/.kaggle/access_token.
# Do NOT set KAGGLE_KEY/KAGGLE_USERNAME with a KGAT token (that sends Basic auth and gives 401).
os.environ['KAGGLE_API_TOKEN'] = token

kaggle_dir = pathlib.Path('/root/.kaggle')
kaggle_dir.mkdir(exist_ok=True)
(kaggle_dir / 'access_token').write_text(token)
os.chmod(kaggle_dir / 'access_token', 0o600)

PROJECT_DRIVE = {repr(project_drive)}
os.makedirs(os.path.join(PROJECT_DRIVE, 'code'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DRIVE, 'models'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DRIVE, 'submissions'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DRIVE, 'plots'), exist_ok=True)

import kagglehub
comp_path = kagglehub.competition_download(PAYLOAD['competition'])
print('competition data downloaded to', comp_path)

base = comp_path
if os.path.isdir(os.path.join(comp_path, 'MAGFiLO_1.0_Kaggle_2026', 'train')):
    base = os.path.join(comp_path, 'MAGFiLO_1.0_Kaggle_2026')
elif not os.path.isdir(os.path.join(comp_path, 'train')):
    for d in os.listdir(comp_path):
        if os.path.isdir(os.path.join(comp_path, d, 'train')):
            base = os.path.join(comp_path, d)
            break

os.environ['FILAMENT_BASE_PATH'] = base
print('FILAMENT_BASE_PATH set to', base)
"""
    nb.cells.append(nbformat.v4.new_code_cell(setup_cell))

    nb.cells.append(nbformat.v4.new_markdown_cell("## Write the project code modules"))
    nb = _write_modules_cells(nb, code_dir, project_drive)

    nb.cells.append(nbformat.v4.new_markdown_cell("## Train and generate submission"))
    train_cell = f"""import os, sys
sys.path.append(os.path.join(PROJECT_DRIVE, 'code'))

os.environ['FILAMENT_TRAIN_RES'] = str(PAYLOAD['train_res'])
os.environ['FILAMENT_INFERENCE_RES'] = str(PAYLOAD['inference_res'])
os.environ['FILAMENT_ENCODER'] = PAYLOAD['encoder']
os.environ['FILAMENT_MODEL_NAME'] = PAYLOAD.get('model_name', 'unet')
os.environ['FILAMENT_BATCH_SIZE'] = str(PAYLOAD['batch_size'])
os.environ['FILAMENT_ACCUMULATION'] = str(PAYLOAD['accumulation'])
os.environ['FILAMENT_EPOCHS'] = str(PAYLOAD['epochs'])
os.environ['FILAMENT_LR'] = str(PAYLOAD['lr'])
os.environ['FILAMENT_MIN_LR'] = str(PAYLOAD['min_lr'])
os.environ['FILAMENT_PATIENCE'] = str(PAYLOAD['patience'])
os.environ['FILAMENT_N_FOLDS'] = str(PAYLOAD['n_folds'])
os.environ['FILAMENT_VAL_FOLD'] = str(PAYLOAD['val_fold'])
os.environ['FILAMENT_SEED'] = str(2026 + PAYLOAD['account_id'])
os.environ['FILAMENT_IN_CHANNELS'] = '3'
os.environ['FILAMENT_USE_AMP'] = '1'
os.environ['FILAMENT_USE_EMA'] = '1'
os.environ['FILAMENT_USE_WARMUP'] = '1'
os.environ['FILAMENT_GRADIENT_CHECKPOINTING'] = '1'
os.environ['FILAMENT_VAL_METRIC'] = 'mIoU_multiscale'
os.environ['FILAMENT_TTA_D4'] = '1'
os.environ['FILAMENT_TTA_SCALES'] = '0.8,1.0,1.25'
os.environ['FILAMENT_NUM_WORKERS'] = '2'

import config, train, infer

best = train.train_fold(
    fold=config.VAL_FOLD,
    epochs=config.EPOCHS,
    save_dir=os.path.join(PROJECT_DRIVE, 'models'),
)

sub = infer.run_test_submission(
    checkpoint_path=best,
    output_csv=os.path.join(PROJECT_DRIVE, 'submissions', 'submission.csv'),
    image_size=config.TRAIN_RES,
    threshold=PAYLOAD['threshold'],
    use_tta=PAYLOAD['tta'],
)
print('submission shape:', sub.shape)
print(sub.head())
"""
    nb.cells.append(nbformat.v4.new_code_cell(train_cell))

    nb.cells.append(nbformat.v4.new_markdown_cell("## Download results"))
    download_cell = f"""import os, glob, shutil
from google.colab import files

if PAYLOAD.get('download_results', True):
    result_dir = os.path.join(PROJECT_DRIVE, 'results_account_{payload['account_id']}')
    os.makedirs(result_dir, exist_ok=True)

    model_files = glob.glob(os.path.join(PROJECT_DRIVE, 'models', '*'))
    submission_files = glob.glob(os.path.join(PROJECT_DRIVE, 'submissions', '*'))

    if not model_files:
        print('WARNING: no model files found; training or inference may have failed.')
    if not submission_files:
        print('WARNING: no submission files found; inference may have failed.')

    for p in model_files:
        shutil.copy(p, result_dir)
    for p in submission_files:
        shutil.copy(p, result_dir)

    if not os.listdir(result_dir):
        raise RuntimeError(
            f"No results to download. Check that training and inference ran successfully."
        )

    zip_path = f'/content/account_{payload['account_id']}_results.zip'
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', result_dir)
    files.download(zip_path)
    print('downloaded', zip_path)
"""
    nb.cells.append(nbformat.v4.new_code_cell(download_cell))

    return nb


def main():
    parser = argparse.ArgumentParser(
        description="build per-account, no-Drive Colab payloads"
    )
    parser.add_argument(
        "--accounts", type=int, default=3, help="number of accounts/folds to generate"
    )
    parser.add_argument("--n-folds", type=int, default=5, help="number of CV folds")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder for payload JSONs and .ipynb files",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=None,
        help="encoder list to rotate across accounts (default: tu-efficientnet_b3)",
    )
    parser.add_argument(
        "--kaggle-username",
        default="srikrishnaos",
        help="Kaggle username for kagglehub auth",
    )
    parser.add_argument(
        "--kaggle-token",
        default="",
        help="Kaggle API access token (optional). If omitted, the notebook will rely on the Colab secret KAGGLE_API_TOKEN.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or (project_root / "notebooks" / "colab_payloads")
    out_dir.mkdir(parents=True, exist_ok=True)
    code_dir = project_root / "code"

    for i in range(1, args.accounts + 1):
        payload = build_payload(i, args.n_folds)
        if args.encoders:
            payload["encoder"] = args.encoders[(i - 1) % len(args.encoders)]

        json_path = out_dir / f"account_{i}_payload_nodrive.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        nb = build_notebook(payload, code_dir, args.kaggle_username, args.kaggle_token)
        nb_path = out_dir / f"account_{i}_payload_nodrive.ipynb"
        nbformat.write(nb, nb_path)

        print(f"wrote account {i}: {json_path} and {nb_path}")


if __name__ == "__main__":
    main()
