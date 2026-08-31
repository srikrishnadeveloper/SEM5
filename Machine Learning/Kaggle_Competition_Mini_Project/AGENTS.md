# AGENTS.md — Solar Filament Segmentation Challenge 2026

> Project root: `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project`  
> OS: Windows (PowerShell / Python 3.11+)  
> Do **not** commit, push, or paste the Kaggle API token into any tracked file.

---

## 1. Project overview and goal

This is the **Solar Filament Segmentation Challenge 2026** (Kaggle + IEEE BigData Cup) mini-project. The task is to predict a pixel-precise binary mask for every solar filament in 2048×2048 full-disk H-alpha images from the MAGFiLO v1.0 dataset and to submit per-filament COCO RLE strings in a single CSV.

The local pipeline is in `code/` and uses:

- `segmentation-models-pytorch` for the segmentation model (U-Net / UNet++ / DeepLabV3+ / FPN with `timm` or torchvision encoders).
- Training at a lower resolution (default `1024`) with random filament-centred cropping and heavy Albumentations augmentation.
- Inference at the full `2048` resolution: resize probability map up with bilinear interpolation, then threshold and post-process.
- Post-processing with solar disk masking, morphological cleanup, watershed instance splitting, and `pycocotools` RLE encoding.
- A 5-fold GroupKFold split by year to avoid multi-annotator and temporal leakage.

The repo has no `.git` yet. **Do not commit or push** anything, and **do not write the Kaggle API token** into any source file.

---

## 2. Local file tree

```
C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project
├── code/                              # all source modules
│   ├── config.py
│   ├── dataset.py
│   ├── infer.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── postprocess.py
│   ├── submit.py
│   ├── train.py
│   ├── __pycache__/
│   └── models/                        # .pth checkpoints + per-fold CSV logs
├── data/
│   ├── filament-segmentation-2026.zip       # original competition archive (~671 MB)
│   └── MAGFiLO_1.0_Kaggle_2026/
│       ├── train/
│       │   ├── train_images/                # 707 JPEGs, 2048×2048
│       │   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│       └── test/
│           └── test_images/                 # 180 JPEGs, 2048×2048
├── notebooks/
│   ├── build_colab_nb.py                    # Google-Drive Colab notebook builder
│   ├── build_colab_nodrive_payloads.py      # multi-account no-Drive payload builder
│   ├── build_kaggle_nb.py                   # Kaggle notebook builder
│   ├── colab_filament_training.ipynb        # Drive-based Colab notebook
│   └── colab_payloads/
│       ├── account_1_payload_nodrive.json
│       ├── account_1_payload_nodrive.ipynb
│       ├── account_2_payload_nodrive.json
│       └── account_2_payload_nodrive.ipynb
├── research/                          # background notes and literature review
│   ├── TOP50_PLAN.md                  # master reference for top-50 strategy
│   └── TOP50_ACTION_PLAN.md           # code audit + step-by-step implementation plan
├── plots/                             # generated plots (empty by default)
├── submissions/                       # generated submission / OOF CSVs (empty by default)
├── requirements.txt
├── WORKLOG.md
└── AGENTS.md
```

---

## 3. Common commands

### 3.1 Install dependencies

Use a Python 3.11+ venv/conda. `pycocotools` often needs the Microsoft C++ Build Tools on Windows, or the prebuilt `pycocotools-windows` wheel.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# if pycocotools fails on Windows:
# pip install pycocotools-windows
```

### 3.2 Local training

From inside the `code/` directory (or set `PYTHONPATH` to `code`):

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\code"
$env:FILAMENT_TRAIN_RES="1024"
$env:FILAMENT_INFERENCE_RES="2048"
$env:FILAMENT_ENCODER="tu-efficientnet_b3"
$env:FILAMENT_BATCH_SIZE="2"
$env:FILAMENT_EPOCHS="50"
$env:FILAMENT_VAL_FOLD="0"
python train.py
```

`train.py` calls `train_fold()` which:

- loads the assigned `VAL_FOLD`,
- writes a metrics CSV to `code/models/fold_{VAL_FOLD}_metrics.csv`,
- saves the best checkpoint to `code/models/best_fold_{VAL_FOLD}.pth` based on validation Panoptic Quality.

### 3.3 Out-of-fold (OOF) validation

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\code"
$env:FILAMENT_VAL_FOLD="0"
$env:FILAMENT_TRAIN_RES="1024"
python submit.py --val-fold 0 --image-size 1024 --tta
```

This produces `submissions/oof_fold_0.csv` for local PQ/mIoU analysis.

### 3.4 Final test submission

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\code"
$env:FILAMENT_TRAIN_RES="1024"
python submit.py --image-size 1024 --tta --out "..\submissions\submission_test.csv"
```

If `--out` is omitted, the CSV is written to `submissions/submission_best_fold_{fold}.pth.csv` (resolved from `config.SUBMISSIONS_DIR`).

### 3.5 Sanity checks / "tests"

There is no `pytest` suite yet. Each module has a small `if __name__ == "__main__"` block:

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\code"
python config.py
python model.py
python metrics.py
python dataset.py
python postprocess.py
python losses.py
```

If you add a formal test suite, put it in `tests/` and run `pytest tests/`.

### 3.6 Lint / typecheck

No `ruff`, `mypy`, or `pytest` config is committed. Useful ad-hoc checks:

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project"
python -m py_compile code\*.py
# if installed:
# ruff check code/
# mypy code/
```

### 3.7 Build the lab report PDF

Use the college `lab-report-pdf` skill. Machine Learning reports get the **polished** style (`plain: false`).

1. Capture real output and plots from a code file:

```powershell
python "C:\Users\srik2\Desktop\College\.opencode\skills\lab-report-pdf\scripts\capture_run.py" `
    "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\code\metrics.py" `
    --cwd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\code" `
    --out-dir "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\report\capture"
```

2. Write a `report/spec.json` using the schema in `C:\Users\srik2\Desktop\College\.opencode\skills\lab-report-pdf\references\spec_schema.md`.

3. Validate and build:

```powershell
python -c "import json; json.load(open('report/spec.json'))"
python "C:\Users\srik2\Desktop\College\.opencode\skills\lab-report-pdf\scripts\build_pdf.py" `
    "report\spec.json" `
    --out "Assignment_Kaggle_Solar_Filament_Srikrishna_O_S.pdf"
```

Student metadata (name/class/regno) is read from `C:\Users\srik2\.lab_report_profile.json` unless provided in the spec.

---

## 4. Colab / multi-account setup notes

### 4.1 Generated no-Drive payloads

`notebooks/build_colab_nodrive_payloads.py` creates per-account, no-Drive Colab payloads under `notebooks/colab_payloads/`:

- `account_{n}_payload_nodrive.json` — per-account config (fold, encoder, resolution, batch size, etc.).
- `account_{n}_payload_nodrive.ipynb` — self-contained Colab notebook that does **not** mount Google Drive.

Generate them with:

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\notebooks"
python build_colab_nodrive_payloads.py --accounts 3

# optionally rotate encoders across accounts:
# python build_colab_nodrive_payloads.py --accounts 3 `
#   --encoders tu-efficientnet_b3 tu-resnet50 tu-efficientnet_b4
```

### 4.2 How the no-Drive payloads work

1. **Runtime & secret setup.** The first cells of the `.ipynb` tell the user to select a **T4 GPU** and to create a Colab secret named `KAGGLE_API_TOKEN` containing the full `kaggle.json` contents.
2. **Install.** It installs `torch==2.5.1+cu118`, `segmentation-models-pytorch`, `albumentations`, and supporting packages (`pycocotools`, `scikit-image`, `scipy`, etc.).
3. **Download data.** It writes the secret to `/root/.kaggle/kaggle.json` and uses `kagglehub.competition_download('filament-segmentation-2026')` to download the ~671 MB dataset into Colab's temporary storage. It then sets `FILAMENT_BASE_PATH` to the detected `MAGFiLO_1.0_Kaggle_2026` subfolder.
4. **Embed the payload.** The JSON config is baked into the notebook as a `PAYLOAD` Python dict, so each `.ipynb` is self-contained and the `.json` is just a human-readable record.
5. **Write code modules.** It creates `/content/filament_kaggle/code/` and writes all `code/*.py` files with `%%writefile` cells.
6. **Set `FILAMENT_*` env vars from `PAYLOAD`** before importing `config`, `train`, `infer`.
7. **Train.** `train.train_fold(..., save_dir='/content/filament_kaggle/models')` saves `best_fold_{VAL_FOLD}.pth` and `fold_{VAL_FOLD}_metrics.csv`.
8. **Submit.** `infer.run_test_submission(...)` with the payload's TTA/threshold produces `/content/filament_kaggle/submissions/submission.csv`.
9. **Download.** If `download_results` is `true` (default), it zips the model and submission and calls `google.colab.files.download` so the user can save them before the Colab session ends.

Because the notebook does **not** mount Drive, every run re-downloads the data and all result files must be downloaded before the session disconnects.

### 4.3 Set `KAGGLE_API_TOKEN` and T4 runtime

1. Go to `https://www.kaggle.com/account` → **Create New API Token**. Save the downloaded `kaggle.json`.
2. In Colab, open the **Secrets** panel (key icon on the left), add a secret named `KAGGLE_API_TOKEN`, and paste the **entire contents** of `kaggle.json` as the value. Toggle **Notebook access** on.
3. **Runtime → Change runtime type → Hardware accelerator = GPU**. Select **T4** (or higher) from the GPU shape dropdown.
4. Do **not** paste the token into any code cell. Do not share the notebook with secrets enabled.

### 4.4 Kaggle notebook

`notebooks/build_kaggle_nb.py` builds `notebooks/kaggle/filament_training.ipynb` with `kernel-metadata.json` (private, GPU enabled, attached to `filament-segmentation-2026`). It installs `torch==2.5.1+cu118` because the preinstalled Kaggle PyTorch 2.13 no longer supports the P100's `sm_60` compute capability. It writes code to `/kaggle/working/code` and saves results under `/kaggle/working`.

---

## 5. Important conventions

### 5.1 `FILAMENT_*` environment variables

`code/config.py` exposes all tunables as environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `FILAMENT_BASE_PATH` | `data\MAGFiLO_1.0_Kaggle_2026` | root containing `train/` and `test/` |
| `FILAMENT_TRAIN_RES` | `1024` | model input size (`512`, `1024`, `1536`) |
| `FILAMENT_INFERENCE_RES` | `2048` | full resolution to resize predictions back to |
| `FILAMENT_ENCODER` | `tu-efficientnet_b3` | `smp` encoder name; `tu-` = `timm` |
| `FILAMENT_MODEL_NAME` | `unet` | `unet`, `unetplusplus`, `deeplabv3plus`, `fpn` |
| `FILAMENT_BATCH_SIZE` | `2` | per-GPU batch (use `1` on T4) |
| `FILAMENT_ACCUMULATION` | `1` | gradient-accumulation steps |
| `FILAMENT_EPOCHS` | `50` | max epochs |
| `FILAMENT_LR` | `1e-3` | AdamW initial learning rate |
| `FILAMENT_MIN_LR` | `1e-5` | `CosineAnnealingLR` floor |
| `FILAMENT_WD` | `1e-4` | AdamW weight decay |
| `FILAMENT_PATIENCE` | `10` | early-stopping patience (not wired yet) |
| `FILAMENT_N_FOLDS` | `5` | CV folds |
| `FILAMENT_VAL_FOLD` | `0` | fold to validate on |
| `FILAMENT_PROB_THRESHOLD` | `0.45` | probability → binary threshold |
| `FILAMENT_MORPH_CLOSE_K` | `3` | morphology kernel size |
| `FILAMENT_MIN_AREA` | `200` | minimum filament area in px |
| `FILAMENT_MAX_AREA` | `500000` | maximum filament area in px |
| `FILAMENT_LOSS_DICE_W` | `0.4` | combined loss weight |
| `FILAMENT_LOSS_FOCAL_W` | `0.4` | combined loss weight |
| `FILAMENT_LOSS_BCE_W` | `0.2` | combined loss weight |
| `FILAMENT_SEED` | `2026` | base random seed |

### 5.2 Inference resolution

`infer.py` loads the full 2048×2048 JPEG, resizes the image to `TRAIN_RES`, runs the model, then resizes the predicted **probability map** back to 2048×2048 with bilinear interpolation and thresholds at full resolution. Thresholding first and then upscaling with `INTER_NEAREST` (the public baseline's mistake) is avoided.

### 5.3 `mIoU_pairwise` and `mIoU_multiscale`

`code/metrics.py` implements both the public-baseline Dice/PQ helpers and the competition-style metrics from the EdgeAttNet / Multiscale IoU literature:

- `mIoU_pairwise`: for every overlapping `(gt_i, pred_j)` pair compute standard IoU and average over those pairs. Pairs with no overlap are excluded.
- `mIoU_multiscale`: extract Sobel edges, downsample with box-counting at cell sizes `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]`, compute the intersection ratio `r = |s(gt) ∩ s(pred)| / |s(gt)|` with `+1` smoothing, integrate over scales with the trapezoidal rule, and average over overlapping pairs.
- `mean_mIoU_pairwise` / `mean_mIoU_multiscale` compute macro and micro averages over a list of images.

`train.py` currently selects the best checkpoint by the **relaxed Panoptic Quality** in `validate()`. This is a known TODO: consider switching model selection to `mIoU_pairwise` / `mIoU_multiscale` once the validation loop is updated.

### 5.4 Model encoder names

`code/model.py` uses `segmentation_models_pytorch`.

- `tu-` prefix (e.g. `tu-efficientnet_b3`, `tu-resnet50`, `tu-mobilenetv3_large_100`, `tu-densenet121`) comes from the `timm` encoder catalog.
- Non-`tu` names (e.g. `resnet34`, `resnet50`, `efficientnet-b0`) are torchvision encoders supported by `smp`.
- The model always uses `in_channels=1`; `smp` adapts the first convolution for grayscale.

Valid `MODEL_NAME` values from `model.py`: `unet`, `unetplusplus`, `deeplabv3plus`, `fpn`.

### 5.5 Data split

`code/dataset.py` groups COCO annotations by **physical filename**, not by `image_id`, to avoid leakage from the multi-annotator setup (1,154 `image_id`s but only 707 files). It uses `GroupKFold` with the year prefix of the filename as the group, so all images from the same year stay in the same fold. Alternative splits in `dataset.py`: `split_by_year` and a random 85/15 split.

---

## 6. Where outputs go

| Output | Default path |
|---|---|
| best checkpoint | `code/models/best_fold_{VAL_FOLD}.pth` |
| per-epoch metrics CSV | `code/models/fold_{VAL_FOLD}_metrics.csv` |
| OOF validation CSV | `submissions/oof_fold_{VAL_FOLD}.csv` |
| final test submission | `submissions/submission_best_fold_{VAL_FOLD}.pth.csv` (or `--out` path) |
| plots / EDA | `plots/` |
| Colab Drive output (Drive notebook) | `/content/drive/MyDrive/filament_kaggle/` |
| Colab no-Drive output | `/content/filament_kaggle/models/` and `submissions/` |
| Kaggle output | `/kaggle/working/` |
| no-Drive downloadable zip | `/content/account_{n}_results.zip` |

---

## 7. Known pitfalls and manual steps

- **Kaggle API token:** create it at `https://www.kaggle.com/settings/api` and keep it secret. Do not commit it or paste it into shared source code. The Colab `KAGGLE_API_TOKEN` secret should contain the **entire `kaggle.json` contents**, or the notebook can set `KAGGLE_USERNAME` + `KAGGLE_KEY` from the token string.
- **Ready-to-run no-Drive notebooks:** the three final payload notebooks live in `C:\Users\srik2\Desktop\Filament_Colab_Run\` (fast/medium/strong configs). They embed the Kaggle token as a fallback, so each account only needs to upload the `.ipynb`, set **Runtime → Change runtime type → T4**, and choose **Runtime → Run all**. Delete the local and Colab copies after running and revoke the token.
- **Competition data paths:** Kaggle input can live at `/kaggle/input/competitions/filament-segmentation-2026/...` or `/kaggle/input/filament-segmentation-2026/...`; `build_kaggle_nb.py` searches both. Colab `kagglehub.competition_download` returns a nested folder; the no-Drive payload and `config.py` auto-detect the `MAGFiLO_1.0_Kaggle_2026` subfolder.
- **P100 CUDA compatibility:** Kaggle's free P100 is `sm_60`, but the preinstalled Kaggle PyTorch 2.13 only supports `sm_70+`. The Kaggle notebook installs `torch==2.5.1+cu118`, which still ships `sm_60` binaries.
- **T4 memory:** a free Colab T4 (~16 GB) can usually train `tu-efficientnet_b3` at `TRAIN_RES=1024`, `BATCH_SIZE=1`, `ACCUMULATION=2` with `torch.amp`. Do not use `1536`+ or larger encoders without gradient checkpointing or a smaller backbone.
- **Colab Drive vs no-Drive:** the Drive notebook persists to Google Drive but needs a one-time `drive.mount`. The no-Drive payloads re-download ~671 MB each run; results must be downloaded before the session expires.
- **Playground mode:** when opening a shared Colab link, use normal notebook mode if you need to set secrets or download results. In Kaggle, attach the notebook to the competition, not a standalone dataset, so `kagglehub` accepts the competition download.
- **Windows `pycocotools`:** `pycocotools` needs a C compiler on Windows or the `pycocotools-windows` wheel. `scikit-image` is now in `requirements.txt` because `postprocess.py` uses watershed and `peak_local_max`.
- **Ambiguous category:** `dataset.py` drops `category_id == 4` (`Ambiguous`) because it has no annotations and only adds confusion.
- **Multi-annotator leakage:** the COCO JSON has 1,154 `image_id`s but only 707 physical JPEGs. Splits are done by filename.
- **RLE format:** `postprocess.py` encodes each instance with `pycocotools.mask.encode` on a Fortran-ordered `(2048, 2048)` `uint8` mask and writes only the `counts` string. The submission CSV must have exactly the columns `filament_id,segmentation_rle`.
- **Evaluation metric mismatch:** the public brief describes a 70 % quantitative / 30 % qualitative split with Mean Dice + Panoptic Quality, but research (`research/verification_and_recommendations.md`) indicates the real Kaggle/IEEE BigData Cup scoring is likely `mIoU_pairwise` and `mIoU_multiscale` (EdgeAttNet). The code supports both; the training selection metric may need to be swapped to mIoU later.

---

## 8. Top-50 research and battle plan

A full literature review and phased top-50 battle plan has been compiled in `research/TOP50_PLAN.md`.  It covers the true competition metrics (MIoU, pairwise IoU, AP), model architectures (EdgeAttNet, U-Net++, DeepLabV3+, SegFormer), encoders, losses, augmentations, TTA/ensembling, threshold/area search, post-processing, pseudo-labeling, and a concrete account-by-account next-config recommendation.  Use it as the master reference for all future decisions.

A second document, `research/TOP50_ACTION_PLAN.md`, contains the **line-by-line code audit** and **step-by-step implementation checklist** for the current pipeline.  It lists the 26 issues found in the code, the exact fixes, public resources, and a phase-by-phase plan for reaching the top 50.

---

## 9. Single-file pipeline (preferred entry point)

`notebooks/filament_top50_single.py` is the whole pipeline in one file, and
`notebooks/build_single_notebook.py` wraps it into one self-contained Colab
notebook at `C:\Users\srik2\Desktop\Filament_Colab_Run\filament_top50_single.ipynb`.

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\notebooks"
python build_single_notebook.py --out-dir "C:\Users\srik2\Desktop\Filament_Colab_Run"
```

### API contract (important)
- `apply_preset(CFG, "balanced")` sets `arch/encoder/img_size/batch_size/accum/epochs`. **Call it first.**
- `resolve(CFG)` only derives paths and is idempotent, so it never overwrites manual edits.
- `run(cfg)` calls `resolve()` internally, **not** `apply_preset()`. Manual overrides survive.
- Presets: `fast` (b2@768, ~35 min), `balanced` (b4@1024, ~70 min), `max` (UNet++ b4@1024, ~3 h) per fold on a T4.

### Metric warning
`miou_multiscale_pair()` applies greedy one-to-one matching and an F1 penalty
`2*n_matched/(n_gt+n_pred)`. Do **not** remove the penalty: without it the metric
is recall-only, and both checkpoint selection and the threshold search will
prefer fragmented/over-predicted masks that score badly on the leaderboard.
Sanity values: perfect `1.0`, one giant blob `~0.11`, 20 fragments `~0.03`,
empty-vs-empty `1.0`.

### Local testing
Nothing was installed locally for a long time, which let real crashes reach
Colab. The dependency stack is now installed (torch CPU, `segmentation-models-pytorch`,
`albumentations`, `pycocotools`, `scikit-image`). Always smoke-test before
uploading a notebook: build a small synthetic MAGFiLO-shaped dataset and run
`run(cfg, do_train=True, do_search=True, do_submit=True)` end to end. Images must
stay `uint8` through Albumentations because `CLAHE`/`ISONoise` reject float input.

**Use `tests/` before spending Colab time.** Two cheap, project-local scripts
catch most bugs before a real GPU run:
- `python tests/test_unit_fixes.py` — pure-function regression checks, <1s.
- `python tests/smoke_synthetic.py` — full pipeline on tiny synthetic images
  covering `unet` default, `unet` + copy-paste + ImageNet-norm + CRF toggle,
  and `edgeattnet`. Runs in well under a minute on CPU.
Run both after ANY change to `filament_top50_single.py` or `build_single_notebook.py`,
and before every `build_single_notebook.py` regeneration that will be uploaded
to Colab. Do NOT reach for a real-2048px-image smoke test as the first line of
defense — it is much slower (real Hough/radial-flatten preprocessing) for no
extra bug-catching power; only use real data for a final sanity pass.

### EdgeAttNet preset and friend-branch notes

- New single-file preset: `edgeattnet` (`UNetEdgeTransformer`, 512x512, 40 epochs, batch 4 / accum 4, ~2 h on T4). It uses no ImageNet pretraining and benefits from the competition-aware `mIoU_multiscale_pair` metric.
- `arch` values in the single-file notebook: `unet`, `unetplusplus`, `fpn`, `deeplabv3plus`, and `edgeattnet`.
- `decoder_attention = "scse"` is supported for `unet`/`unetplusplus`; friend branch `solaris_pipeline.py` uses the same `scse` attention.
- Friend branch (`solar-filament-segmentation-boobathi_branch`) uses `UnetPlusPlus/efficientnet-b4` at 512, 4-flip TTA, 5-fold ensemble, and pixel-Dice threshold search. Their OOF dice is reported as ~0 at threshold 0.8, which is consistent with thresholding on the wrong metric. Our search uses the real multiscale instance metric.

### Building and verifying the single notebook

```powershell
cd "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\notebooks"
python -m py_compile filament_top50_single.py
python build_single_notebook.py --out-dir "C:\Users\srik2\Desktop\Filament_Colab_Run"
python "C:\Users\srik2\AppData\Local\Temp\opencode\verify_notebook.py"
```

### Colab run (single notebook)

1. Upload `C:\Users\srik2\Desktop\Filament_Colab_Run\filament_top50_single.ipynb`.
2. Runtime -> Change runtime type -> T4 GPU.
3. Secrets -> `KAGGLE_API_TOKEN` = full `kaggle.json` contents.
4. Cell 4: choose `preset = "balanced"` or `"max"` or `"edgeattnet"` and `fold = 0..4`.
5. Runtime -> Run all.
6. Download the model, submission, and probability maps before the session expires.

---

## 10. V3 "Beast Mode" Production Engine (`v3/`)

`v3/` contains the maximum-performance production pipeline:
- **Location:** `v3/pipeline.py`, `v3/build_notebook.py`, `v3/test_pipeline.py`.
- **Standalone Notebook:** `C:\Users\srik2\Desktop\Filament_Colab_Run\v3_beast_solar_filament_pipeline.ipynb`.
- **Key Specifications:**
  - **Resolution:** 1024×1024 high resolution (effective batch size 8 with `batch_size=2`, `accum_steps=4`).
  - **Heterogeneous Ensemble:** 10 models (5 folds `UnetPlusPlus/efficientnet-b4/scse` + 5 folds `DeepLabV3Plus/resnet50/ASPP`).
  - **Astronomical 3-Channel Map:** Raw H-alpha + CLAHE + High-pass unsharp ridge filter.
  - **Loss:** Quad-Hybrid (0.25 BCE + 0.30 SoftDice + 0.25 Focal + 0.20 Lovasz-Hinge).
  - **Inference:** 40-pass 4-flip TTA with weighted blending ($0.55 / 0.45$) + 2048px bilinear upsample.
  - **Post-Processing:** Optical solar limb suppression + morphological closing + anti-fragmentation area filter ($150 - 750$ px).
- **VRAM Hardening Rule:** Always use `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, `batch_size=2` at 1024px, and call `del model, optimizer` with `torch.cuda.empty_cache()` between folds.

---

## 11. V4 "Apex Grandmaster" Production Engine (`v4/`)

`v4/` contains the apex competitive pipeline:
- **Location:** `v4/pipeline.py`, `v4/build_notebook.py`, `v4/build_standalone_infer_nb.py`, `v4/infer_standalone.py`.
- **Standalone Notebook:** `C:\Users\srik2\Desktop\Filament_Colab_Run\v4_grandmaster_solar_filament_pipeline.ipynb`.
- **Key Specifications:**
  - **Tri-Architecture Ensemble:** UnetPlusPlus (`efficientnet-b4`) + SegFormer Transformer (`mit_b3`) + DeepLabV3Plus (`resnet50`).
  - **Penta-Hybrid Loss:** 0.20 BCE + 0.25 Dice + 0.20 Focal + 0.20 Lovasz + 0.15 Differentiable Sobel Boundary Edge Loss.
  - **Watershed Instance Separation:** Marker-controlled distance transform separation of touching/intersecting solar filaments.
  - **Native 2048px Gaussian Tiling:** Zero seam artifacts with 2D Gaussian apodization blending window.
  - **Single-Device Memory Safety:** Single CUDA device execution with explicit `del logits, loss, images, masks` per batch (<2 GB Host RAM, zero thread lockups).

---

## 12. Modular Checkpoint Fast Inference Engine

- **Standalone Inference Notebook:** `C:\Users\srik2\Desktop\Filament_Colab_Run\standalone_inference_pipeline.ipynb`.
- **CLI Engine:** `v4/infer_standalone.py`.
- **Purpose:** Decouples training from inference. Loads any trained `.pth` checkpoint in 5 seconds and generates `submission.csv` with full 4-flip TTA, 2048px upsampling, and watershed separation in **<90 seconds**.
- **Kaggle Execution Rule:** Never wrap models in `nn.DataParallel` inside Jupyter notebooks to avoid host CPU RAM accumulation.



