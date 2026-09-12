# GROK ADDENDUM — Kaggle Dual T4 + Internet ON
# Status: FROZEN EXECUTION TARGET
# Paste this into Antigravity with 04_GROK_VERDICT_PHASE_C.md if not already applied.

Human instruction: this pipeline must run the same way previous v3/v4/v7 kernels ran.
**Platform = Kaggle Notebook. Accelerator = GPU T4 x2. Internet = ON.**
Local Windows CPU is only for dry-run / unit tests. Full train and test inference happen on Kaggle.

---

## Kernel settings (human clicks these)
- New notebook or reuse `V8_1_Kaggle_Production.ipynb`
- Accelerator: **GPU T4 x2**
- Internet: **ON** (needed for `pip install ultralytics` and `yolo11s-seg.pt`)
- Persistence: save output
- Add data:
  - Competition: `filament-segmentation-2026`
  - Existing datasets if present: `solar-filament-9models`, `grandmaster_model` (refiner fallback only)
- Do not attach the full MAGFiLO Dataverse.

## Path map
```
/kaggle/input/filament-segmentation-2026/   # official train/test + json
/kaggle/working/                            # all writes
/kaggle/working/submission.csv              # MUST be this path for submit
```
Detect Kaggle vs local:

```python
from pathlib import Path
KAGGLE = Path("/kaggle/input").exists()
DATA = Path("/kaggle/input/filament-segmentation-2026") if KAGGLE else Path("data/MAGFiLO_1.0_Kaggle_2026")
OUT  = Path("/kaggle/working") if KAGGLE else Path(".")
```

Do not hardcode `C:\Users\srik2\...` anywhere in v8_1 or the notebook.

## Install block (first cell, internet ON)
```
import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
%pip -q install ultralytics segmentation-models-pytorch pycocotools albumentations timm
```
No Drive. No Colab-only paths. Flush prints. Console tqdm, not notebook widgets.

## Dual T4 rules (same as previous kernels)
- `n_gpus = torch.cuda.device_count()`  # expect 2
- **Never** `nn.DataParallel`
- YOLO train: `device=0,1` if n_gpus==2 else `device=0` (ultralytics DDP)
- Crop refiner train: single GPU `cuda:0` is enough
- Infer cascade model-parallel:
  - YOLO on `cuda:0`
  - crop U-Net on `cuda:1` if present else `cuda:0`
- After each stage: `del`, `torch.cuda.empty_cache()`
- Log `torch.cuda.memory_reserved(i)` per device, not only allocated

## What the Kaggle notebook must do, in order
1. Install + print GPU names and VRAM
2. Copy or import `metrics/`, `scripts/`, `v8_1/` into `/kaggle/working` (or define them in cells)
3. If `data/yolo_seg` missing: run `convert_coco_to_yolo.py` against Kaggle input paths
4. `1_train_yolo.py` — yolo11s-seg @ 1024, 20 epochs, AMP, batch 4 accum 2 (OOM → batch 2)
5. Print unique-file val `pq_mean` / `pq_max` at conf 0.15/0.25/0.35
6. `2_train_crop_refiner.py` on train-fold GT crops (or YOLO boxes if best.pt exists)
7. `3_infer_cascade.py` on the 180 test JPEGs, `--residual 0` first pass
8. Write `/kaggle/working/submission.csv`
9. Print n_rows, unique stems (must be 180), empty-row count, rows/image mean

Total wall time budget: under 4 hours. If YOLO will not finish, cut to 12 epochs, do not add a 14-model ensemble to fill time.

## Submit contract
- File: `/kaggle/working/submission.csv`
- Columns: `filament_id,segmentation_rle`
- 180 stems exactly, at least one row each
- Empty stem → `{stem}_1` + `encode_mask(zeros(2048,2048))`
- No `PPP2` literal

## Still do first on local CPU
`python v8_1/1_train_yolo.py --dry-run` must exit 0 and print the frozen config.
Then human uploads the notebook to Kaggle and runs it. Antigravity must not pretend a local CPU 20-epoch train is the run.

## Footer when done
Show the notebook cell outline + dry-run stdout. Do not claim a LB score.
