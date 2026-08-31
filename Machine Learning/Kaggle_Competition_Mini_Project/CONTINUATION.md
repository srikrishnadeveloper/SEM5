# Solar Filament Segmentation 2026 — Project Runbook

This document is a single source of truth for continuing this project with another agent. It describes what has been attempted, the current state, what works, what failed, and how to proceed.

---

## 1. Project Layout

```
C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project
├── data\MAGFiLO_1.0_Kaggle_2026\     # local copy of the competition data (not used on Kaggle)
├── notebooks\filament_top50_single.py  # the full single-file pipeline
├── notebooks\kaggle\                    # Kaggle notebook output folder
│   ├── filament_5fold_ensemble.ipynb      # v2 notebook (currently stuck on Kaggle)
│   ├── filament_5fold_ensemble_v3.ipynb   # v3 (balanced, robust, pushed, but not run)
│   ├── filament_5fold_ensemble_v4_fast.ipynb  # v4 (fast preset, built, not pushed)
│   └── kernel-metadata.json               # metadata for the next push (currently v4)
├── submission.csv                       # previous local single-fold submission (if any)
└── run_single.py                        # old local run script
```

Temp files (build scripts, logs, polling scripts):

```
C:\Users\srik2\AppData\Local\Temp\poll_kaggle_v2.py
C:\Users\srik2\AppData\Local\Temp\build_kaggle_5fold.py
C:\Users\srik2\AppData\Local\Temp\build_kaggle_5fold_v3.py
C:\Users\srik2\AppData\Local\Temp\build_kaggle_v4_fast.py
C:\Users\srik2\AppData\Local\Temp\kaggle_v2_status.log
C:\Users\srik2\AppData\Local\Temp\kaggle_memory.md
```

Kaggle API token:

```
C:\Users\srik2\.kaggle\kaggle.json
C:\Users\srik2\.kaggle\access_token  # fixed from BOM / UTF-16 issues
```

---

## 2. The Core Pipeline

### File: `notebooks/filament_top50_single.py`

This is the only Python file that matters for training/inference. It contains:

- `CFG` class with all hyperparameters
- `apply_preset()` — selects fast / balanced / max / edgeattnet
- `build_cache()` — decodes JPEGs and masks into `.npy`/`.memmap`
- `build_model()` — creates U-Net/UNet++/EdgeAttNet with SMP
- `train_fold()` — training loop with AMP, EMA, gradient accumulation, early stopping
- `write_oof_probs()` — full-res out-of-fold probability maps for threshold search
- `build_submission()` — test inference, post-processing, RLE encoding
- `ensemble_probs()` — average probability maps across folds
- `submission_from_prob_dir()` — generate `submission.csv` from an averaged probability directory

### Presets (in `filament_top50_single.py`)

```python
PRESETS = {
    "fast":     dict(arch="unet",         encoder="timm-efficientnet-b2", img_size=768,  batch_size=4, accum=2, epochs=20, tta_scales=(1.0,)),
    "balanced": dict(arch="unet",         encoder="timm-efficientnet-b4", img_size=1024, batch_size=2, accum=4, epochs=25, tta_scales=(0.9, 1.0, 1.1)),
    "max":      dict(arch="unetplusplus", encoder="timm-efficientnet-b4", img_size=1024, batch_size=1, accum=8, epochs=30, tta_scales=(0.85, 1.0, 1.15)),
    "edgeattnet": dict(arch="edgeattnet", encoder="none", img_size=512, batch_size=4, accum=4, epochs=40, tta_scales=(0.9, 1.0, 1.1), in_channels=1),
}
```

### Important CLI entry points

The `if __name__ == "__main__":` block at the bottom of `filament_top50_single.py` starts training **immediately** with default settings. **This must be stripped before embedding into a Kaggle notebook**, or the notebook will start training as soon as the source cell is executed.

```
# Kaggle v1 failed because this block was not removed.
# v2/v3/v4 builders all strip this with a regex.
```

### `run(cfg, ...)` function

Called by the Kaggle notebooks. It:
1. Creates cache
2. Builds model
3. Trains fold
4. Runs OOF threshold search
5. Builds `submission.csv` for the single fold

Returns a dict with `best_path`, `threshold`, `min_area`.

---

## 3. What Has Been Attempted

### Camber (free GPU L4)

- Job 24307 failed due to missing `cv2`.
- Job 24309 consumed the entire 5-hour Camber GPU quota and was cancelled.
- Network issues with `xdg-open` and Camber CLI were common.
- **Result:** no submission. Moved to Kaggle.

### Kaggle v1 — `Filament 5 Fold Ensemble`

- First Kaggle notebook.
- **Failure:** within 3 minutes because the `if __name__ == "__main__"` block was still present and immediately started training before the config cell could run.
- URL: https://www.kaggle.com/code/srikrishnaos/filament-5-fold-ensemble

### Kaggle v2 — `Filament 5 Fold V2`

- URL: https://www.kaggle.com/code/srikrishnaos/filament-5-fold-v2
- Fixed the `__main__` issue.
- Used `balanced` preset.
- P100 GPU.
- **Status:** pushed and started. Currently **stuck/hung**.
- **Runtime:** 11+ hours.
- **Output:** 0 B.
- **Last log:** finished install and started `FOLD 0/4` after ~3 minutes; no epoch/progress output since.
- **Diagnosis:** not training. Likely hung on pretrained weight download, model init, or first DataLoader batch.
- **Will not finish:** Kaggle 12-hour hard limit will kill it soon.

### Kaggle v3 — `Filament 5 Fold V3`

- URL: https://www.kaggle.com/code/srikrishnaos/filament-5-fold-v3
- Built with per-fold `try/except` and ensemble fallback.
- Uses `balanced` preset.
- Pushed successfully but **not run** (because v2 was already consuming the GPU).
- Builder: `C:\Users\srik2\AppData\Local\Temp\build_kaggle_5fold_v3.py`
- **Note:** v3 is still too slow to finish on a P100 in 12h.

### Kaggle v4 — `Filament 5 Fold V4 Fast`

- URL (intended): https://www.kaggle.com/code/srikrishnaos/filament-5fold-v4-fast
- Built with `fast` preset, `num_workers=0`, `PYTHONUNBUFFERED=1`, and `tqdm` forced to console output.
- Builder: `C:\Users\srik2\AppData\Local\Temp\build_kaggle_v4_fast.py`
- Generated notebook: `notebooks\kaggle\filament_5fold_ensemble_v4_fast.ipynb`
- `kernel-metadata.json` already updated to v4.
- **Status:** built, **not pushed yet** because Kaggle returned:
  ```
  Maximum batch GPU session count of 2 reached
  ```
- Needs at least one of the two currently running GPU sessions to be cancelled before it can be pushed.

---

## 4. Current State

- **Time of last check:** ~20:39 UTC, 2026-08-27
- **v2 Kaggle session:** still `RUNNING` but stuck, 11+ hours elapsed, 0 B output
- **2 GPU sessions running on Kaggle** — v2 and one other (likely v3 or an older run)
- **Local polling script for v2** (`poll_kaggle_v2.py`) was restarted, but v2 is expected to die at the 12-hour mark
- **v4 fast notebook is ready to push and run**

---

## 5. What Works

- Kaggle API token is fixed and authenticated.
- `python -m kaggle kernels push` works (when not blocked by the 2-GPU-session limit).
- `python -m kaggle kernels status <id>` works (when network is stable).
- `python -m kaggle kernels output <id> -p <dir>` works to download artifacts.
- The v4 notebook builds correctly and strips the `__main__` block.

---

## 6. What Is Wrong / Known Issues

### v2 is hung, not slow
- No progress output after fold 0 start.
- `balanced` preset with `timm-efficientnet-b4` at 1024px is too heavy for P100 inside Kaggle's 12h limit.
- Even if not hung, 5 folds of `balanced` on P100 would likely exceed 12h.

### Kaggle logs do not show real-time progress
- The original `filament_top50_single.py` uses `from tqdm.auto import tqdm` and plain `print`.
- `tqdm.auto` in a Kaggle notebook writes to a widget, not always flushed to the Logs tab.
- v4 addresses this by switching to `from tqdm import tqdm`, `PYTHONUNBUFFERED=1`, and `flush=True` on prints.

### GPU session limit
- Kaggle allows at most 2 running GPU sessions for this account.
- You must cancel a running session before pushing v4.

### Local network is flaky
- `ConnectionResetError(10054)` happens frequently.
- Polling scripts with retry are needed; manual CLI commands often fail.

### Devin subagent quota exhausted
- The current Devin session can no longer spawn subagents.
- All work must be done directly.

---

## 7. Commands Reference

### Build and push a Kaggle notebook

```powershell
# Build v4 fast
python C:\Users\srik2\AppData\Local\Temp\build_kaggle_v4_fast.py

# Push to Kaggle (requires at least one free GPU slot)
python -m kaggle kernels push -p "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\notebooks\kaggle"
```

### Check status

```powershell
python -m kaggle kernels status srikrishnaos/filament-5-fold-v2
python -m kaggle kernels status srikrishnaos/filament-5fold-v4-fast
```

### Download output

```powershell
python -m kaggle kernels output srikrishnaos/filament-5fold-v4-fast -p C:\Users\srik2\AppData\Local\Temp\kaggle_v4_output
```

### Submit to competition

```powershell
python -m kaggle competitions submit -c filament-segmentation-2026 -f "C:\Users\srik2\AppData\Local\Temp\kaggle_v4_output\submission.csv" -m "v4 fast ensemble"
```

### Poll v4 after it starts

```powershell
# Use the v2 polling script as a template and update the KERNEL variable
Get-Content C:\Users\srik2\AppData\Local\Temp\kaggle_v2_status.log -Tail 30
```

---

## 8. Kaggle URLs

- v2 (stuck): https://www.kaggle.com/code/srikrishnaos/filament-5-fold-v2
- v3 (pushed, not run): https://www.kaggle.com/code/srikrishnaos/filament-5-fold-v3
- v4 fast (ready): https://www.kaggle.com/code/srikrishnaos/filament-5fold-v4-fast

---

## 9. Recommended Next Steps

1. Cancel all running Kaggle kernels at https://www.kaggle.com/srikrishnaos/code to free GPU slots.
2. Push v4 fast:
   ```powershell
   python -m kaggle kernels push -p "C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\notebooks\kaggle"
   ```
3. Start a local polling script for v4:
   ```powershell
   python C:\Users\srik2\AppData\Local\Temp\poll_kaggle_v2.py
   ```
   (update `KERNEL` to `srikrishnaos/filament-5fold-v4-fast` inside the script first)
4. Wait for v4 to complete. Expected time: **3–5 hours** on P100.
5. Download `submission.csv` and submit.

If v4 also hangs, the next mitigation would be:
- Run a **single-fold fast** notebook to get a quick submission, or
- Reduce `num_workers=0` further, disable `use_amp`, or use the `edgeattnet` preset, or
- Investigate the exact hang by adding print statements around `build_model()` and `build_cache()`.

---

## 10. Architecture Notes

- The model uses `segmentation_models_pytorch` with `timm` encoders.
- P100 is `sm_60`, so `cu118` is required. `torch==2.5.1+cu118` is explicitly installed in all Kaggle notebooks.
- The Kaggle image comes with `torch==2.10.0+cu128` preinstalled, which is overwritten.
- `channels_last` is used for better P100 performance.
- D4 TTA (8 flips) is used during inference unless overridden.
- The final submission is generated from an **average of probability maps**, not from averaging RLE masks.

---

*Document generated by Devin on 2026-08-27. Update it as the project progresses.*
