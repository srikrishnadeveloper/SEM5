# GROK VERDICT — patched V8.1 notebook
# Status: GO, after you confirm cell 6 batch
# Human runs this on Kaggle. Antigravity is done until logs come back.

Fatal 1 and Fatal 2 look fixed:
- convert uses Python `subprocess` + `str(data_dir)` + `--data-dir` / `--out-dir`
- infer uses raw `xyxy` clipped to 2048, no `* (w/1024)`
- unreadable image still emits `{stem}_1` + encoded zeros
- first-image xyxy debug with assert ≤ 2048+50
- train default: batch 2, device 0
- refiner: `--boxes-from-gt`
- empty count via decode sum == 0

## One check before upload
Open `V8_1_Kaggle_Production.ipynb` cell 6 (YOLO train).

Must be:

```python
!python v8_1/1_train_yolo.py --batch 2 --epochs 20
```

or no `--batch` flag at all (script default is 2).

If that cell still says `--batch 4`, change it. The notebook flag overrides `config.py`.

## Human click list
1. kaggle.com → New Notebook
2. Upload `C:\Users\srik2\Desktop\Filament_Colab_Run\V8_1_Kaggle_Production.ipynb`
3. Accelerator: **GPU T4 x2**
4. Internet: **ON**
5. Add data: competition `filament-segmentation-2026` only
6. Do not attach V7 / 14-model datasets
7. Do not run `V7_14Models_Mega_Master.ipynb` in this kernel
8. Save & Run All
9. Watch cell 5: `Verified YOLO dataset images` must be train>0 and val>0
10. Watch cell 6 first epoch: no CUDA OOM. If OOM, stop and rerun that cell with `--batch 1`
11. Watch cell 8 first line: `[DEBUG Image 0] YOLO raw xyxy range: min=..., max=...` — max must be ~hundreds to 2048, not ~4000
12. Cell 9: unique stems == 180
13. Submit `/kaggle/working/submission.csv`

Expected wall time: 2–4 hours. Do not stop after YOLO unless it errors.

## What to paste back here (not earlier)
From the Kaggle log, these blocks only:

- convert counts (train/val images)
- YOLO last 3 epochs + the pq_mean / pq_max table
- `[DEBUG Image 0] YOLO raw xyxy range: ...`
- submission audit: n_rows, unique stems, empty count, rows/image mean
- public LB score if you submitted

Do not paste the whole notebook.

## How to read the first score
| Public PQ | Meaning |
|---|---|
| < 0.15 | boxes still wrong or weights missing |
| 0.15–0.30 | pipeline lives, recall/threshold still bad |
| 0.30–0.40 | first real instance run; sweep conf next |
| > 0.40 | stay on this stack; do not go back to 14 U-Nets |

If rows/image mean < 2 or empty disks > 40%, do not train bigger YOLO. Fix recall first (`--conf 0.15` infer only).
