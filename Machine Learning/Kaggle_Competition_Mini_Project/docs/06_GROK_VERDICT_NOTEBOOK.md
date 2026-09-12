# GROK VERDICT — V8.1 Kaggle notebook
# Status: DO NOT RUN ALL YET
# Paste this into Antigravity. Human: do not upload until the two fatal bugs are patched.

The architecture and Dual-T4 layout are right. Two bugs will produce a garbage or crashed run.

---

## FATAL 1 — notebook cell 5 cannot see `data_dir`

```python
!python scripts/convert_coco_to_yolo.py --data-dir "$data_dir" --out-dir "/kaggle/working/data/yolo_seg"
```

`data_dir` is a Python variable from cell 3. `$data_dir` is a **bash** variable and is empty on Kaggle. Converter dies or points at nothing. YOLO then trains on missing data.

Replace that cell with Python, not shell interpolation:

```python
from pathlib import Path
import subprocess, sys
out = Path("/kaggle/working/data/yolo_seg")
cmd = [sys.executable, "scripts/convert_coco_to_yolo.py",
       "--data-dir", str(data_dir),
       "--out-dir", str(out)]
print("CONVERT:", cmd, flush=True)
subprocess.check_call(cmd)
```

Confirm `convert_coco_to_yolo.py` actually accepts `--data-dir` and `--out-dir`. If the flags are `--data_dir`, match them. Do not guess.

Also print `data.yaml` after convert and assert train/val image counts > 0.

## FATAL 2 — box coordinates scaled twice

In `v8_1/3_infer_cascade.py` you call:

```python
yolo_model.predict(source=str(p), imgsz=1024, ...)  # p is a 2048 JPEG
boxes = yolo_preds[0].boxes.xyxy   # already in original 2048 px
scale = w / 1024
scaled_bbox = box * scale          # now ~4096. WRONG
```

Ultralytics `boxes.xyxy` is in the **source image** frame. You passed the 2048 file, so do **not** multiply by 2048/1024.

Old V8 notebook scaled because it predicted on a **resized 1024 array**. Different input, different rule.

Fix:

```python
boxes = yolo_preds[0].boxes.xyxy.cpu().numpy()  # already 2048 space
# clip to image
x1, y1, x2, y2 = boxes[i]
x1, y1 = max(0, int(x1)), max(0, int(y1))
x2, y2 = min(w, int(x2)), min(h, int(y2))
```

If `preds[0].masks` exists, prefer the YOLO mask resized with `orig_shape`, then refine. Do not scale boxes as if the source were 1024.

Add a one-line debug on the first test image: print raw xyxy min/max. Must be ≤ 2048.

## Must-fix smaller holes

3. `if raw is None: continue` drops the stem. Unique stems < 180 and cell 9 assert dies. Emit `{stem}_1` + `encode_mask(zeros)` instead.

4. `--residual 1` is accepted and ignored. Either implement the IoU<0.30 residual path or delete the flag. Default stays 0.

5. `YOLOConfig.ACCUM` is printed and never passed to `model.train()`. Either pass ultralytics `nbs=batch*accum` or stop claiming effective batch 8.

6. Cell 9 `segmentation_rle == "PPPP4"` is a check, not an encoder. Prefer `decode_rle(...).sum()==0` for empty count.

7. First Kaggle train: **single GPU**. `device=[0,1]` DDP inside a notebook `!python` subprocess hangs often on Kaggle. Use `device=0`, `--batch 2` for pass 1. If VRAM after epoch 1 is < 10 GB reserved, bump to batch 4. Dual T4 still used at infer (YOLO 0, refiner 1).

8. Crop refiner cell should be:
   `!python v8_1/2_train_crop_refiner.py --boxes-from-gt`
   Your trainer already uses GT polygons. The YOLO-weight gate is pointless for pass 1 and will block if YOLO writes `best.pt` under a slightly different path.

9. After YOLO train, print the exact `best.pt` path that exists. Infer must load that same path. Do not assume `runs/v8_1_yolo_s1024/weights/best.pt` if ultralytics incremented `v8_1_yolo_s10242`.

## Answers

1. **Not ready for Run All.** Upload only after FATAL 1+2 are patched and you paste the new infer snippet + convert cell here.
2. **Start `--batch 2` + `device=0`.** Not batch 4 + DDP.

## What the human should click (after the patch)
- New Kaggle notebook, upload `V8_1_Kaggle_Production.ipynb`
- Accelerator: GPU T4 x2
- Internet: ON
- Add competition data `filament-segmentation-2026`
- Do **not** also run `V7_14Models_Mega_Master.ipynb` in the same session
- Save version after it finishes, submit `/kaggle/working/submission.csv`

## What to paste back
- The patched convert cell
- The patched box-handling block from `3_infer_cascade.py`
- Confirmation that `--data-dir` is a real argparse flag
- Then I say upload.
