"""
🚀 V8 PRODUCTION ENGINE: STAGE 1 — TRAIN YOLO SEED DETECTOR (AUDITED & CERTIFIED)
==================================================================================
Prepares MAGFiLO dataset into YOLO format and trains high-capacity seed detector.
Audited with all Devin CLI & OpenCode fixes applied (sys import, enhanced unsharp, robust grouping).
"""

import os, sys, json, cv2, shutil, glob, time, gc, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import torch

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(2026)
np.random.seed(2026)

print("=" * 80, flush=True)
print("🚀 V8 STAGE 1: TRAINING YOLO SEED DETECTOR (MAGFiLO)", flush=True)
print("=" * 80, flush=True)

candidate_bases = [
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026"),
    Path("data/MAGFiLO_1.0_Kaggle_2026"),
    Path("../data/MAGFiLO_1.0_Kaggle_2026"),
]

data_base = next((c for c in candidate_bases if c.exists()), None)
if data_base is None:
    raise FileNotFoundError("❌ MAGFiLO dataset not found!")

train_img_dir = data_base / "train" / "train_images"
json_matches = list(data_base.glob("**/*train*.json"))
if not json_matches: raise FileNotFoundError("No train JSON found in dataset")
ann_json_path = json_matches[0]

# Parse COCO Annotations
with open(ann_json_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
id_to_shape = {img['id']: (img.get('height', 2048), img.get('width', 2048)) for img in coco_data['images']}

file_to_polys = {}
for ann in coco_data['annotations']:
    img_id = ann.get('image_id')
    fn = id_to_file.get(img_id)
    if not fn: continue
    h, w = id_to_shape[img_id]
    for poly in ann.get('segmentation', []):
        if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
            coords = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if not np.any(np.isnan(coords)) and not np.any(np.isinf(coords)):
                coords[:, 0] = np.clip(coords[:, 0] / w, 0.0, 1.0)
                coords[:, 1] = np.clip(coords[:, 1] / h, 0.0, 1.0)
                file_to_polys.setdefault(fn, []).append(coords.reshape(-1).tolist())

yolo_root = Path("/kaggle/working/yolo_filament_dataset") if os.path.exists("/kaggle/working") else Path("./yolo_filament_dataset")
if yolo_root.exists(): shutil.rmtree(yolo_root)

for split in ["train", "val"]:
    (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

all_files = sorted(list(file_to_polys.keys()))
df = pd.DataFrame({"filename": all_files})
# Robust year group extraction
df["group"] = df["filename"].apply(lambda x: x[:4] if x[:4].isdigit() else "0000")
train_idx, val_idx = next(GroupKFold(n_splits=5).split(df, groups=df["group"]))
train_files = set(df.iloc[train_idx]["filename"])

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

for idx, (fn, polys) in enumerate(file_to_polys.items()):
    try:
        src_path = train_img_dir / fn
        if not src_path.exists(): continue
        split_tag = "train" if fn in train_files else "val"
        dst_img_path = yolo_root / "images" / split_tag / fn
        dst_lbl_path = yolo_root / "labels" / split_tag / f"{Path(fn).stem}.txt"
        
        raw = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if raw is None: continue
        
        enhanced = clahe.apply(raw)
        # Apply unsharp mask to enhanced image for maximum contrast
        unsharp = cv2.addWeighted(enhanced, 1.5, cv2.GaussianBlur(enhanced, (0, 0), 3.0), -0.5, 0)
        img_3ch = np.stack([raw, enhanced, unsharp], axis=-1)
        img_1024 = cv2.resize(img_3ch, (1024, 1024), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_img_path), img_1024)
        
        with open(dst_lbl_path, 'w') as lf:
            for p in polys:
                lf.write("0 " + " ".join(f"{coord:.6f}" for coord in p) + "\n")
                
        del raw, enhanced, unsharp, img_3ch, img_1024
        if idx % 50 == 0: gc.collect()
    except Exception as e:
        print(f"⚠️ Error {fn}: {e}", flush=True)

with open(yolo_root / "data.yaml", 'w') as f:
    f.write(f"path: {yolo_root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")

print("✅ YOLO Dataset prepared successfully!", flush=True)

from ultralytics import YOLO
model = YOLO('yolov8x-seg.pt')

results = model.train(
    data=str(yolo_root / "data.yaml"),
    epochs=30,
    imgsz=1024,
    batch=2,
    device=0 if torch.cuda.is_available() else 'cpu',
    workers=2,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    degrees=45.0,
    fliplr=0.5,
    flipud=0.5,
    scale=0.2,
    hsv_v=0.15,
    save=True,
    project="/kaggle/working/yolo_output" if os.path.exists("/kaggle/working") else "./yolo_output",
    name="yolo_seed_detector",
    exist_ok=True
)

print("\n🎉 STAGE 1 TRAINING COMPLETE! Saved to:", model.trainer.best, flush=True)
