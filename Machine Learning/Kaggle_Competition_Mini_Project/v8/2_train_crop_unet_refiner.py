"""
🧠 V8 PRODUCTION ENGINE: STAGE 2 — TRAIN CROP U-NET REFINER (AUDITED & CERTIFIED)
==================================================================================
Extracts high-resolution bounding box crops from ground truth filaments and trains
a specialized ResNet-34 Crop U-Net refiner with Lovász-Hinge + SoftDice loss.
Audited with all Devin CLI fixes applied (sys import, enhanced unsharp, Lovász clamp, GradScaler fix).
"""

import os, sys, glob, json, time, math, cv2, gc
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(2026)
np.random.seed(2026)

print("=" * 80, flush=True)
print("🧠 V8 STAGE 2: TRAINING SPECIALIZED CROP U-NET REFINER", flush=True)
print("=" * 80, flush=True)

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# 2. Locate Dataset
candidate_bases = [
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026"),
    Path("data/MAGFiLO_1.0_Kaggle_2026"),
    Path("../data/MAGFiLO_1.0_Kaggle_2026"),
]
data_base = next((c for c in candidate_bases if c.exists()), None)
train_img_dir = data_base / "train" / "train_images"
json_matches = list(data_base.glob("**/*train*.json"))
if not json_matches: raise FileNotFoundError("No train JSON found!")
ann_json_path = json_matches[0]

# 3. Parse COCO Instance Polygons & Bounding Boxes
with open(ann_json_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
id_to_shape = {img['id']: (img.get('height', 2048), img.get('width', 2048)) for img in coco_data['images']}

clahe_op = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# GroupKFold Split
all_images = sorted(list(set(id_to_file.values())))
df_imgs = pd.DataFrame({"filename": all_images})
df_imgs["group"] = df_imgs["filename"].apply(lambda x: x[:4] if x[:4].isdigit() else "0000")
train_img_idx, val_img_idx = next(GroupKFold(n_splits=5).split(df_imgs, groups=df_imgs["group"]))
train_img_set = set(df_imgs.iloc[train_img_idx]["filename"])

# 4. Extract Crop Pairs from Ground Truth Instances
CROP_SIZE = 256
PADDING = 20

train_crop_records = []
val_crop_records = []

print("📦 Extracting instance crops from 707 training images...", flush=True)

anns_by_file = {}
for ann in coco_data['annotations']:
    img_id = ann.get('image_id')
    fn = id_to_file.get(img_id)
    if fn and ann.get('segmentation'):
        anns_by_file.setdefault(fn, []).append(ann)

for fn, anns in tqdm(anns_by_file.items(), desc="Extracting Crops"):
    img_path = train_img_dir / fn
    if not img_path.exists(): continue
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None: continue
    
    h_orig, w_orig = raw.shape[:2]
    c_img = clahe_op.apply(raw)
    unsharp = cv2.addWeighted(c_img, 1.5, cv2.GaussianBlur(c_img, (0, 0), 3.0), -0.5, 0)
    img_3ch = np.stack([raw, c_img, unsharp], axis=-1)
    
    is_train = fn in train_img_set
    
    for ann in anns:
        segs = ann.get('segmentation', [])
        if not segs: continue
        
        inst_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for poly in segs:
            if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0], 0, w_orig - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h_orig - 1)
                cv2.fillPoly(inst_mask, [pts.astype(np.int32)], 1)
                
        area = int(inst_mask.sum())
        if area < 100: continue
        
        y_idx, x_idx = np.where(inst_mask > 0)
        x1, y1 = np.min(x_idx), np.min(y_idx)
        x2, y2 = np.max(x_idx), np.max(y_idx)
        
        x1_p = max(0, x1 - PADDING)
        y1_p = max(0, y1 - PADDING)
        x2_p = min(w_orig, x2 + PADDING)
        y2_p = min(h_orig, y2 + PADDING)
        
        if (x2_p - x1_p) < 8 or (y2_p - y1_p) < 8: continue
        
        img_crop = cv2.resize(img_3ch[y1_p:y2_p, x1_p:x2_p], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        mask_crop = (cv2.resize(inst_mask[y1_p:y2_p, x1_p:x2_p], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR) > 0.50).astype(np.uint8)
        
        record = (img_crop, mask_crop)
        if is_train:
            train_crop_records.append(record)
        else:
            val_crop_records.append(record)

print(f"📊 Extracted {len(train_crop_records)} Train Crops | {len(val_crop_records)} Val Crops!", flush=True)

# 5. Crop Dataset & Transforms
mean_3ch = (0.485, 0.456, 0.406)
std_3ch = (0.229, 0.224, 0.225)

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.10, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=mean_3ch, std=std_3ch),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Normalize(mean=mean_3ch, std=std_3ch),
    ToTensorV2()
])

class FilamentCropDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        img, mask = self.records[idx]
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()
        return img, mask

train_loader = DataLoader(FilamentCropDataset(train_crop_records, train_transforms), batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(FilamentCropDataset(val_crop_records, val_transforms), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# 6. Model & Loss Function
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
model.to(device)

def soft_dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3)) + smooth
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
    return 1.0 - (num / den).mean()

def lovasz_hinge(logits, targets):
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)
    if len(targets_flat) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    signs = 2.0 * targets_flat.float() - 1.0
    errors = 1.0 - logits_flat * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = targets_flat[perm]
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / torch.clamp(union, min=1e-7)
    if p > 1: jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return torch.dot(F.relu(errors_sorted), jaccard)

def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(logits, targets)
    lov = lovasz_hinge(logits, targets)
    return 0.3 * bce + 0.4 * dice + 0.3 * lov

# 7. Optimizer & Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

best_val_dice = 0.0
save_path = "/kaggle/working/best_crop_unet_refiner.pth" if os.path.exists("/kaggle/working") else "models/best_crop_unet_refiner.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print("\n🏋️ Training Crop U-Net Refiner for 15 Epochs...", flush=True)

for epoch in range(1, 16):
    model.train()
    total_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(imgs)
            loss = combined_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    scheduler.step()
    
    # Validation
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                probs = torch.sigmoid(model(imgs))
                preds = (probs > 0.50).float()
                num = 2.0 * (preds * masks).sum(dim=(2, 3)) + 1e-6
                den = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-6
                dices.extend((num / den).cpu().numpy().tolist())
                
    mean_dice = float(np.mean(dices))
    print(f"Epoch {epoch:02d}/15 | Train Loss: {total_loss/len(train_loader):.4f} | Val Dice: {mean_dice:.4f}", flush=True)
    
    if mean_dice > best_val_dice:
        best_val_dice = mean_dice
        torch.save({"model_state_dict": model.state_dict(), "arch": "Unet", "encoder": "resnet34", "val_dice": best_val_dice}, save_path)
        print(f"   🏆 Saved Best Checkpoint -> {save_path} (Val Dice: {best_val_dice:.4f})", flush=True)

print(f"\n🎉 STAGE 2 TRAINING COMPLETE! Best Val Dice: {best_val_dice:.4f}", flush=True)
