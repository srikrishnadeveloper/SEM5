"""
Unit test for dataset split isolation.
Ensures zero physical file_name leakage between train and val splits.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_yolo_seg_split_isolation():
    """Verify that images in data/yolo_seg/images/train and val have zero overlap in physical stem."""
    yolo_dir = PROJECT_ROOT / "data" / "yolo_seg"
    if not yolo_dir.exists():
        print("[SKIP] data/yolo_seg does not exist yet.")
        return

    train_imgs = list((yolo_dir / "images" / "train").glob("*.jpeg"))
    val_imgs = list((yolo_dir / "images" / "val").glob("*.jpeg"))

    # Stems are formatted as {physical_stem}_ann_{img_id}
    train_physical = {p.stem.split("_ann_")[0] for p in train_imgs}
    val_physical = {p.stem.split("_ann_")[0] for p in val_imgs}

    intersection = train_physical.intersection(val_physical)
    assert len(intersection) == 0, f"LEAKAGE DETECTED! Physical files in both train and val: {intersection}"
    print(f"[PASS] YOLO dataset split isolation: {len(train_physical)} train files, {len(val_physical)} val files, 0 overlap.")


def test_all_5folds_groupkfold_isolation():
    """Verify that across all 5 folds of GroupKFold by year, no physical file_name is ever in both splits."""
    json_path = PROJECT_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not json_path.exists():
        print("[SKIP] COCO JSON not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    df = pd.DataFrame(images)
    df["group"] = df["file_name"].apply(lambda x: x[:4] if x[:4].isdigit() else "0000")

    gkf = GroupKFold(n_splits=5)
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(df, groups=df["group"])):
        trn_files = set(df.iloc[trn_idx]["file_name"])
        val_files = set(df.iloc[val_idx]["file_name"])
        overlap = trn_files.intersection(val_files)
        assert len(overlap) == 0, f"Fold {fold} has overlap: {overlap}"

    print(f"[PASS] All 5 folds tested: 0 physical file_name leakage across every fold.")


if __name__ == "__main__":
    test_yolo_seg_split_isolation()
    test_all_5folds_groupkfold_isolation()
    print("\n=== DATASET SPLIT ISOLATION TESTS PASSED ===")
