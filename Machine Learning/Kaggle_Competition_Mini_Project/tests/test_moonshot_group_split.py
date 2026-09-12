"""
tests/test_moonshot_group_split.py — Test Grouped Split and Zero Physical File Leakage.

Authority: ChatGPT Master
Directive: 17
"""

import pytest
import pandas as pd
from pathlib import Path
from moonshot_2048.data import build_grouped_split, assert_zero_leakage


def test_grouped_split_zero_leakage_synthetic():
    """Verify that multiple annotator observations for duplicate physical images stay in the same fold."""
    synthetic_images = [
        # Physical image 1 (year 2013): 2 annotator observations
        {"id": 101, "file_name": "20130101_0000_disk.jpeg", "width": 2048, "height": 2048},
        {"id": 102, "file_name": "20130101_0000_disk.jpeg", "width": 2048, "height": 2048},
        # Physical image 2 (year 2013): 1 annotator observation
        {"id": 103, "file_name": "20130601_1200_disk.jpeg", "width": 2048, "height": 2048},
        # Physical image 3 (year 2014): 2 annotator observations
        {"id": 104, "file_name": "20140201_0800_disk.jpeg", "width": 2048, "height": 2048},
        {"id": 105, "file_name": "20140201_0800_disk.jpeg", "width": 2048, "height": 2048},
        # Physical image 4 (year 2015): 1 annotator observation
        {"id": 106, "file_name": "20150301_1000_disk.jpeg", "width": 2048, "height": 2048},
        # Physical image 5 (year 2016): 1 annotator observation
        {"id": 107, "file_name": "20160401_1400_disk.jpeg", "width": 2048, "height": 2048},
    ]
    disk_filenames = {img["file_name"] for img in synthetic_images}

    df = build_grouped_split(synthetic_images, disk_filenames, n_splits=3)
    assert len(df) == len(synthetic_images)

    # Check that each physical file is only in one fold
    for fn, group in df.groupby("file_name"):
        folds = group["fold"].unique()
        assert len(folds) == 1, f"Physical file {fn} split across folds: {folds}"

    # Verify assert_zero_leakage succeeds
    for fold in range(3):
        assert_zero_leakage(df, val_fold=fold)


def test_assert_zero_leakage_raises_on_leak():
    """Verify that assert_zero_leakage detects and catches physical file leakage."""
    leaked_df = pd.DataFrame([
        {"file_name": "leaked_img.jpeg", "fold": 0},
        {"file_name": "leaked_img.jpeg", "fold": 1},
        {"file_name": "safe_img.jpeg", "fold": 0},
    ])
    with pytest.raises(AssertionError, match="FATAL LEAKAGE DETECTED"):
        assert_zero_leakage(leaked_df, val_fold=1)


def test_magfilo_stage0_exact_counts():
    """Verify exact Stage-0 counts on official dataset: 707 physical images, 1154 observations, 579 train, 128 val."""
    from moonshot_2048.config import MAGFILO_DIR
    from moonshot_2048.data import parse_magfilo_coco
    
    train_dir = MAGFILO_DIR / "train"
    json_candidates = list(train_dir.glob("*.json"))
    if not json_candidates:
        pytest.skip("Official MAGFiLO dataset not found locally; skipping exact counts test.")

    images, annotations, categories = parse_magfilo_coco(json_candidates[0])
    assert len(images) == 1154, f"Expected 1,154 observations, got {len(images)}"
    json_filenames = {img["file_name"] for img in images}
    assert len(json_filenames) == 707, f"Expected 707 physical files, got {len(json_filenames)}"

    img_dir = train_dir / "train_images"
    jpegs_on_disk = list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg"))
    disk_filenames = {p.name for p in jpegs_on_disk}
    assert len(disk_filenames) >= 707

    df_samples = build_grouped_split(images, disk_filenames, n_splits=5)
    assert_zero_leakage(df_samples, val_fold=0)

    n_trn = int(df_samples[df_samples["fold"] != 0]["file_name"].nunique())
    n_val = int(df_samples[df_samples["fold"] == 0]["file_name"].nunique())
    assert n_trn == 579, f"Expected 579 train files, got {n_trn}"
    assert n_val == 128, f"Expected 128 val files, got {n_val}"


if __name__ == "__main__":
    test_grouped_split_zero_leakage_synthetic()
    test_assert_zero_leakage_raises_on_leak()
    test_magfilo_stage0_exact_counts()
    print("test_moonshot_group_split passed successfully.")
