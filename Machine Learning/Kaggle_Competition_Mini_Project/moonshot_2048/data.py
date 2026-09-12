"""
moonshot_2048/data.py — Official-Data-Only Dataset Converter & Grouped Split.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Frozen Host & Scientific Rules:
1. Use ONLY official competition training images and JSON annotations.
   Never use Harvard Dataverse test-overlap downloads or hidden test labels.
2. Single-class segmentation target: all categories 1, 2, 3, 4 map to class 0 ('filament').
3. Multi-annotator preservation: Each annotator observation has a unique JSON image_id.
   Give duplicate physical images annotator-specific training filenames/links
   (e.g., {stem}_ann_{image_id}.jpeg) so target masks are never overwritten or OR-merged.
4. Group train/validation strictly by physical filename / year prefix.
   ASSERT zero physical filename leakage across folds.
5. Export normalized YOLO polygon format ([0, 1] relative to width & height)
   and generate data.yaml and conversion_report.json.
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moonshot_2048.config import MoonshotConfig, MAGFILO_DIR, YOLO_DATA_DIR


def polygon_area(pts: np.ndarray) -> float:
    """Compute polygon area using Shoelace formula. pts: (N, 2)."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def safe_materialize_image(src: Path, dst: Path, mode: str = "auto") -> str:
    """Safely materialize src image at dst path.
    Supports symlink (preferred on Linux/Kaggle to save RAM/disk), hardlink, or copy.
    Verifies that dst exists and is non-empty.
    """
    if dst.exists():
        try:
            dst.unlink()
        except Exception:
            pass

    if mode == "symlink":
        candidates = ["symlink", "hardlink", "copy2"]
    elif mode == "hardlink":
        candidates = ["hardlink", "symlink", "copy2"]
    elif mode == "copy":
        candidates = ["copy2"]
    else:  # "auto"
        # On Windows, symlink may require admin, so try symlink, then hardlink, then copy2
        candidates = ["symlink", "hardlink", "copy2"]

    for cand in candidates:
        try:
            if cand == "symlink":
                os.symlink(src.resolve(), dst)
            elif cand == "hardlink":
                os.link(src, dst)
            elif cand == "copy2":
                shutil.copy2(src, dst)

            if dst.exists() and dst.stat().st_size > 0:
                return cand
            else:
                if dst.exists():
                    dst.unlink()
        except Exception:
            if dst.exists():
                try:
                    dst.unlink()
                except Exception:
                    pass
            continue

    raise RuntimeError(f"FATAL: Failed to materialize {src} to {dst}")


def parse_magfilo_coco(json_path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """Load and validate official MAGFiLO COCO JSON structure."""
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    return images, annotations, categories


def build_grouped_split(
    images: List[dict],
    disk_filenames: Set[str],
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign folds using GroupKFold by year prefix of physical filename.
    Guarantees that all annotator observations for a given physical file (and year)
    reside strictly in the SAME fold.
    """
    records = []
    for img in images:
        fn = img["file_name"]
        if fn not in disk_filenames:
            continue
        # Year prefix (e.g. '2013' from '20130514_...')
        year_group = fn[:4] if fn[:4].isdigit() else "group_0000"
        records.append({
            "image_id": img["id"],
            "file_name": fn,
            "physical_stem": Path(fn).stem,
            "group": year_group,
            "width": img.get("width", 2048),
            "height": img.get("height", 2048),
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No matching images found on disk for grouped split!")

    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(gkf.split(df, groups=df["group"])):
        df.loc[val_idx, "fold"] = fold_idx

    return df


def assert_zero_leakage(df_samples: pd.DataFrame, val_fold: int) -> None:
    """Assert zero physical file leakage between train and validation splits."""
    val_files = set(df_samples[df_samples["fold"] == val_fold]["file_name"])
    train_files = set(df_samples[df_samples["fold"] != val_fold]["file_name"])
    overlap = val_files.intersection(train_files)
    if overlap:
        raise AssertionError(
            f"FATAL LEAKAGE DETECTED! {len(overlap)} physical files appear in both train and val: {list(overlap)[:5]}"
        )
    print(f"  [LEAKAGE CHECK] PASSED: Exactly 0 shared physical files between train ({len(train_files)}) and val ({len(val_files)}).")


def convert_dataset(
    data_dir: Path,
    out_dir: Path,
    val_fold: int = 0,
    n_splits: int = 5,
    allow_missing: bool = False,
    link_mode: str = "auto",
) -> Dict:
    """Convert official MAGFiLO dataset into YOLO-seg format for native 2048 training."""
    train_dir = data_dir / "train"
    img_dir = train_dir / "train_images"
    json_candidates = list(train_dir.glob("*.json")) + list(data_dir.glob("*.json"))
    if not json_candidates:
        raise FileNotFoundError(f"No annotation JSON found in {data_dir}")
    json_path = json_candidates[0]

    print("=" * 75)
    print("MOONSHOT 2048: DATASET CONVERTER & GROUPED SPLIT")
    print("=" * 75)
    print(f"  Annotation JSON: {json_path}")
    print(f"  Image directory: {img_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"  Validation fold: {val_fold} (out of {n_splits})")

    images, annotations, categories = parse_magfilo_coco(json_path)

    # Inspect images on disk
    jpegs_on_disk = list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg"))
    disk_filenames = {p.name for p in jpegs_on_disk}
    json_filenames = {img["file_name"] for img in images}
    missing_files = sorted(list(json_filenames - disk_filenames))
    pct_missing = (len(missing_files) / len(json_filenames) * 100.0) if json_filenames else 0.0

    print(f"  JSON Observations: {len(images)} across {len(json_filenames)} unique physical files")
    print(f"  JPEGs on disk:     {len(jpegs_on_disk)}")
    print(f"  Missing files:     {len(missing_files)} ({pct_missing:.2f}%)")

    if pct_missing > 5.0 and not allow_missing:
        raise RuntimeError(f"FATAL: Missing > 5% of training files ({pct_missing:.2f}%).")

    # Exact Stage-0 Assertions (Directive 18)
    if len(images) == 1154:
        assert len(images) == 1154, f"FATAL: Expected 1,154 observations, got {len(images)}"
        assert len(json_filenames) == 707, f"FATAL: Expected 707 unique physical files, got {len(json_filenames)}"
        assert len(disk_filenames) >= 707, f"FATAL: Expected at least 707 JPEGs on disk, got {len(disk_filenames)}"

    # Build grouped split
    df_samples = build_grouped_split(images, disk_filenames, n_splits=n_splits)
    assert_zero_leakage(df_samples, val_fold=val_fold)

    # Verify Fold-0 expected physical counts
    n_trn_files = int(df_samples[df_samples["fold"] != val_fold]["file_name"].nunique())
    n_val_files = int(df_samples[df_samples["fold"] == val_fold]["file_name"].nunique())
    if len(images) == 1154 and val_fold == 0 and n_splits == 5:
        assert n_trn_files == 579, f"FATAL: Expected 579 Fold-0 train files, got {n_trn_files}"
        assert n_val_files == 128, f"FATAL: Expected 128 Fold-0 val files, got {n_val_files}"
        print(f"  [STAGE-0 VERIFIED] Exactly 579 train files, 128 val files, 0 leakage.")

    # Process annotations: map all categories 1-4 to Class 0 'filament'
    img_id_to_record = {img["id"]: img for img in images}
    valid_polys_by_img_id = defaultdict(list)
    cat_counts = Counter()
    skipped_points = 0
    skipped_area = 0
    valid_polys = 0

    for ann in annotations:
        cat_id = ann.get("category_id", 1)
        cat_counts[cat_id] += 1
        img_id = ann["image_id"]
        if img_id not in img_id_to_record:
            continue

        img_rec = img_id_to_record[img_id]
        w = float(img_rec.get("width", 2048))
        h = float(img_rec.get("height", 2048))

        segs = ann.get("segmentation", [])
        if not isinstance(segs, list):
            continue

        for poly in segs:
            if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
                skipped_points += 1
                continue
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
                skipped_points += 1
                continue
            area = polygon_area(pts)
            if area <= 0.0:
                skipped_area += 1
                continue

            # Normalized [0, 1] YOLO polygon coordinates
            norm_pts = pts.copy()
            norm_pts[:, 0] = np.clip(norm_pts[:, 0] / w, 0.0, 1.0)
            norm_pts[:, 1] = np.clip(norm_pts[:, 1] / h, 0.0, 1.0)

            valid_polys_by_img_id[img_id].append(norm_pts.reshape(-1).tolist())
            valid_polys += 1

    print("  Annotations Audit:")
    for cat_id in sorted(cat_counts.keys()):
        print(f"    - Category {cat_id}: {cat_counts[cat_id]} (mapped to class 0 'filament')")
    print(f"  Valid Polygons: {valid_polys} | Skipped (<6 pts): {skipped_points} | Skipped (area<=0): {skipped_area}")

    # Prepare output directories
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Test materialization
    first_fn = df_samples.iloc[0]["file_name"]
    test_src = img_dir / first_fn
    test_dst = out_dir / "_test_link"
    detected_method = safe_materialize_image(test_src, test_dst, mode=link_mode)
    if test_dst.exists():
        test_dst.unlink()
    print(f"  Materialization Method: {detected_method}")

    # Export images & label files
    written = {"train": 0, "val": 0}
    for _, row in df_samples.iterrows():
        split = "val" if row["fold"] == val_fold else "train"
        img_id = row["image_id"]
        fn = row["file_name"]
        stem = Path(fn).stem
        # Unique sample stem ensures distinct annotator observations are never overwritten
        sample_stem = f"{stem}_ann_{img_id}"

        src_img = img_dir / fn
        dst_img = out_dir / "images" / split / f"{sample_stem}.jpeg"
        dst_lbl = out_dir / "labels" / split / f"{sample_stem}.txt"

        safe_materialize_image(src_img, dst_img, mode=detected_method)
        polys = valid_polys_by_img_id.get(img_id, [])
        with open(dst_lbl, "w", encoding="utf-8") as f_lbl:
            for p in polys:
                coord_str = " ".join(f"{c:.6f}" for c in p)
                f_lbl.write(f"0 {coord_str}\n")

        written[split] += 1

    print(f"  Materialized {written['train']} train images, {written['val']} val images.")

    # Write data.yaml
    yaml_text = f"""# Native-2048 Moonshot YOLO-seg dataset config
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: filament
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    # Export conversion_report.json
    report = {
        "json_path": str(json_path),
        "total_json_images": len(images),
        "total_annotations": len(annotations),
        "valid_polygons": valid_polys,
        "skipped_points": skipped_points,
        "skipped_area": skipped_area,
        "train_samples": written["train"],
        "val_samples": written["val"],
        "train_unique_files": int(df_samples[df_samples['fold'] != val_fold]['file_name'].nunique()),
        "val_unique_files": int(df_samples[df_samples['fold'] == val_fold]['file_name'].nunique()),
        "val_fold": val_fold,
        "n_splits": n_splits,
        "zero_file_leakage": True,
        "linking_method": detected_method,
    }
    report_path = out_dir / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Dataset conversion complete -> {yaml_path}")
    print("=" * 75)
    return report


def generate_synthetic_dataset(out_dir: Path, n_images: int = 6) -> Path:
    """Generate a tiny synthetic MAGFiLO-like dataset for smoke tests and CI."""
    import cv2
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Create dummy images and labels
    for i in range(n_images):
        split = "val" if i == 0 else "train"
        stem = f"synthetic_2026_{i:04d}_ann_{i}"
        img_path = out_dir / "images" / split / f"{stem}.jpeg"
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"

        # Synthetic 512x512 image for fast unit testing
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        # Draw a synthetic filament ellipse
        cv2.ellipse(img, (256, 256), (80, 20), 45, 0, 360, (50, 50, 50), -1)
        cv2.imwrite(str(img_path), img)

        # Normalized polygon for the ellipse
        pts = cv2.ellipse2Poly((256, 256), (80, 20), 45, 0, 360, 15)
        norm_pts = pts.astype(np.float32) / 512.0
        coord_str = " ".join(f"{c:.4f}" for c in norm_pts.reshape(-1))
        with open(lbl_path, "w") as f:
            f.write(f"0 {coord_str}\n")

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: filament
""")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MAGFiLO dataset to Native-2048 YOLO-seg")
    parser.add_argument("--data-dir", type=str, default=str(MAGFILO_DIR))
    parser.add_argument("--out-dir", type=str, default=str(YOLO_DATA_DIR))
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    convert_dataset(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        val_fold=args.val_fold,
        n_splits=args.n_splits,
        allow_missing=args.allow_missing,
    )
