"""
Convert COCO Annotations to YOLO-seg format for MAGFiLO Dataset.

Requirements (Grok Phase B Verdict):
1. Key off JSON image_id, not pooled filename (one annotator observation = one YOLO sample).
2. No polygon OR-merging (eliminates 11.6 polygon/image label inflation bug).
3. Include categories 1, 2, 3, 4 as solar filaments.
4. Single class 'filament' (id 0) for YOLO-seg.
5. Verification and reporting:
   - JSON images count
   - JPEGs on disk count
   - file_name missing on disk
   - JPEGs with zero annotations
   - polygons skipped (too few points, zero/degenerate area)
6. Exit non-zero if > 5% of JSON file_names are missing unless --allow-missing.
7. GroupKFold by year prefix to avoid multi-annotator and temporal leakage.
8. Write data/yolo_seg/data.yaml and conversion_report.json.
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def polygon_area(pts: np.ndarray) -> float:
    """Compute polygon area using Shoelace formula. pts: (N, 2)."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def materialize_file(src: Path, dst: Path, mode: str = "auto") -> str:
    """Safely materialize src at dst.
    Preferred order:
      1. symlink (robust cross-mount support on Linux/Kaggle without copying 700MB)
      2. hardlink (same filesystem zero-copy)
      3. shutil.copy2 (universal fallback)
    Verifies destination exists and is non-empty.
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

    raise RuntimeError(f"FATAL: Failed to materialize {src} to {dst} with any method.")


def convert_coco_to_yolo(
    data_dir: Path,
    out_dir: Path,
    val_fold: int = 0,
    n_splits: int = 5,
    allow_missing: bool = False,
    link_mode: str = "auto",  # 'hardlink', 'symlink', 'copy'
) -> dict:
    """Perform annotator-aware COCO to YOLO-seg conversion."""

    train_dir = data_dir / "train"
    img_dir = train_dir / "train_images"
    json_candidates = list(train_dir.glob("*.json")) + list(data_dir.glob("*.json"))
    if not json_candidates:
        raise FileNotFoundError(f"No annotation JSON found in {data_dir}")
    json_path = json_candidates[0]

    print("=" * 70)
    print("COCO -> YOLO-SEG CONVERTER (Annotator-Aware, Zero-OR)")
    print("=" * 70)
    print(f"  Annotation JSON: {json_path}")
    print(f"  Image directory: {img_dir}")
    print(f"  Output root:     {out_dir}")
    print(f"  Val fold:        {val_fold} (of {n_splits})")
    print()

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    json_images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    print("PRE-FLIGHT DATASET AUDIT:")
    print(f"  1. JSON images count:          {len(json_images)}")
    
    # Check JPEGs on disk
    jpegs_on_disk = list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg"))
    disk_filenames = {p.name for p in jpegs_on_disk}
    print(f"  2. JPEGs on disk:              {len(jpegs_on_disk)}")

    # Check missing file_names
    json_file_names = {img["file_name"] for img in json_images}
    missing_files = sorted(list(json_file_names - disk_filenames))
    pct_missing = (len(missing_files) / len(json_file_names) * 100.0) if json_file_names else 0.0
    print(f"  3. file_name missing on disk:  {len(missing_files)} ({pct_missing:.2f}%)")
    if missing_files:
        print(f"     Sample missing: {missing_files[:5]}")

    if pct_missing > 5.0 and not allow_missing:
        print(f"❌ ERROR: More than 5% of JSON file_names are missing ({pct_missing:.2f}%). Exiting.")
        sys.exit(1)

    # Annotations per image_id and per physical file_name
    img_id_to_anns = defaultdict(list)
    fn_to_anns = defaultdict(list)
    img_id_to_record = {img["id"]: img for img in json_images}

    for ann in annotations:
        img_id = ann["image_id"]
        img_id_to_anns[img_id].append(ann)
        if img_id in img_id_to_record:
            fn_to_anns[img_id_to_record[img_id]["file_name"]].append(ann)

    zero_ann_jpegs = [fn for fn in disk_filenames if len(fn_to_anns[fn]) == 0]
    print(f"  4. JPEGs with zero annotations: {len(zero_ann_jpegs)}")

    # Category distribution audit
    from collections import Counter
    cat_counts = Counter(ann.get("category_id") for ann in annotations)
    cat_1_count = cat_counts.get(1, 0)
    cat_2_count = cat_counts.get(2, 0)
    cat_3_count = cat_counts.get(3, 0)
    cat_4_count = cat_counts.get(4, 0)

    # Polygon validation audit (All categories 1, 2, 3, 4 are valid filaments for class 0)
    skipped_points = 0
    skipped_area = 0
    valid_polys_count = 0

    valid_anns_by_img_id = defaultdict(list)

    for ann in annotations:
        # All categories are mapped to class 0 'filament'
        img_id = ann["image_id"]
        if img_id not in img_id_to_record:
            continue
        
        img_rec = img_id_to_record[img_id]
        h = float(img_rec.get("height", 2048))
        w = float(img_rec.get("width", 2048))

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

            # Normalized coordinates for YOLO: [0, 1]
            norm_pts = pts.copy()
            norm_pts[:, 0] = np.clip(norm_pts[:, 0] / w, 0.0, 1.0)
            norm_pts[:, 1] = np.clip(norm_pts[:, 1] / h, 0.0, 1.0)

            valid_polys_count += 1
            valid_anns_by_img_id[img_id].append(norm_pts.reshape(-1).tolist())

    print(f"  5. Categories & Polygons audit (All categories -> Class 0 'filament'):")
    print(f"     - Category 1 (Left):           {cat_1_count}")
    print(f"     - Category 2 (Right):          {cat_2_count}")
    print(f"     - Category 3 (Unidentifiable): {cat_3_count}")
    print(f"     - Category 4 (Ambiguous):      {cat_4_count}")
    print(f"     - Valid filaments:             {valid_polys_count}")
    print(f"     - Skipped (<6 pts/coords):     {skipped_points}")
    print(f"     - Skipped (zero area):         {skipped_area}")
    print()

    # Determine train/val split using GroupKFold by year
    # Each sample corresponds to a single JSON image_id
    sample_records = []
    for img_rec in json_images:
        fn = img_rec["file_name"]
        if fn not in disk_filenames:
            continue
        img_id = img_rec["id"]
        # Group by year prefix (first 4 digits of filename)
        year_group = fn[:4] if fn[:4].isdigit() else "0000"
        sample_records.append({
            "image_id": img_id,
            "file_name": fn,
            "group": year_group,
            "n_polys": len(valid_anns_by_img_id[img_id]),
        })

    df_samples = pd.DataFrame(sample_records)
    gkf = GroupKFold(n_splits=n_splits)
    df_samples["fold"] = -1
    for fold_idx, (trn_idx, v_idx) in enumerate(gkf.split(df_samples, groups=df_samples["group"])):
        df_samples.loc[v_idx, "fold"] = fold_idx

    val_mask = df_samples["fold"] == val_fold
    train_samples = df_samples[~val_mask]
    val_samples = df_samples[val_mask]

    print(f"SPLIT ASSIGNMENT (GroupKFold by year, val_fold={val_fold}):")
    print(f"  Train samples (annotator observations): {len(train_samples)}")
    print(f"  Val samples (annotator observations):   {len(val_samples)}")
    print(f"  Train unique physical JPEGs:            {train_samples['file_name'].nunique()}")
    print(f"  Val unique physical JPEGs:              {val_samples['file_name'].nunique()}")
    print()

    # Setup directories
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Write samples
    print("WRITING SAMPLES TO DISK...")
    
    # Test link / materialization strategy
    test_src = img_dir / sample_records[0]["file_name"]
    test_dst = out_dir / "_test_materialize"
    detected_method = materialize_file(test_src, test_dst, mode=link_mode)
    if test_dst.exists():
        test_dst.unlink()

    print(f"  File materialization method verified: {detected_method}")

    written_counts = {"train": 0, "val": 0}

    for idx, row in df_samples.iterrows():
        split = "val" if row["fold"] == val_fold else "train"
        img_id = row["image_id"]
        fn = row["file_name"]
        stem = Path(fn).stem
        # Distinct sample name keyed off image_id
        sample_stem = f"{stem}_ann_{img_id}"

        src_img = img_dir / fn
        dst_img = out_dir / "images" / split / f"{sample_stem}.jpeg"
        dst_lbl = out_dir / "labels" / split / f"{sample_stem}.txt"

        # Safely materialize image with existence & non-zero size verification
        materialize_file(src_img, dst_img, mode=detected_method)
        if not (dst_img.exists() and dst_img.stat().st_size > 0):
            raise RuntimeError(f"FATAL: Materialized image {dst_img} is missing or 0 bytes!")

        # Write YOLO segmentation label file
        polys = valid_anns_by_img_id[img_id]
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            for p in polys:
                coord_str = " ".join(f"{c:.6f}" for c in p)
                lf.write(f"0 {coord_str}\n")

        written_counts[split] += 1

    print(f"  Wrote {written_counts['train']} train files, {written_counts['val']} val files")
    print()

    # Write data.yaml
    yaml_content = f"""# YOLO-seg dataset config for Solar Filament Segmentation 2026
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: filament
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  Written data.yaml: {yaml_path}")

    # Write conversion_report.json
    report = {
        "json_path": str(json_path),
        "json_images_count": len(json_images),
        "jpegs_on_disk_count": len(jpegs_on_disk),
        "unique_json_filenames": len(json_file_names),
        "missing_on_disk_count": len(missing_files),
        "missing_on_disk_pct": pct_missing,
        "zero_ann_jpegs_count": len(zero_ann_jpegs),
        "category_1_count": cat_1_count,
        "category_2_count": cat_2_count,
        "category_3_count": cat_3_count,
        "category_4_count": cat_4_count,
        "skipped_polygons_invalid_pts": skipped_points,
        "skipped_polygons_zero_area": skipped_area,
        "valid_polygons_count": valid_polys_count,
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
        "train_unique_jpegs": int(train_samples["file_name"].nunique()),
        "val_unique_jpegs": int(val_samples["file_name"].nunique()),
        "val_fold": val_fold,
        "n_splits": n_splits,
        "linking_method": detected_method,
        "classes": {"0": "filament"},
    }
    report_path = out_dir / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Written conversion_report.json: {report_path}")
    print("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser(description="COCO to YOLO-seg dataset converter")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/MAGFiLO_1.0_Kaggle_2026",
        help="Path to MAGFiLO competition root",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/yolo_seg",
        help="Path to output YOLO dataset",
    )
    parser.add_argument("--val-fold", type=int, default=0, help="Fold index to use as val")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing images above 5% threshold without error",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="auto",
        choices=["auto", "hardlink", "symlink", "copy"],
        help="How to place image files in YOLO folders",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    out_dir = PROJECT_ROOT / args.out_dir

    convert_coco_to_yolo(
        data_dir=data_dir,
        out_dir=out_dir,
        val_fold=args.val_fold,
        n_splits=args.n_splits,
        allow_missing=args.allow_missing,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()
