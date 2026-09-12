"""
moonshot_2048/build_fulldata_notebook.py -- Build Fixed Kaggle Notebook (Full Data + Mosaic)

Critical Fixes vs Previous 0.36 Attempt:
  1. Train on ALL 1,154 annotator observations (no validation holdout)
  2. Enable mosaic augmentation (mosaic=1.0, close_mosaic=10)
  3. Safe epoch budget (50 epochs) to guarantee completion within Kaggle 12h limit
  4. Auto-detect existing trained checkpoint (fine-tunes if attached, or trains from scratch)
  5. Correct pycocotools RLE encoding: pure Fortran counts string (matches Host Evaluation V6)
  6. 1-indexed filament IDs: {stem}_1, {stem}_2, ... (matching host convention)
  7. Inference conf=0.30, min_area=50, max_det=100
"""

import json
import textwrap
from pathlib import Path


def build_notebook(out_path: Path):
    """Build a self-contained Kaggle notebook (.ipynb) for full-data training."""

    cells = []

    def add_code(source: str, cell_id: str = None):
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
        if cell_id:
            cell["id"] = cell_id
        cells.append(cell)

    def add_md(source: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        })

    # -- Cell 0: Title --
    add_md("""
        # Moonshot 2048 -- Full-Data Training (Target: 0.50+ PQ)

        **Key Upgrades:**
        1. **100% Training Data:** All 1,154 annotator observations (no 18% holdout loss).
        2. **Mosaic Augmentation:** `mosaic=1.0, close_mosaic=10` for scale & position invariance.
        3. **Safe 50 Epochs:** Tuned to complete comfortably within Kaggle's 12-hour session limit (~10.5 hours).
        4. **Checkpoint Auto-Detect:** Automatically fine-tunes if previous `best.pt` is attached, otherwise trains from base `yolov8l-seg.pt`.
        5. **Verified Evaluation Alignment:** Pure COCO Fortran RLE counts string matching Host Self-Evaluation V6.
        6. **Inference Recipe:** `conf=0.30`, `iou=0.00`, `min_area=50`, greedy zero-overlap sanitizer.
    """)

    # -- Cell 1: Install dependencies --
    add_code("""
        %%time
        # Install YOLO + dependencies
        !pip install -q ultralytics==8.3.145
        !pip install -q pycocotools

        import json
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    """, "install")

    # -- Cell 2: Discover data --
    add_code("""
        from pathlib import Path
        import os, shutil

        # Discover competition data
        CANDIDATES = [
            Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/kaggle/input/filament-segmentation-2026"),
        ]
        MAGFILO_DIR = next((p for p in CANDIDATES if p.exists()), None)
        assert MAGFILO_DIR is not None, "Competition data not found! Attach filament-segmentation-2026 as input."

        TRAIN_IMG_DIR = MAGFILO_DIR / "train" / "train_images"
        TEST_IMG_DIR = MAGFILO_DIR / "test" / "test_images"
        JSON_PATH = next((MAGFILO_DIR / "train").glob("*.json"))

        n_train = len(list(TRAIN_IMG_DIR.glob("*.jpeg")))
        n_test = len(list(TEST_IMG_DIR.glob("*.jpeg")))
        print(f"Train images: {n_train}")
        print(f"Test images:  {n_test}")
        print(f"Annotation JSON: {JSON_PATH}")
    """, "discover")

    # -- Cell 3: Convert ALL data to YOLO format (NO VAL SPLIT) --
    add_code("""
        %%time
        import json
        import numpy as np
        from collections import defaultdict, Counter

        # Parse COCO annotations
        with open(JSON_PATH) as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]
        print(f"Total COCO observations: {len(images)}")
        print(f"Total annotations: {len(annotations)}")

        # Build lookup
        img_lookup = {img["id"]: img for img in images}
        disk_filenames = {p.name for p in TRAIN_IMG_DIR.glob("*.jpeg")}

        # Extract normalized YOLO polygons (all categories -> class 0)
        valid_polys_by_imgid = defaultdict(list)
        cat_counts = Counter()
        skipped = 0

        for ann in annotations:
            cat_counts[ann.get("category_id", 1)] += 1
            img_id = ann["image_id"]
            if img_id not in img_lookup:
                continue
            img_rec = img_lookup[img_id]
            w, h = float(img_rec.get("width", 2048)), float(img_rec.get("height", 2048))

            for poly in ann.get("segmentation", []):
                if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
                    skipped += 1
                    continue
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
                    skipped += 1
                    continue
                # Shoelace area check
                x, y = pts[:, 0], pts[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                if area <= 0:
                    skipped += 1
                    continue
                norm_pts = pts.copy()
                norm_pts[:, 0] = np.clip(norm_pts[:, 0] / w, 0.0, 1.0)
                norm_pts[:, 1] = np.clip(norm_pts[:, 1] / h, 0.0, 1.0)
                valid_polys_by_imgid[img_id].append(norm_pts.reshape(-1).tolist())

        for cid in sorted(cat_counts):
            print(f"  Category {cid}: {cat_counts[cid]} annotations -> class 0 filament")
        print(f"  Valid polygons: {sum(len(v) for v in valid_polys_by_imgid.values())}, skipped: {skipped}")
    """, "parse_coco")

    # -- Cell 4: Write YOLO dataset (ALL images in train, minimal duplicates in val) --
    add_code("""
        %%time
        OUT_DIR = Path("/kaggle/working/data/yolo_fulldata2048")
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)

        for split in ["train", "val"]:
            (OUT_DIR / "images" / split).mkdir(parents=True)
            (OUT_DIR / "labels" / split).mkdir(parents=True)

        # Put ALL images into train.
        # YOLO requires a non-empty val directory, so we link 5 sample images there too.
        written_train = 0
        val_samples = []

        for img_rec in images:
            fn = img_rec["file_name"]
            if fn not in disk_filenames:
                continue
            img_id = img_rec["id"]
            stem = Path(fn).stem
            sample_stem = f"{stem}_ann_{img_id}"

            src = TRAIN_IMG_DIR / fn
            if not src.exists():
                continue

            dst_img = OUT_DIR / "images" / "train" / f"{sample_stem}.jpeg"
            dst_lbl = OUT_DIR / "labels" / "train" / f"{sample_stem}.txt"

            os.symlink(src, dst_img)

            polys = valid_polys_by_imgid.get(img_id, [])
            with open(dst_lbl, "w") as f:
                for p in polys:
                    f.write("0 " + " ".join(f"{c:.6f}" for c in p) + "\\n")

            written_train += 1
            if len(val_samples) < 5:
                val_samples.append((src, sample_stem, polys))

        # Write minimal val set (duplicates for YOLO validation loop)
        for src, sample_stem, polys in val_samples:
            dst_img = OUT_DIR / "images" / "val" / f"{sample_stem}.jpeg"
            dst_lbl = OUT_DIR / "labels" / "val" / f"{sample_stem}.txt"
            os.symlink(src, dst_img)
            with open(dst_lbl, "w") as f:
                for p in polys:
                    f.write("0 " + " ".join(f"{c:.6f}" for c in p) + "\\n")

        # Write data.yaml
        yaml_text = f\"\"\"path: {OUT_DIR.as_posix()}
        train: images/train
        val: images/val

        names:
          0: filament
        \"\"\"
        (OUT_DIR / "data.yaml").write_text(yaml_text)

        print(f"Training samples: {written_train} (100% of dataset)")
        print(f"Val samples: {len(val_samples)} (duplicates for YOLO loop)")
        print(f"Data config: {OUT_DIR / 'data.yaml'}")
    """, "write_yolo")

    # -- Cell 5: Train YOLOv8l-seg (FULL DATA, MOSAIC ON, 50 EPOCHS) --
    add_code("""
        %%time
        from ultralytics import YOLO

        # Check if an existing checkpoint is available to fine-tune from
        candidates = list(Path("/kaggle/input").glob("**/best.pt")) + list(Path("/kaggle/input").glob("**/*yolov8l*.pt"))
        if candidates:
            init_weights = str(candidates[0])
            print(f"Found existing weights: {init_weights} -> Fine-tuning on full data!")
        else:
            init_weights = "yolov8l-seg.pt"
            print(f"Starting from base weights: {init_weights}")

        model = YOLO(init_weights)

        results = model.train(
            data=str(OUT_DIR / "data.yaml"),
            imgsz=2048,
            epochs=50,       # 50 epochs safely fits within 12h Kaggle timeout (~10.5 hours)
            batch=2,
            patience=15,
            lr0=0.01,
            lrf=0.01,
            # -- CRITICAL UPGRADE: Mosaic Augmentation ON --
            mosaic=1.0,
            close_mosaic=10, # Turn off mosaic during last 10 epochs for fine boundary tuning
            # -- Augmentations tailored for solar chromosphere --
            degrees=10.0,
            flipud=0.5,
            fliplr=0.5,
            scale=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,
            # -- Execution settings --
            cache="ram",     # Cache all images in RAM as PyTorch tensors for 2x-3x faster epoch speed
            amp=True,
            workers=2,
            device=0,
            seed=42,
            project="/kaggle/working/runs",
            name="moonshot_fulldata",
            exist_ok=True,
            verbose=True,
        )

        # Locate trained weights
        BEST_WEIGHTS = Path("/kaggle/working/runs/moonshot_fulldata/weights/best.pt")
        LAST_WEIGHTS = Path("/kaggle/working/runs/moonshot_fulldata/weights/last.pt")
        WEIGHTS = BEST_WEIGHTS if BEST_WEIGHTS.exists() else LAST_WEIGHTS
        print(f"\\nBest weights: {WEIGHTS}")
        if WEIGHTS.exists():
            print(f"Weights size: {WEIGHTS.stat().st_size / 1e6:.1f} MB")
    """, "train")

    # -- Cell 6: Run inference on 180 test images (Exact Host Format) --
    add_code("""
        %%time
        import csv
        import json
        import cv2
        import numpy as np
        import pandas as pd
        import torch
        from pycocotools import mask as mask_utils

        # Load trained model
        model = YOLO(str(WEIGHTS))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_images = sorted(list(TEST_IMG_DIR.glob("*.jpeg")) + list(TEST_IMG_DIR.glob("*.jpg")))
        assert len(test_images) == 180, f"Expected 180 test images, got {len(test_images)}"
        print(f"Running pure PyTorch GPU inference & calibration sweep on {len(test_images)} test images using {device}...")

        # -- Precompute circular solar limb mask on GPU once --
        SOLAR_R_FRAC = 0.93
        cx, cy = 1024, 1024
        max_r = int(1024 * SOLAR_R_FRAC)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(2048, device=device),
            torch.arange(2048, device=device),
            indexing="ij",
        )
        solar_mask_gpu = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= (max_r ** 2)

        def encode_rle(mask_np):
            rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
            counts = rle["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")
            return counts

        # 1. First-pass: extract raw candidates per image at base threshold 0.20
        raw_candidates_per_image = []
        for idx_img, img_path in enumerate(test_images):
            stem = img_path.stem
            preds = model.predict(
                source=str(img_path),
                imgsz=2048,
                conf=0.20,
                iou=0.00,
                max_det=100,
                device=0 if device.type == "cuda" else "cpu",
                verbose=False,
            )
            pred = preds[0]
            if pred.masks is not None and len(pred.masks.data) > 0:
                raw_m = pred.masks.data.to(device) > 0.5
                raw_c = pred.boxes.conf.to(device)
                if raw_m.shape[1:] != (2048, 2048):
                    raw_m = torch.nn.functional.interpolate(
                        raw_m.unsqueeze(1).float(),
                        size=(2048, 2048),
                        mode="nearest",
                    ).squeeze(1) > 0.5
                raw_m = raw_m & solar_mask_gpu.unsqueeze(0)
                raw_candidates_per_image.append((stem, raw_m, raw_c))
            else:
                raw_candidates_per_image.append((stem, None, None))

            if (idx_img + 1) % 45 == 0:
                print(f"  [{idx_img + 1:3d}/180] images scanned on GPU...")

        # 2. Calibration Sweep across conf thresholds requested by ChatGPT Master
        SWEEP_CONFS = [0.20, 0.25, 0.30, 0.35]
        MIN_AREA = 50
        sweep_records = []
        submissions_by_conf = {}

        print("\\n" + "=" * 95)
        print("CALIBRATION SWEEP SUMMARY (ChatGPT Master Directive)")
        print("=" * 95)
        print(f"{'Conf':>6} | {'MinArea':>7} | {'Rows':>6} | {'Active Disks':>12} | {'Zero Disks':>10} | {'Mean/Disk':>10} | {'Target File':<25}")
        print("-" * 95)

        for conf_th in SWEEP_CONFS:
            rows_conf = []
            zero_count = 0
            for stem, raw_m, raw_c in raw_candidates_per_image:
                if raw_m is None or len(raw_m) == 0:
                    zero_count += 1
                    continue

                mask_sel = raw_c >= conf_th
                if mask_sel.sum() == 0:
                    zero_count += 1
                    continue

                m_sub = raw_m[mask_sel]
                c_sub = raw_c[mask_sel]

                order = torch.argsort(c_sub, descending=True)
                m_sub = m_sub[order]

                occupied = torch.zeros((2048, 2048), dtype=torch.bool, device=device)
                clean_masks = []
                for k in range(m_sub.shape[0]):
                    m = m_sub[k] & ~occupied
                    if m.sum().item() >= MIN_AREA:
                        occupied |= m
                        clean_masks.append(m)

                if len(clean_masks) == 0:
                    zero_count += 1
                    continue

                for i, m_gpu in enumerate(clean_masks):
                    fid = f"{stem}_{i+1}"
                    m_cpu = m_gpu.cpu().numpy().astype(np.uint8)
                    rows_conf.append({"filament_id": fid, "segmentation_rle": encode_rle(m_cpu)})

            active_disks = 180 - zero_count
            mean_inst = len(rows_conf) / 180.0
            conf_str = f"{int(conf_th * 100)}"
            out_name = f"submission_conf{conf_str}.csv"
            out_file = Path(f"/kaggle/working/{out_name}")
            pd.DataFrame(rows_conf).to_csv(out_file, index=False)
            submissions_by_conf[conf_th] = out_file

            sweep_records.append({
                "conf": conf_th,
                "min_area": MIN_AREA,
                "total_rows": len(rows_conf),
                "active_disks": active_disks,
                "zero_disks": zero_count,
                "mean_per_disk": round(mean_inst, 2),
                "file_path": str(out_file),
            })
            print(f"{conf_th:6.2f} | {MIN_AREA:7d} | {len(rows_conf):6d} | {active_disks:12d} | {zero_count:10d} | {mean_inst:10.2f} | {out_name:<25}")

        print("=" * 95)
        sweep_df = pd.DataFrame(sweep_records)
        sweep_df.to_csv("/kaggle/working/sweep_summary.csv", index=False)

        # 3. Set Primary Submission to conf=0.30 (Approved by Master)
        PRIMARY_CONF = 0.30
        primary_file = submissions_by_conf[PRIMARY_CONF]
        final_sub = Path("/kaggle/working/submission.csv")
        shutil.copy2(primary_file, final_sub)
        print(f"\\n[PRIMARY SUBMISSION FROZEN] -> {final_sub} (from {primary_file.name}, {sweep_df.loc[sweep_df['conf']==PRIMARY_CONF, 'total_rows'].values[0]} rows)")
    """, "inference")

    # -- Cell 7: Forensic Assertions on Primary Submission --
    add_code("""
        import pandas as pd
        from pycocotools import mask as mask_utils

        sub_path = Path("/kaggle/working/submission.csv")
        assert sub_path.exists(), "submission.csv does not exist!"
        df = pd.read_csv(sub_path)

        print(f"Primary submission: {sub_path}")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        assert list(df.columns) == ["filament_id", "segmentation_rle"], f"Invalid columns: {df.columns}"
        assert len(df) > 0, "Empty submission dataframe!"

        # Host Evaluation V6 decode test
        for i in range(min(5, len(df))):
            fid = df.iloc[i]["filament_id"]
            rle_str = df.iloc[i]["segmentation_rle"]
            rle_dict = {"size": [2048, 2048], "counts": rle_str}
            m = mask_utils.decode(rle_dict)
            assert m.shape == (2048, 2048), f"Row {i} ({fid}): bad shape {m.shape}"
            assert m.sum() > 0, f"Row {i} ({fid}): zero-area mask!"
            print(f"  Audit check row {i}: {fid}, area = {int(m.sum())} px - VALID")

        print("\\n[ALL FORENSIC AUDIT CHECKS PASSED] Ready for Kaggle leaderboard submission!")
    """, "write_csv")

    # -- Cell 8: Save weights & artifacts for download --
    add_code("""
        import shutil

        # Copy best and last weights
        dst_best = Path("/kaggle/working/best_fulldata.pt")
        if WEIGHTS.exists():
            shutil.copy2(WEIGHTS, dst_best)
            print(f"Best weights saved to: {dst_best} ({dst_best.stat().st_size / 1e6:.1f} MB)")

        if LAST_WEIGHTS.exists() and LAST_WEIGHTS != WEIGHTS:
            dst_last = Path("/kaggle/working/last_fulldata.pt")
            shutil.copy2(LAST_WEIGHTS, dst_last)
            print(f"Last weights saved to: {dst_last} ({dst_last.stat().st_size / 1e6:.1f} MB)")

        # Copy training metrics log if exists
        train_results = Path("/kaggle/working/runs/moonshot_fulldata/results.csv")
        if train_results.exists():
            shutil.copy2(train_results, Path("/kaggle/working/results.csv"))
    """, "save_weights")

    # -- Assemble notebook --
    nb = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")

    # Write kernel-metadata.json
    meta = {
        "id": "INSERT_YOUR_KAGGLE_USERNAME/moonshot-fulldata-2048",
        "title": "moonshot-fulldata-2048",
        "code_file": out_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": ["filament-segmentation-2026"],
        "keywords": ["solar", "filament", "segmentation", "yolov8l"],
    }
    meta_path = out_path.parent / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Kernel metadata: {meta_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(__file__).parent.parent / "notebooks" / "kaggle" / "moonshot_fulldata.ipynb"
    build_notebook(out)
