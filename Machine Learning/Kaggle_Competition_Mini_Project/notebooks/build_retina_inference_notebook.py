"""
notebooks/build_retina_inference_notebook.py

Builds a self-contained, one-shot Kaggle notebook that performs native-2048
inference with a trained YOLOv8l-seg checkpoint using retina_masks=True.

The generated notebook:
  - Installs torch==2.5.1+cu118, ultralytics, pycocotools
  - Loads /kaggle/input/moonshot-best-pt/best.pt
  - Runs YOLO predict at imgsz=2048 with retina_masks=True
  - Applies a GPU greedy zero-overlap sanitizer and an r=0.93 solar limb mask
  - Generates 9 submission CSVs for conf in {0.25, 0.27, 0.29}
    and min_area in {30, 50, 80}
  - Validates exactly 180 test images, correct RLE encoding, and zero overlap

Outputs:
  notebooks/kaggle/retina_inference.ipynb
  notebooks/kaggle/kernel-metadata.json
"""

import json
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NB_PATH = PROJECT_ROOT / "notebooks" / "kaggle" / "retina_inference.ipynb"


def build_notebook(out_path: Path):
    """Construct the .ipynb file and the companion kernel-metadata.json."""
    cells = []

    def add_md(source: str):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(source).strip().splitlines(keepends=True),
            }
        )

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

    # ------------------------------------------------------------------
    # Cell 0: Markdown overview
    # ------------------------------------------------------------------
    add_md(
        """
        # RetinaMask 2048: YOLOv8l-seg One-Shot Inference Sweep

        Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup / Kaggle)

        - Single forward pass per test disk at native 2048x2048 with `retina_masks=True`
        - Greedy GPU zero-overlap sanitizer + solar limb mask (r = 0.93)
        - Generates 9 submission CSVs: conf in {0.25, 0.27, 0.29} x min_area in {30, 50, 80}
        - Loads weights from `/kaggle/input/moonshot-best-pt/best.pt`
        - Validates exactly 180 test images, COCO RLE encoding, and zero overlap

        Kaggle runtime: GPU T4, Internet ON.
        """
    )

    # ------------------------------------------------------------------
    # Cell 1: Install dependencies
    # ------------------------------------------------------------------
    add_code(
        """
        import os
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        !pip install -q torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
        !pip install -q ultralytics pycocotools

        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} (Count: {torch.cuda.device_count()})")
        """,
        "install",
    )

    # ------------------------------------------------------------------
    # Cell 2: Locate dataset, weights, and assert 180 test images
    # ------------------------------------------------------------------
    add_code(
        """
        from pathlib import Path
        from ultralytics import YOLO

        # Discover competition data
        CANDIDATES = [
            Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/kaggle/input/filament-segmentation-2026"),
        ]
        MAGFILO_DIR = next((p for p in CANDIDATES if p.exists()), None)
        assert MAGFILO_DIR is not None, "FATAL: Competition dataset not attached. Add filament-segmentation-2026 as input."

        TEST_IMG_DIR = MAGFILO_DIR / "test" / "test_images"
        assert TEST_IMG_DIR.exists(), f"FATAL: Test image directory not found: {TEST_IMG_DIR}"

        test_images = sorted(list(TEST_IMG_DIR.glob("*.jpeg")) + list(TEST_IMG_DIR.glob("*.jpg")))
        print(f"Test images discovered: {len(test_images)}")
        assert len(test_images) == 180, f"FATAL: Expected exactly 180 test images, found {len(test_images)}"

        # Resolve the trained checkpoint
        WEIGHTS = Path("/kaggle/input/moonshot-best-pt/best.pt")
        if not WEIGHTS.exists():
            candidates = sorted(list(Path("/kaggle/input").glob("**/best.pt")), key=lambda p: p.stat().st_mtime)
            assert candidates, "FATAL: best.pt not found. Attach the moonshot-best-pt dataset as input."
            WEIGHTS = candidates[-1]
        print(f"Loading model from: {WEIGHTS} ({WEIGHTS.stat().st_size / (1024**2):.2f} MB)")

        model = YOLO(str(WEIGHTS))
        print("Model loaded successfully.")
        """,
        "load",
    )

    # ------------------------------------------------------------------
    # Cell 3: Precompute solar limb mask and helper functions
    # ------------------------------------------------------------------
    add_code(
        """
        import numpy as np
        import pandas as pd
        import torch
        from pycocotools import mask as mask_utils

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Solar disk center for native 2048x2048
        CX, CY = 1024, 1024
        SOLAR_R_FRAC = 0.93
        MAX_R = int(1024 * SOLAR_R_FRAC)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(2048, device=device, dtype=torch.float32),
            torch.arange(2048, device=device, dtype=torch.float32),
            indexing="ij",
        )
        solar_mask_gpu = ((x_grid - CX) ** 2 + (y_grid - CY) ** 2) <= (MAX_R ** 2)

        def binarize_masks(m):
            """Robustly binarize YOLO mask tensor regardless of whether it is
            already binary, a probability in [0, 1], or a pre-sigmoid logit."""
            if m.dtype == torch.float:
                if m.max() <= 1.0 and m.min() >= 0.0:
                    return m > 0.5
                return m > 0.0
            return m.bool()

        def encode_rle(mask_np):
            """Encode a 2D binary mask to a COCO Fortran RLE counts string."""
            mask_f = np.asfortranarray(mask_np.astype(np.uint8))
            rle = mask_utils.encode(mask_f)
            counts = rle["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")
            return counts

        def sanitize_one_image(masks, confs, conf_th, min_area, solar_mask):
            """Greedy confidence-ordered pixel carve on GPU tensors.

            Args:
                masks:  (N, H, W) torch.BoolTensor, already clipped to solar disk
                confs:  (N,) torch.FloatTensor of confidences
                conf_th: float, confidence threshold
                min_area: int, post-carve minimum area in pixels
                solar_mask: (H, W) torch.BoolTensor, solar disk mask
            Returns:
                list of (filament_id, rle_counts) tuples for this image
            """
            sel = confs >= conf_th
            if sel.sum() == 0:
                return []

            m_sub = masks[sel] & solar_mask.unsqueeze(0)
            c_sub = confs[sel]

            # Sort by descending confidence
            order = torch.argsort(c_sub, descending=True)
            m_sub = m_sub[order]

            occupied = torch.zeros((2048, 2048), dtype=torch.bool, device=m_sub.device)
            rows = []

            for k in range(m_sub.shape[0]):
                m = m_sub[k] & ~occupied
                if m.sum().item() >= min_area:
                    occupied |= m
                    m_cpu = m.cpu().numpy().astype(np.uint8)
                    rows.append(m_cpu)

            return rows

        print("Solar limb mask precomputed on GPU.")
        print(f"Solar disk radius: {MAX_R} px (r_frac = {SOLAR_R_FRAC})")
        """,
        "helpers",
    )

    # ------------------------------------------------------------------
    # Cell 4: First-pass inference with retina_masks=True
    # ------------------------------------------------------------------
    add_code(
        """
        %%time
        # One forward pass per image at the lowest requested confidence (0.25).
        # Confidences 0.27 and 0.29 are obtained by post-filtering the same raw
        # candidates because the model uses confidence-sorted NMS (iou=0.00).
        
        candidates_by_image = []

        for idx, img_path in enumerate(test_images):
            stem = img_path.stem

            preds = model.predict(
                source=str(img_path),
                imgsz=2048,
                conf=0.25,
                iou=0.00,
                max_det=100,
                device="0",
                verbose=False,
                retina_masks=True,  # KEY FIX: native 2048x2048 smooth masks
            )

            pred = preds[0]
            if pred.masks is not None and len(pred.masks.data) > 0:
                # With retina_masks=True, masks.data has shape (N, 2048, 2048)
                raw_masks = binarize_masks(pred.masks.data).to(device)
                raw_confs = pred.boxes.conf.to(device)

                # Apply solar limb mask immediately to keep memory clean
                raw_masks = raw_masks & solar_mask_gpu.unsqueeze(0)
                candidates_by_image.append((stem, raw_masks, raw_confs))
            else:
                candidates_by_image.append((stem, None, None))

            if (idx + 1) % 30 == 0 or (idx + 1) == len(test_images):
                print(f"  [{idx + 1:3d}/{len(test_images)}] images scanned")

        print(f"First-pass inference complete: {len(candidates_by_image)} images")
        """,
        "inference",
    )

    # ------------------------------------------------------------------
    # Cell 5: Generate 9 submission CSVs (conf x min_area sweep)
    # ------------------------------------------------------------------
    add_code(
        """
        %%time
        import shutil

        CONF_THRESHOLDS = [0.25, 0.27, 0.29]
        MIN_AREAS = [30, 50, 80]

        # Accumulate rows per (conf, min_area) combination
        rows_by_combo = {(c, a): [] for c in CONF_THRESHOLDS for a in MIN_AREAS}

        for stem, raw_masks, raw_confs in candidates_by_image:
            if raw_masks is None or raw_masks.shape[0] == 0:
                continue

            for conf_th in CONF_THRESHOLDS:
                for min_area in MIN_AREAS:
                    clean_masks = sanitize_one_image(
                        raw_masks, raw_confs, conf_th, min_area, solar_mask_gpu
                    )
                    for i, m_cpu in enumerate(clean_masks):
                        fid = f"{stem}_{i + 1}"  # 1-indexed
                        rle = encode_rle(m_cpu)
                        rows_by_combo[(conf_th, min_area)].append({
                            "filament_id": fid,
                            "segmentation_rle": rle,
                        })

        sweep_records = []
        for conf_th in CONF_THRESHOLDS:
            for min_area in MIN_AREAS:
                rows = rows_by_combo[(conf_th, min_area)]
                out_name = f"submission_conf{int(conf_th * 100):02d}_area{min_area:02d}.csv"
                out_path = Path(f"/kaggle/working/{out_name}")
                df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
                df.to_csv(out_path, index=False)

                active_stems = {r["filament_id"].rsplit("_", 1)[0] for r in rows}
                zero_disks = 180 - len(active_stems)

                sweep_records.append({
                    "conf": conf_th,
                    "min_area": min_area,
                    "rows": len(df),
                    "active_disks": len(active_stems),
                    "zero_disks": zero_disks,
                    "file": out_name,
                    "path": str(out_path),
                })
                print(f"{out_name}: {len(df):5d} rows | {len(active_stems):3d} active disks | {zero_disks:3d} zero disks")

        # Save sweep summary
        sweep_df = pd.DataFrame(sweep_records)
        sweep_df.to_csv("/kaggle/working/retina_sweep_summary.csv", index=False)

        # Promote the conf=0.25, min_area=50 operating point to the primary submission.csv
        primary = next(r for r in sweep_records if r["conf"] == 0.25 and r["min_area"] == 50)
        primary_path = Path(primary["path"])
        shutil.copy2(primary_path, "/kaggle/working/submission.csv")
        print(f"\nPrimary submission written: /kaggle/working/submission.csv (from {primary['file']})")
        """,
        "sweep",
    )

    # ------------------------------------------------------------------
    # Cell 6: Audit / validation
    # ------------------------------------------------------------------
    add_code(
        """
        from collections import defaultdict

        def audit_submission(path: Path, strict: bool = False):
            df = pd.read_csv(path)
            assert list(df.columns) == ["filament_id", "segmentation_rle"], f"Invalid columns in {path}: {list(df.columns)}"
            assert df["filament_id"].is_unique, f"Duplicate filament_id in {path}"

            # Group by image stem
            stem_to_rles = defaultdict(list)
            for _, row in df.iterrows():
                stem = row["filament_id"].rsplit("_", 1)[0]
                stem_to_rles[stem].append(row["segmentation_rle"])

            # Every RLE must decode to (2048, 2048) with positive area
            total_area = 0
            for stem, rles in stem_to_rles.items():
                for rle in rles:
                    rle_dict = {"size": [2048, 2048], "counts": rle}
                    m = mask_utils.decode(rle_dict)
                    assert m.shape == (2048, 2048), f"Bad decoded shape in {path}: {m.shape}"
                    assert m.sum() > 0, f"Zero-area mask in {path} under stem {stem}"
                    total_area += m.sum()

                # Zero-overlap check: sum of individual areas must equal union area
                if len(rles) > 1 and strict:
                    occupied = np.zeros((2048, 2048), dtype=np.uint8)
                    individual_sum = 0
                    for rle in rles:
                        rle_dict = {"size": [2048, 2048], "counts": rle}
                        m = mask_utils.decode(rle_dict)
                        individual_sum += int(m.sum())
                        occupied = np.maximum(occupied, m)
                    assert individual_sum == int(occupied.sum()), (
                        f"Overlap violation in {path} on stem {stem}: "
                        f"sum={individual_sum}, union={occupied.sum()}"
                    )

            return len(df), total_area

        print("=" * 70)
        print("AUDIT: 9 submission CSVs")
        print("=" * 70)
        for rec in sweep_records:
            n_rows, area = audit_submission(Path(rec["path"]), strict=False)
            print(f"{rec['file']:40s} | rows={n_rows:5d} | total_area={area:>12d}")

        # Strict pairwise audit on the primary submission
        print("\nStrict zero-overlap audit on primary submission.csv ...")
        n_rows, area = audit_submission(Path("/kaggle/working/submission.csv"), strict=True)
        print(f"Primary submission.csv: {n_rows} rows, total_area={area}, ZERO OVERLAP VERIFIED")

        # Confirm test stems are from the official set
        official_stems = {p.stem for p in test_images}
        for rec in sweep_records:
            df = pd.read_csv(rec["path"])
            pred_stems = {fid.rsplit("_", 1)[0] for fid in df["filament_id"].tolist()}
            unknown = pred_stems - official_stems
            assert not unknown, f"Unknown stems in {rec['file']}: {unknown}"

        print("\nAll 9 submissions passed audit: correct columns, positive RLEs, known stems.")
        print("Primary submission.csv passed strict zero-overlap verification.")
        """,
        "audit",
    )

    # ------------------------------------------------------------------
    # Assemble notebook
    # ------------------------------------------------------------------
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
            "accelerator": "GPU",
            "gpuClass": "standard",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Write kernel-metadata.json
    # ------------------------------------------------------------------
    meta = {
        "id": "your-username/retina-inference-2048",
        "title": "retina-inference-2048",
        "code_file": out_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": ["your-username/moonshot-best-pt"],
        "keywords": ["solar", "filament", "segmentation", "yolov8l", "retina-masks"],
    }
    meta_path = out_path.parent / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Kernel metadata written: {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the RetinaMask 2048 Kaggle notebook")
    parser.add_argument("--out", type=str, default=str(DEFAULT_NB_PATH))
    args = parser.parse_args()

    build_notebook(Path(args.out))
