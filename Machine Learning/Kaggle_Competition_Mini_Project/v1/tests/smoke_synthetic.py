"""Cheap synthetic end-to-end smoke test for notebooks/filament_top50_single.py.

Uses tiny (96x96) synthetic images so the WHOLE pipeline (cache -> train ->
OOF -> threshold search -> submit -> ensemble) runs in well under a minute
on CPU, for every architecture/feature combination a user might enable.
This is deliberately cheap: it is not meant to produce a good score, only to
catch crashes / shape mismatches / wrong-file-extension type bugs fast.

Usage:
    python tests/smoke_synthetic.py
"""
import json
import os
import shutil
import sys
import traceback

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
NB_DIR = os.path.join(PROJECT, "notebooks")
SCRATCH = os.path.join(HERE, "_scratch_synth")

FULL = 256  # stand-in for the real 2048, keeps Hough/radial-flatten cheap
N_TRAIN, N_TEST = 10, 4


def build_scratch_dataset():
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH, ignore_errors=True)
    base = os.path.join(SCRATCH, "MAGFiLO_1.0_Kaggle_2026")
    tr_img = os.path.join(base, "train", "train_images")
    te_img = os.path.join(base, "test", "test_images")
    os.makedirs(tr_img); os.makedirs(te_img)

    rng = np.random.default_rng(0)
    images, annotations = [], []
    ann_id = 1
    years = ["2011", "2012", "2013"]

    def disk_image(seed):
        r = np.random.default_rng(seed)
        img = np.zeros((FULL, FULL), np.uint8)
        cv2.circle(img, (FULL // 2, FULL // 2), int(FULL * 0.45), 200, -1)
        img = cv2.add(img, (r.random((FULL, FULL)) * 20).astype(np.uint8))
        return img

    for i in range(N_TRAIN):
        fn = f"{years[i % len(years)]}0{i % 9 + 1}15_sample_{i:03d}.jpeg"  # real ext
        img = disk_image(i)
        polys = []
        for j in range(rng.integers(1, 3)):
            cx = int(rng.integers(int(FULL * 0.3), int(FULL * 0.7)))
            cy = int(rng.integers(int(FULL * 0.3), int(FULL * 0.7)))
            w, h = int(rng.integers(15, 30)), int(rng.integers(4, 8))
            pts = np.array([[cx - w, cy - h], [cx + w, cy - h], [cx + w, cy + h], [cx - w, cy + h]], np.int32)
            cv2.fillPoly(img, [pts], 60)
            polys.append([float(v) for p in pts for v in p])
        cv2.imwrite(os.path.join(tr_img, fn), img)
        img_id = i
        images.append({"id": img_id, "file_name": fn, "height": FULL, "width": FULL})
        for poly in polys:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                               "segmentation": [poly], "area": 100.0,
                               "bbox": [0, 0, 10, 10], "iscrowd": 0})
            ann_id += 1

    json.dump({"images": images, "annotations": annotations,
              "categories": [{"id": 1, "name": "filament"}, {"id": 4, "name": "Ambiguous"}]},
              open(os.path.join(base, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), "w"))

    for i in range(N_TEST):
        img = disk_image(100 + i)
        if i < N_TEST - 1:
            cv2.fillPoly(img, [np.array([[80, 80], [120, 80], [120, 88], [80, 88]], np.int32)], 60)
        cv2.imwrite(os.path.join(te_img, f"2020011{i}_test_{i:03d}.jpeg"), img)  # real ext

    print(f"[scratch] {N_TRAIN} train / {N_TEST} test synthetic images @ {FULL} -> {base}")
    return base


def run_case(name, base, overrides):
    print(f"\n{'=' * 70}\nCASE: {name}\n{'=' * 70}")
    sys.path.insert(0, NB_DIR)
    import importlib
    if "filament_top50_single" in sys.modules:
        importlib.reload(sys.modules["filament_top50_single"])
    import filament_top50_single as P

    cfg = P.CFG
    cfg.base = base
    cfg.out_dir = os.path.join(SCRATCH, "out_" + name)
    cfg = P.apply_preset(cfg, "fast")
    cfg.full_res = FULL
    cfg.img_size = 64
    cfg.cache_res = 64
    cfg.encoder = "timm-efficientnet-b0"
    cfg.batch_size = 2
    cfg.accum = 1
    cfg.epochs = 1
    cfg.warmup_epochs = 1
    cfg.patience = 5
    cfg.n_folds = 3
    cfg.fold = 0
    cfg.num_workers = 0
    cfg.miou_eval_res = 32
    cfg.min_area = 4
    cfg.threshold = 0.3
    cfg.no_filament_prob = 0.02
    cfg.search_thresholds = np.arange(0.2, 0.65, 0.2)
    cfg.search_min_areas = (4, 16)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg = P.resolve(cfg)

    res = P.run(cfg, do_train=True, do_search=True, do_submit=True)
    df = res["submission"]
    assert list(df.columns) == ["filament_id", "segmentation_rle"], f"[{name}] wrong columns"

    import pycocotools.mask as mu
    if len(df):
        rle = {"size": [cfg.full_res, cfg.full_res], "counts": df.iloc[0]["segmentation_rle"].encode()}
        dec = mu.decode(rle)
        assert dec.shape == (cfg.full_res, cfg.full_res), f"[{name}] RLE decode shape mismatch: {dec.shape}"

    # regression check: ensemble path must find the .jpeg test images
    d1 = os.path.join(cfg.prob_dir, f"test_fold_{cfg.fold}")
    ens = P.ensemble_probs([d1, d1], os.path.join(cfg.prob_dir, "ens"))
    df2 = P.submission_from_prob_dir(cfg, ens, res["threshold"], res["min_area"])
    assert list(df2.columns) == ["filament_id", "segmentation_rle"], f"[{name}] ensemble wrong columns"

    print(f"[{name}] OK - submission rows: {len(df)}, ensemble rows: {len(df2)}")
    return True


def main():
    base = build_scratch_dataset()
    cases = {
        "unet_default": {},
        "unet_copy_paste_imagenet_norm_crf": dict(
            use_copy_paste=True, copy_paste_prob=1.0, imagenet_norm=True, use_crf=True,
        ),
        "edgeattnet_1ch": dict(arch="edgeattnet", in_channels=1),
    }

    failures = []
    for name, overrides in cases.items():
        try:
            run_case(name, base, overrides)
        except Exception:
            failures.append(name)
            print(f"\n[{name}] FAILED:\n{traceback.format_exc()}")

    print(f"\n{'=' * 70}")
    if failures:
        print(f"SMOKE TEST FAILED for: {failures}")
        sys.exit(1)
    print("ALL SYNTHETIC SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
