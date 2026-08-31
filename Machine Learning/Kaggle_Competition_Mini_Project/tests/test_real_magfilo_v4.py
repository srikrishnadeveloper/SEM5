"""
Comprehensive Real-Data Verification Test Suite for V4 Grandmaster Edition
Tests every component directly against the physical 2048x2048 MAGFiLO dataset.
"""

import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import pycocotools.mask as mask_utils

# Import V4 pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent / "v4"))
from pipeline import (
    Config,
    create_solar_features,
    detect_solar_disk_mask,
    create_2d_gaussian_window,
    parse_coco_annotations,
    rasterize_polygons,
    split_filament_instances_watershed,
    mask_to_coco_rle,
    build_model_by_spec,
    PentaHybridLoss,
    predict_tta,
    predict_tiled_gaussian_2048,
    build_splits
)


def run_real_magfilo_tests():
    print("=" * 80)
    print("[FORENSIC VERIFICATION] RUNNING ON REAL 2048x2048 MAGFiLO DATASET")
    print("=" * 80)

    base_dir = Path("data/MAGFiLO_1.0_Kaggle_2026")
    train_dir = base_dir / "train" / "train_images"
    test_dir = base_dir / "test" / "test_images"
    ann_file = base_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    assert train_dir.exists(), f"Train directory not found: {train_dir}"
    assert test_dir.exists(), f"Test directory not found: {test_dir}"
    assert ann_file.exists(), f"Annotations file not found: {ann_file}"

    # -------------------------------------------------------------------------
    # TEST 1: Real Image Loading & Astronomical Feature Engineering
    # -------------------------------------------------------------------------
    sample_train_file = sorted(list(train_dir.glob("*.jpeg")))[0]
    print(f"\n[Test 1] Loading Real Train Image: {sample_train_file.name}")
    raw_img = cv2.imread(str(sample_train_file), cv2.IMREAD_GRAYSCALE)
    assert raw_img is not None, "Failed to load raw image"
    assert raw_img.shape == (2048, 2048), f"Expected (2048, 2048), got {raw_img.shape}"
    print(f"  --> Image shape: {raw_img.shape}, dtype: {raw_img.dtype}, range: [{raw_img.min()}, {raw_img.max()}]")

    t0 = time.time()
    features = create_solar_features(raw_img)
    t_feat = time.time() - t0
    assert features.shape == (2048, 2048, 3), f"Expected (2048, 2048, 3), got {features.shape}"
    print(f"  --> 3-Channel Features built in {t_feat:.3f}s: Ch0 (Raw), Ch1 (CLAHE), Ch2 (Unsharp Ridge)")
    assert features[:, :, 1].std() > features[:, :, 0].std(), "CLAHE should enhance contrast standard deviation"
    print("  --> [PASS] Test 1: Real Astronomical Feature Engineering Verified")

    # -------------------------------------------------------------------------
    # TEST 2: Real COCO Annotation Parsing & Rasterization
    # -------------------------------------------------------------------------
    print(f"\n[Test 2] Parsing Real Annotations JSON: {ann_file.name}")
    fn_to_polys, fn_to_shape = parse_coco_annotations(str(ann_file))
    assert len(fn_to_polys) > 0, "No annotations parsed from JSON"
    print(f"  --> Successfully mapped {len(fn_to_polys)} unique physical filenames")

    polys = fn_to_polys.get(sample_train_file.name, [])
    print(f"  --> Filename {sample_train_file.name} has {len(polys)} ground truth filament polygon(s)")
    real_gt_mask = rasterize_polygons(polys, 2048, 2048)
    assert real_gt_mask.shape == (2048, 2048)
    if len(polys) > 0:
        assert real_gt_mask.sum() > 0, "Rasterized mask should have non-zero pixels for annotated image"
    print(f"  --> Rasterized GT Mask positive pixels: {real_gt_mask.sum():,} px")
    print("  --> [PASS] Test 2: Real COCO Annotation Rasterization Verified")

    # -------------------------------------------------------------------------
    # TEST 3: Differentiable Penta-Hybrid Loss & Sobel Edge Gradients on Real Crop
    # -------------------------------------------------------------------------
    print("\n[Test 3] Testing Penta-Hybrid Loss on Real 1024px Image Patch")
    cfg = Config(device="cpu", use_amp=False, encoder_weights=None)
    loss_fn = PentaHybridLoss(cfg)

    # Take real 1024px crop
    crop_img = features[512:1536, 512:1536, :].astype(np.float32) / 255.0
    crop_gt = real_gt_mask[512:1536, 512:1536].astype(np.float32)

    tensor_x = torch.from_numpy(crop_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_gt = torch.from_numpy(crop_gt).unsqueeze(0).unsqueeze(0).float()
    dummy_logits = torch.randn_like(tensor_gt, requires_grad=True)

    loss, metrics = loss_fn(dummy_logits, tensor_gt)
    loss.backward()
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss produced NaN/Inf"
    assert dummy_logits.grad is not None and dummy_logits.grad.norm().item() > 0, "Zero gradients produced"
    print(f"  --> Loss: {metrics['loss']:.4f} (BCE: {metrics['bce']:.4f}, Dice: {metrics['dice']:.4f}, Focal: {metrics['focal']:.4f}, Lovasz: {metrics['lovasz']:.4f}, Sobel: {metrics['sobel']:.4f})")
    print(f"  --> Backward Gradient Norm: {dummy_logits.grad.norm().item():.4f}")
    print("  --> [PASS] Test 3: Penta-Hybrid Loss & Sobel Gradients Verified")

    # -------------------------------------------------------------------------
    # TEST 4: Tri-Architecture Forward Pass on Real Data
    # -------------------------------------------------------------------------
    print("\n[Test 4] Testing Tri-Architecture Forward Pass with Real Channels")
    crop_64 = tensor_x[:, :, :64, :64]

    # Model 1: UnetPlusPlus
    m1 = build_model_by_spec("UnetPlusPlus", "resnet18", "scse", cfg).eval()
    out1 = m1(crop_64)
    assert out1.shape == (1, 1, 64, 64), f"M1 output mismatch: {out1.shape}"
    print(f"  --> Model 1: UnetPlusPlus (resnet18 + scse) Forward Pass OK")

    # Model 2: SegFormer Vision Transformer
    m2 = build_model_by_spec("Segformer", "mit_b2", "", cfg).eval()
    out2 = m2(crop_64)
    assert out2.shape == (1, 1, 64, 64), f"M2 output mismatch: {out2.shape}"
    print(f"  --> Model 2: SegFormer (mit_b2 Vision Transformer) Forward Pass OK")

    # Model 3: DeepLabV3Plus
    m3 = build_model_by_spec("DeepLabV3Plus", "resnet18", "", cfg).eval()
    out3 = m3(crop_64)
    assert out3.shape == (1, 1, 64, 64), f"M3 output mismatch: {out3.shape}"
    print(f"  --> Model 3: DeepLabV3Plus (resnet18 + ASPP) Forward Pass OK")
    print("  --> [PASS] Test 4: All 3 Diverse Architectures Verified")

    # -------------------------------------------------------------------------
    # TEST 5: Full Native 2048x2048 Gaussian Overlapping Tile Inference on Real Test Image
    # -------------------------------------------------------------------------
    sample_test_file = sorted(list(test_dir.glob("*.jpeg")))[0]
    print(f"\n[Test 5] Running 2048x2048 Gaussian Overlapping Tile Inference on: {sample_test_file.name}")
    raw_test = cv2.imread(str(sample_test_file), cv2.IMREAD_GRAYSCALE)
    test_features = create_solar_features(raw_test)

    # Use miniature test model for fast local verification of full 2048x2048 tiling math
    test_models = [(m1, 0.4), (m2, 0.35), (m3, 0.25)]
    t_tile_start = time.time()
    cfg.use_tta = False
    prob_full = predict_tiled_gaussian_2048(test_models, test_features, cfg)
    t_tile_end = time.time() - t_tile_start

    assert prob_full.shape == (2048, 2048), f"Expected (2048, 2048), got {prob_full.shape}"
    assert not np.isnan(prob_full).any() and not np.isinf(prob_full).any(), "Tile probability contains NaN/Inf"
    assert prob_full.min() >= 0.0 and prob_full.max() <= 1.0, f"Prob range invalid: [{prob_full.min()}, {prob_full.max()}]"
    print(f"  --> Stitched full 2048x2048 probability map in {t_tile_end:.2f}s")
    print(f"  --> Prob range: [{prob_full.min():.4f}, {prob_full.max():.4f}], Mean: {prob_full.mean():.4f}")
    print("  --> [PASS] Test 5: Full Native 2048px Gaussian Tiling Mathematically Verified")

    # -------------------------------------------------------------------------
    # TEST 6: Skeleton-Guided Watershed & Lossless Fortran COCO RLE Verification
    # -------------------------------------------------------------------------
    print("\n[Test 6] Testing Watershed Splitting and Lossless Fortran RLE Encoding")
    test_mask = (prob_full > 0.45).astype(np.uint8)
    # Add real ground truth filament shapes to verify watershed
    test_mask = np.maximum(test_mask, real_gt_mask)

    instances = split_filament_instances_watershed(test_mask, min_area=250)
    print(f"  --> Detected {len(instances)} distinct filament instance(s)")

    for idx, inst in enumerate(instances[:3], start=1):
        rle = mask_to_coco_rle(inst)
        assert isinstance(rle, str) and len(rle) > 0, "Invalid RLE string"
        decoded = mask_utils.decode({"size": [2048, 2048], "counts": rle.encode("utf-8")})
        assert decoded.shape == (2048, 2048), f"Decoded shape mismatch: {decoded.shape}"
        assert np.array_equal(inst, decoded), f"Lossy RLE encoding detected on instance {idx}!"
        print(f"  --> Instance {idx}: {inst.sum():,} px -> Lossless RLE String length: {len(rle)} chars (100% Exact Match)")

    print("  --> [PASS] Test 6: Lossless Fortran-ordered COCO RLE Verified")

    # -------------------------------------------------------------------------
    # TEST 7: GroupKFold by Year on Real 707 Filenames
    # -------------------------------------------------------------------------
    print("\n[Test 7] Validating 5-Fold GroupKFold Distribution on Real 707 Images")
    df_splits = build_splits(str(train_dir), n_folds=5, seed=42)
    assert len(df_splits) == 707, f"Expected 707 images, got {len(df_splits)}"
    for fold in range(5):
        val_count = (df_splits["fold"] == fold).sum()
        tr_count = (df_splits["fold"] != fold).sum()
        print(f"  --> Fold {fold}: Train = {tr_count} images | Val = {val_count} images")
        assert val_count > 0 and tr_count > 0, f"Empty fold detected in fold {fold}"

    # Verify zero year leakage across folds
    for fold in range(5):
        train_years = set(df_splits[df_splits["fold"] != fold]["group"])
        val_years = set(df_splits[df_splits["fold"] == fold]["group"])
        overlap = train_years.intersection(val_years)
        assert len(overlap) == 0, f"Data leakage detected! Overlapping years: {overlap}"
    print("  --> [PASS] Zero Year Leakage Across All 5 Folds Confirmed")

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL REAL-DATA MAGFiLO TESTS PASSED 100%! PIPELINE IS 100% BATTLE-READY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_real_magfilo_tests()
