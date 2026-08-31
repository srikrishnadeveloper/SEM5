"""
Unit and Integration Test Suite for V4 Grandmaster Edition
"""

import os
import sys
import tempfile
import json
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add v4 to path
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    Config,
    create_solar_features,
    detect_solar_disk_mask,
    create_2d_gaussian_window,
    split_filament_instances_watershed,
    build_model_by_spec,
    QuadHybridLoss,
    run
)


def test_gaussian_2d_tiling():
    print("\n--- Test 1: 2D Gaussian Tile Window & Reconstruction ---")
    g_win = create_2d_gaussian_window(1024)
    assert g_win.shape == (1024, 1024), f"Unexpected shape {g_win.shape}"
    assert np.isclose(g_win.max(), 1.0, atol=1e-4), f"Max should be ~1.0, got {g_win.max()}"
    assert g_win.min() > 0.0, f"Min should be positive, got {g_win.min()}"

    # Test 2048px grid reconstruction
    h, w = 2048, 2048
    p_acc = np.zeros((h, w), dtype=np.float32)
    w_acc = np.zeros((h, w), dtype=np.float32)
    steps = [0, 512, 1024]
    for y in steps:
        for x in steps:
            p_acc[y:y+1024, x:x+1024] += np.ones((1024, 1024), dtype=np.float32) * g_win
            w_acc[y:y+1024, x:x+1024] += g_win
    recon = p_acc / (w_acc + 1e-7)
    assert np.allclose(recon, 1.0, atol=1e-4), f"Reconstruction failed, min: {recon.min()}, max: {recon.max()}"
    print("[PASS] Test 1: Gaussian 2D Tiling & Seamless 2048px Reconstruction")


def test_watershed_instance_splitting():
    print("\n--- Test 2: Skeleton-Guided Watershed Instance Splitter ---")
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (20, 20), (180, 180), 1, 8)
    cv2.line(mask, (20, 180), (180, 20), 1, 8)

    instances = split_filament_instances_watershed(mask, min_area=50)
    assert len(instances) >= 1, f"Expected instances, got {len(instances)}"
    print(f"[PASS] Test 2: Watershed Instance Splitter (detected {len(instances)} instances)")


def test_tri_architecture_factory():
    print("\n--- Test 3: Tri-Architecture Grandmaster Model Factory ---")
    cfg = Config(classes=1, in_channels=3, encoder_weights=None)

    # 1. UnetPlusPlus
    m1 = build_model_by_spec("UnetPlusPlus", "resnet18", "scse", cfg)
    x = torch.randn(2, 3, 64, 64)
    out1 = m1(x)
    assert out1.shape == (2, 1, 64, 64), f"M1 shape mismatch: {out1.shape}"

    # 2. SegFormer (Vision Transformer)
    m2 = build_model_by_spec("Segformer", "mit_b2", "", cfg)
    out2 = m2(x)
    assert out2.shape == (2, 1, 64, 64), f"M2 shape mismatch: {out2.shape}"

    # 3. DeepLabV3Plus
    m3 = build_model_by_spec("DeepLabV3Plus", "resnet18", "", cfg)
    out3 = m3(x)
    assert out3.shape == (2, 1, 64, 64), f"M3 shape mismatch: {out3.shape}"

    print("[PASS] Test 3: Tri-Architecture Model Factory (UnetPlusPlus, SegFormer ViT, DeepLabV3Plus)")


def test_synthetic_end_to_end_v4():
    print("\n--- Test 4: End-to-End Synthetic Pipeline Execution ---")
    with tempfile.TemporaryDirectory(prefix="filament_v4_test_") as tmp_dir:
        train_dir = os.path.join(tmp_dir, "train", "train_images")
        test_dir = os.path.join(tmp_dir, "test", "test_images")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        images_info = []
        annotations = []
        ann_id = 1

        # Create 6 synthetic training images
        for i in range(6):
            fn = f"201{i%3}08000{i}20000Ch.jpg"
            img = np.zeros((128, 128), dtype=np.uint8) + 120
            cv2.circle(img, (64, 64), 58, 200, -1)
            cv2.line(img, (30 + i*5, 40), (90 + i*5, 80), 40, 4)
            cv2.imwrite(os.path.join(train_dir, fn), img)

            images_info.append({"id": i + 1, "file_name": fn, "height": 128, "width": 128})
            poly = [float(30 + i*5), 40.0, float(90 + i*5), 80.0, float(90 + i*5), 84.0, float(30 + i*5), 44.0]
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "segmentation": [poly],
                "area": 100.0
            })
            ann_id += 1

        # Create 2 synthetic test images
        for i in range(2):
            fn = f"201708000{i}120000Ch.jpg"
            img = np.zeros((128, 128), dtype=np.uint8) + 120
            cv2.circle(img, (64, 64), 58, 200, -1)
            cv2.line(img, (40, 40), (80, 80), 40, 4)
            cv2.imwrite(os.path.join(test_dir, fn), img)

        json_path = os.path.join(tmp_dir, "train", "annotations.json")
        with open(json_path, "w") as f:
            json.dump({
                "images": images_info,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "Filament"}]
            }, f)

        cfg = Config(
            exp_name="v4_smoke_test",
            base_dir=tmp_dir,
            train_images=train_dir,
            train_json=json_path,
            test_images=test_dir,
            output_dir=os.path.join(tmp_dir, "submissions"),
            models_dir=os.path.join(tmp_dir, "models"),
            img_size=64,
            full_res=128,
            batch_size=2,
            accum_steps=1,
            epochs=2,
            n_folds=2,
            train_all_folds=True,
            train_all_architectures=True,
            enable_gaussian_tiling=False,  # disabled for tiny 128px synthetic test
            enable_watershed_splitting=True,
            arch_1="UnetPlusPlus",
            encoder_1="resnet18",
            arch_2="Segformer",
            encoder_2="mit_b2",
            arch_3="DeepLabV3Plus",
            encoder_3="resnet18",
            encoder_weights=None,
            device="cpu",
            use_amp=False,
            num_workers=0
        )

        run(cfg)

        sub_file = os.path.join(cfg.output_dir, "submission.csv")
        assert os.path.exists(sub_file), f"Submission not found: {sub_file}"
        df_sub = pd.read_csv(sub_file)
        assert len(df_sub) >= 2, f"Expected rows, got {len(df_sub)}"
        assert "filament_id" in df_sub.columns and "segmentation_rle" in df_sub.columns
        print("[PASS] Test 4: End-to-End Tri-Architecture V4 Pipeline Execution")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("RUNNING V4 GRANDMASTER INTEGRATION TEST SUITE")
    print("=" * 65)
    test_gaussian_2d_tiling()
    test_watershed_instance_splitting()
    test_tri_architecture_factory()
    test_synthetic_end_to_end_v4()
    print("\n" + "=" * 65)
    print("ALL V4 GRANDMASTER TESTS PASSED 100%!")
    print("=" * 65 + "\n")
