"""
Unit & Integration Test Suite for V3 Beast Mode Pipeline
"""

import os
import sys
import tempfile
import shutil
import json
import cv2
import numpy as np
import pandas as pd
import torch

from pipeline import (
    Config,
    create_solar_features,
    detect_solar_disk_mask,
    LovaszHingeLoss,
    QuadHybridLoss,
    build_model_by_spec,
    parse_coco_annotations,
    rasterize_polygons,
    build_splits,
    run_heterogeneous_test_submission,
    mask_to_coco_rle,
    run
)


def create_synthetic_dataset(temp_dir: str):
    train_dir = os.path.join(temp_dir, "train", "train_images")
    test_dir = os.path.join(temp_dir, "test", "test_images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_files = [
        "201405000120000Ch.jpg",
        "201405001120000Ch.jpg",
        "201506000120000Ch.jpg",
        "201506001120000Ch.jpg",
        "201607000120000Ch.jpg",
        "201607001120000Ch.jpg",
    ]
    test_files = [
        "201708000120000Ch.jpg",
        "201708001120000Ch.jpg",
    ]

    images_ann = []
    annotations_ann = []
    ann_id = 1

    for idx, fn in enumerate(train_files, start=1):
        img = np.full((128, 128), 120, dtype=np.uint8)
        # Add circular solar disk
        cv2.circle(img, (64, 64), 58, 180, -1)
        # Add filament structure
        cv2.ellipse(img, (64, 64), (30, 8), 45, 0, 360, 40, -1)
        cv2.imwrite(os.path.join(train_dir, fn), img)

        images_ann.append({"id": idx, "file_name": fn, "height": 128, "width": 128})
        poly = [50.0, 50.0, 78.0, 50.0, 78.0, 78.0, 50.0, 78.0]
        annotations_ann.append({
            "id": ann_id,
            "image_id": idx,
            "category_id": 1,
            "segmentation": [poly]
        })
        ann_id += 1

    for fn in test_files:
        img = np.full((128, 128), 120, dtype=np.uint8)
        cv2.circle(img, (64, 64), 58, 180, -1)
        cv2.ellipse(img, (64, 64), (25, 6), 30, 0, 360, 40, -1)
        cv2.imwrite(os.path.join(test_dir, fn), img)

    json_path = os.path.join(temp_dir, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"images": images_ann, "annotations": annotations_ann, "categories": [{"id": 1, "name": "filament"}]}, f)

    return train_dir, test_dir, json_path


def test_v3_pipeline():
    print("\n" + "=" * 65)
    print("RUNNING V3 BEAST MODE INTEGRATION TEST SUITE")
    print("=" * 65)

    temp_dir = tempfile.mkdtemp(prefix="filament_v3_test_")
    try:
        train_dir, test_dir, json_path = create_synthetic_dataset(temp_dir)

        # Test 1: Astronomical 3-Channel Feature Engineering
        test_gray = np.full((64, 64), 128, dtype=np.uint8)
        cv2.circle(test_gray, (32, 32), 25, 200, -1)
        feat_map = create_solar_features(test_gray)
        assert feat_map.shape == (64, 64, 3), f"Wrong feature map shape: {feat_map.shape}"
        assert feat_map.dtype == np.uint8, "Feature map must be uint8"
        print("[PASS] Test 1: Astronomical 3-Channel Feature Engineering (Raw + CLAHE + Unsharp)")

        # Test 2: Solar Disk Limb Suppression
        mask = detect_solar_disk_mask(test_gray, margin=1.0)
        assert mask.shape == (64, 64)
        assert mask[32, 32] == 1, "Center of solar disk must be inside mask"
        assert mask[0, 0] == 0, "Corner of frame must be outside solar disk"
        print("[PASS] Test 2: Solar Disk Limb Boundary Detection & Suppression")

        # Test 3: Lovasz-Hinge Loss & Quad-Hybrid Loss Gradients
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        lov_fn = LovaszHingeLoss()
        lov_loss = lov_fn(logits, targets)
        assert lov_loss.item() >= 0.0
        lov_loss.backward()
        assert logits.grad is not None, "Lovasz loss must produce valid gradients"

        cfg = Config(base_dir=temp_dir, img_size=64, full_res=128, batch_size=2, epochs=2, encoder_weights=None)
        loss_fn = QuadHybridLoss(cfg)
        logits2 = torch.randn(2, 1, 32, 32, requires_grad=True)
        q_loss, q_metrics = loss_fn(logits2, targets)
        assert "lovasz" in q_metrics and "dice" in q_metrics and "focal" in q_metrics
        q_loss.backward()
        assert logits2.grad is not None
        print("[PASS] Test 3: Lovász-Hinge & Quad-Hybrid Loss Backward Gradients")

        # Test 4: Heterogeneous Model Factory
        m1 = build_model_by_spec("UnetPlusPlus", "resnet18", "", cfg)
        m2 = build_model_by_spec("DeepLabV3Plus", "resnet18", "", cfg)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            out1 = m1(torch.randn(1, 3, 64, 64))
            out2 = m2(torch.randn(1, 3, 64, 64))
        assert out1.shape == (1, 1, 64, 64)
        assert out2.shape == (1, 1, 64, 64)
        print("[PASS] Test 4: Heterogeneous Multi-Architecture Model Factory")

        # Test 5: Full End-to-End V3 Pipeline Execution
        test_cfg = Config(
            exp_name="v3_smoke_test",
            base_dir=temp_dir,
            train_images=train_dir,
            train_json=json_path,
            test_images=test_dir,
            output_dir=os.path.join(temp_dir, "submissions"),
            models_dir=os.path.join(temp_dir, "models"),
            arch_1="UnetPlusPlus",
            encoder_1="resnet18",
            arch_2="DeepLabV3Plus",
            encoder_2="resnet18",
            encoder_weights=None,
            img_size=64,
            full_res=128,
            batch_size=2,
            accum_steps=1,
            epochs=2,
            n_folds=2,
            val_fold=0,
            train_all_folds=True,
            train_both_architectures=True,
            device="cpu",
            use_amp=False,
            num_workers=0
        )
        run(test_cfg)

        sub_file = os.path.join(test_cfg.output_dir, "submission.csv")
        assert os.path.exists(sub_file), "Submission file was not generated"
        sub_df = pd.read_csv(sub_file)
        assert len(sub_df) >= 2, "Submission must contain rows for all test images"
        assert "filament_id" in sub_df.columns and "segmentation_rle" in sub_df.columns
        print("[PASS] Test 5: Full End-to-End V3 Multi-Architecture Ensemble Execution")

        print("\n" + "=" * 65)
        print("ALL V3 BEAST MODE TESTS PASSED 100%!")
        print("=" * 65)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_v3_pipeline()
