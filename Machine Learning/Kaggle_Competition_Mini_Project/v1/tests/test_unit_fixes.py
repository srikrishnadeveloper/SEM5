"""Cheap, no-training unit tests for the bugs found in a static review of
notebooks/filament_top50_single.py. These run in well under a second each -
no cache building, no training loop, no real/synthetic images.

Usage:
    python tests/test_unit_fixes.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
NB_DIR = os.path.join(PROJECT, "notebooks")
sys.path.insert(0, NB_DIR)

import filament_top50_single as P  # noqa: E402


def test_load_for_inference_matches_train_norm():
    """load_for_inference() must use the SAME mean/std as build_transforms()
    for both the (0.5,0.5,0.5) and ImageNet-stat branches, and for both
    in_channels=1 and in_channels=3. This was bug #1 (silently ignored
    cfg.imagenet_norm at inference time)."""
    img = (np.random.rand(32, 32) * 255).astype(np.uint8)
    tmp_path = os.path.join(HERE, "_tmp_infer.png")
    import cv2
    cv2.imwrite(tmp_path, img)
    try:
        for in_channels in (1, 3):
            for imagenet_norm in (False, True):
                P.CFG.in_channels = in_channels
                P.CFG.imagenet_norm = imagenet_norm
                P.CFG.img_size = 32
                x, shape = P.load_for_inference(P.CFG, tmp_path)
                assert x.shape == (1, in_channels, 32, 32), (in_channels, imagenet_norm, x.shape)
                assert shape == (32, 32)
                assert torch_isfinite(x), "load_for_inference produced non-finite values"
                if imagenet_norm:
                    # ImageNet-normalised channels must NOT all be identical
                    # (each channel used a different mean/std) when in_channels==3
                    if in_channels == 3:
                        assert not np.allclose(x[0, 0].numpy(), x[0, 1].numpy()), \
                            "ImageNet norm should differ per channel"
                else:
                    # (0.5,0.5,0.5) norm: all channels must be identical (same source pixel)
                    for c in range(1, in_channels):
                        assert np.allclose(x[0, 0].numpy(), x[0, c].numpy())
    finally:
        os.remove(tmp_path)
    print("PASS: load_for_inference_matches_train_norm")


def torch_isfinite(t):
    import torch
    return bool(torch.isfinite(t).all())


def test_build_transforms_imagenet_norm_1channel_no_crash():
    """Bug #2: build_transforms() used a hardcoded 3-tuple mean/std for
    ImageNet normalisation regardless of cfg.in_channels, which crashed
    Albumentations when in_channels=1 (e.g. the edgeattnet preset)."""
    P.CFG.in_channels = 1
    P.CFG.imagenet_norm = True
    tf = P.build_transforms(P.CFG, train=False)
    img = np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    out = tf(image=img, mask=mask)
    assert out["image"].shape[0] == 1
    print("PASS: build_transforms_imagenet_norm_1channel_no_crash")


def test_submission_from_prob_dir_finds_jpeg():
    """Bug #3: submission_from_prob_dir() hardcoded '.jpg' when looking up the
    source test image for disk-masking/CRF, but MAGFiLO test images are
    '.jpeg'. Verify the extension-search logic finds a real .jpeg file."""
    import glob
    stage = os.path.join(HERE, "_tmp_ext_check")
    os.makedirs(os.path.join(stage, "test_images"), exist_ok=True)
    try:
        import cv2
        img = (np.random.rand(16, 16) * 255).astype(np.uint8)
        jpeg_path = os.path.join(stage, "test_images", "sample.jpeg")
        cv2.imwrite(jpeg_path, img)

        cfg = P.CFG
        cfg.test_images = os.path.join(stage, "test_images")
        stem = "sample"
        found = next((c for ext in (".jpg", ".jpeg", ".png")
                     if os.path.exists(c := os.path.join(cfg.test_images, stem + ext))), None)
        assert found is not None and found.endswith(".jpeg"), \
            f"extension search failed to find the real .jpeg file: {found}"
    finally:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
    print("PASS: submission_from_prob_dir_finds_jpeg")


def test_notebook_download_cell_counts_jpeg():
    """Bug #4: the generated notebook's data-download cell only globbed
    '*.jpg' then asserted the count > 0, which would crash on every Colab
    run against the real (.jpeg) dataset. Verify the FIXED source counts
    both extensions."""
    build_script = os.path.join(NB_DIR, "build_single_notebook.py")
    src = open(build_script, encoding="utf-8").read()
    assert '"*.jpeg"' in src, "build_single_notebook.py no longer globs .jpeg - regression!"
    assert 'IMG_EXTS' in src, "expected the multi-extension IMG_EXTS fix to be present"
    print("PASS: notebook_download_cell_counts_jpeg")


def test_edgeattnet_preset_uses_1_channel():
    """The edgeattnet preset should use in_channels=1 (no ImageNet stem to
    benefit from 3-channel replication)."""
    cfg = P.apply_preset(P.CFG, "edgeattnet")
    assert cfg.arch == "edgeattnet"
    assert cfg.in_channels == 1, f"expected in_channels=1, got {cfg.in_channels}"
    print("PASS: edgeattnet_preset_uses_1_channel")


if __name__ == "__main__":
    test_load_for_inference_matches_train_norm()
    test_build_transforms_imagenet_norm_1channel_no_crash()
    test_submission_from_prob_dir_finds_jpeg()
    test_notebook_download_cell_counts_jpeg()
    test_edgeattnet_preset_uses_1_channel()
    print("\nALL UNIT TESTS PASSED")
