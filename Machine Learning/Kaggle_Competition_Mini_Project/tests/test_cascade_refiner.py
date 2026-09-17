"""Unit tests for Cascade 2.0 Refiner Model, Loss, and Zero-Overlap Sanitizer."""

import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from cascade.model_refiner import build_crop_refiner, CompositeRefinerLoss
from cascade.infer_cascade import sanitize_zero_overlap, encode_mask_rle


def test_composite_refiner_loss():
    loss_fn = CompositeRefinerLoss(bce_weight=0.4, dice_weight=0.4, focal_weight=0.2)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = (torch.rand(2, 1, 64, 64) > 0.7).float()

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_build_crop_refiner_forward():
    # Test U-Net++ without pretraining for rapid offline unit check
    model = build_crop_refiner(arch="unetplusplus", encoder_name="resnet18", in_channels=3, encoder_weights=None)
    model.eval()

    dummy_inp = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy_inp)

    assert out.shape == (2, 1, 128, 128)


def test_sanitize_zero_overlap():
    H, W = 500, 500
    # Two heavily overlapping circular masks
    m1 = np.zeros((H, W), dtype=np.uint8)
    m2 = np.zeros((H, W), dtype=np.uint8)

    # m1: disk at (200, 200) radius 50
    # m2: disk at (220, 220) radius 50 (overlapping m1)
    y, x = np.ogrid[:H, :W]
    dist1 = np.sqrt((x - 200)**2 + (y - 200)**2)
    dist2 = np.sqrt((x - 220)**2 + (y - 220)**2)

    m1[dist1 <= 50] = 1
    m2[dist2 <= 50] = 1

    overlap_before = (m1 & m2).sum()
    assert overlap_before > 0, "Test setup error: masks must overlap initially"

    # Sanitizer with m1 having higher confidence
    sanitized, confs = sanitize_zero_overlap([m1, m2], [0.85, 0.70], min_area=100)

    assert len(sanitized) == 2
    overlap_after = (sanitized[0] & sanitized[1]).sum()
    assert overlap_after == 0, f"Expected 0 overlap, got {overlap_after}"

    # Verify RLE encode
    rle1 = encode_mask_rle(sanitized[0])
    rle2 = encode_mask_rle(sanitized[1])
    assert isinstance(rle1, str) and len(rle1) > 0
    assert isinstance(rle2, str) and len(rle2) > 0


if __name__ == "__main__":
    test_composite_refiner_loss()
    test_build_crop_refiner_forward()
    test_sanitize_zero_overlap()
    print("✅ All Cascade Refiner and Sanitizer unit tests PASSED!")
