"""
Unit tests for v8_1/geometry.py strict square adaptive crop geometry.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v8_1.geometry import compute_adaptive_crop_side, square_bounds_from_bbox


def test_square_bounds_center():
    bbox = [950, 950, 1050, 1050]
    side = 300
    x1, y1, x2, y2 = square_bounds_from_bbox(bbox, 2048, 2048, side)
    assert x2 - x1 == side
    assert y2 - y1 == side
    assert x1 == 1000 - 150
    assert y1 == 1000 - 150
    print("[PASS] test_square_bounds_center")


def test_square_bounds_left_edge():
    bbox = [10, 500, 60, 600]
    side = 300
    x1, y1, x2, y2 = square_bounds_from_bbox(bbox, 2048, 2048, side)
    assert x1 == 0
    assert x2 == 300
    assert x2 - x1 == side
    assert y2 - y1 == side
    print("[PASS] test_square_bounds_left_edge")


def test_square_bounds_right_edge():
    bbox = [2000, 500, 2040, 600]
    side = 300
    x1, y1, x2, y2 = square_bounds_from_bbox(bbox, 2048, 2048, side)
    assert x2 == 2048
    assert x1 == 2048 - 300
    assert x2 - x1 == side
    assert y2 - y1 == side
    print("[PASS] test_square_bounds_right_edge")


def test_square_bounds_top_edge():
    bbox = [500, 10, 600, 60]
    side = 300
    x1, y1, x2, y2 = square_bounds_from_bbox(bbox, 2048, 2048, side)
    assert y1 == 0
    assert y2 == 300
    assert x2 - x1 == side
    assert y2 - y1 == side
    print("[PASS] test_square_bounds_top_edge")


def test_square_bounds_bottom_edge():
    bbox = [500, 2000, 600, 2040]
    side = 300
    x1, y1, x2, y2 = square_bounds_from_bbox(bbox, 2048, 2048, side)
    assert y2 == 2048
    assert y1 == 2048 - 300
    assert x2 - x1 == side
    assert y2 - y1 == side
    print("[PASS] test_square_bounds_bottom_edge")


def test_square_bounds_corners():
    side = 300
    # Top-Left
    x1, y1, x2, y2 = square_bounds_from_bbox([0, 0, 50, 50], 2048, 2048, side)
    assert (x1, y1, x2, y2) == (0, 0, 300, 300)
    # Top-Right
    x1, y1, x2, y2 = square_bounds_from_bbox([2000, 0, 2048, 50], 2048, 2048, side)
    assert (x1, y1, x2, y2) == (2048 - 300, 0, 2048, 300)
    # Bottom-Left
    x1, y1, x2, y2 = square_bounds_from_bbox([0, 2000, 50, 2048], 2048, 2048, side)
    assert (x1, y1, x2, y2) == (0, 2048 - 300, 300, 2048)
    # Bottom-Right
    x1, y1, x2, y2 = square_bounds_from_bbox([2000, 2000, 2048, 2048], 2048, 2048, side)
    assert (x1, y1, x2, y2) == (2048 - 300, 2048 - 300, 2048, 2048)
    print("[PASS] test_square_bounds_corners (TL, TR, BL, BR)")


def test_adaptive_crop_sizing():
    # Small bbox -> clamped to crop_min=256
    s1 = compute_adaptive_crop_side([100, 100, 150, 150])
    assert s1 == 256
    # Medium bbox -> 1.2 * max(w, h)
    s2 = compute_adaptive_crop_side([100, 100, 400, 200])
    assert s2 == int(1.2 * 300)  # 360
    # Large bbox -> clamped to crop_max=512
    s3 = compute_adaptive_crop_side([100, 100, 700, 700])
    assert s3 == 512
    print("[PASS] test_adaptive_crop_sizing (min, scaled, capped)")


def test_paste_back_shape_consistency():
    h, w = 2048, 2048
    side = 384
    bbox = [1900, 1900, 2000, 2000]
    x1, y1, x2, y2 = square_bounds_from_bbox(bbox, h, w, side)
    crop_mask = np.ones((side, side), dtype=np.uint8)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = crop_mask
    assert full_mask.sum() == side * side
    assert full_mask.shape == (2048, 2048)
    print("[PASS] test_paste_back_shape_consistency")


if __name__ == "__main__":
    test_square_bounds_center()
    test_square_bounds_left_edge()
    test_square_bounds_right_edge()
    test_square_bounds_top_edge()
    test_square_bounds_bottom_edge()
    test_square_bounds_corners()
    test_adaptive_crop_sizing()
    test_paste_back_shape_consistency()
    print("\n=== ALL GEOMETRY TESTS PASSED ===")
