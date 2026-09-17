"""Unit tests for Cascade 2.0 geometry calculations."""

import pytest
import numpy as np
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.geometry import (
    compute_adaptive_crop_side,
    square_bounds_from_bbox,
    splice_crop_prob_to_canvas,
)


def test_compute_adaptive_crop_side():
    # Small bbox (30x40) -> should be clamped to crop_min=256
    s1 = compute_adaptive_crop_side([100, 100, 130, 140], crop_min=256, crop_max=2048, crop_padding=1.25)
    assert s1 == 256, f"Expected 256, got {s1}"

    # Medium bbox (300x200) -> 300 * 1.25 = 375
    s2 = compute_adaptive_crop_side([500, 500, 800, 700], crop_min=256, crop_max=2048, crop_padding=1.25)
    assert s2 == 375, f"Expected 375, got {s2}"

    # Large elongated filament (900x120) -> 900 * 1.25 = 1125 (NOT truncated like V8.1's 512!)
    s3 = compute_adaptive_crop_side([100, 100, 1000, 220], crop_min=256, crop_max=2048, crop_padding=1.25)
    assert s3 == 1125, f"Expected 1125, got {s3}"

    # Giant filament (1800x500) -> 1800 * 1.25 = 2250 -> clamped to 2048
    s4 = compute_adaptive_crop_side([50, 50, 1850, 550], crop_min=256, crop_max=2048, crop_padding=1.25)
    assert s4 == 2048, f"Expected 2048, got {s4}"


def test_square_bounds_from_bbox():
    # Central box: side 400 around center (1000, 1000)
    x1, y1, x2, y2 = square_bounds_from_bbox([950, 950, 1050, 1050], 2048, 2048, side=400)
    assert (x2 - x1) == 400
    assert (y2 - y1) == 400
    assert x1 == 800 and y1 == 800
    assert x2 == 1200 and y2 == 1200

    # Near left-top corner: (20, 30, 80, 90) -> shift to prevent negative coords
    x1, y1, x2, y2 = square_bounds_from_bbox([20, 30, 80, 90], 2048, 2048, side=300)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 2048 and y2 <= 2048
    assert (x2 - x1) == 300
    assert (y2 - y1) == 300

    # Near right-bottom corner: (2000, 2010, 2040, 2040)
    x1, y1, x2, y2 = square_bounds_from_bbox([2000, 2010, 2040, 2040], 2048, 2048, side=300)
    assert x1 >= 0 and y1 >= 0
    assert x2 == 2048 and y2 == 2048
    assert (x2 - x1) == 300
    assert (y2 - y1) == 300


def test_splice_crop_prob_to_canvas():
    crop_prob = np.ones((384, 384), dtype=np.float32)
    bounds = (500, 600, 900, 1000)  # width 400, height 400
    canvas = splice_crop_prob_to_canvas(crop_prob, bounds, (2048, 2048))

    assert canvas.shape == (2048, 2048)
    assert np.allclose(canvas[600:1000, 500:900], 1.0)
    assert canvas[0:500, :].sum() == 0
    assert canvas[1000:, :].sum() == 0
    assert canvas[:, 0:500].sum() == 0
    assert canvas[:, 900:].sum() == 0


if __name__ == "__main__":
    test_compute_adaptive_crop_side()
    test_square_bounds_from_bbox()
    test_splice_crop_prob_to_canvas()
    print("✅ All Cascade Geometry unit tests PASSED!")
