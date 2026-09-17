"""Cascade 2.0 (Crop & Zoom High-Resolution Refinement Engine).
Designed for Solar Filament Segmentation Challenge 2026.
"""

from cascade.geometry import (
    compute_adaptive_crop_side,
    square_bounds_from_bbox,
    splice_crop_prob_to_canvas,
)

__all__ = [
    "compute_adaptive_crop_side",
    "square_bounds_from_bbox",
    "splice_crop_prob_to_canvas",
]
