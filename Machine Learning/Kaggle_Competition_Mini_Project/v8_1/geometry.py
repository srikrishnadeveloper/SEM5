"""Shared geometry helpers for Solar Filament instance crop operations.

Ensures strict square bounds without aspect-ratio distortion near image boundaries.
Both training (Stage 2) and inference (Stage 3) MUST use this exact contract.
"""

from typing import Sequence, Tuple


def compute_adaptive_crop_side(
    bbox: Sequence[int],
    crop_min: int = 256,
    crop_max: int = 512,
    crop_padding: float = 1.2,
) -> int:
    """Compute adaptive crop side length based on proposal dimensions.

    side = min(crop_max, max(crop_min, int(crop_padding * max(w, h))))
    """
    x1, y1, x2, y2 = bbox
    bw = max(0, int(x2 - x1))
    bh = max(0, int(y2 - y1))
    side = int(round(crop_padding * max(bw, bh)))
    return min(crop_max, max(crop_min, side))


def square_bounds_from_bbox(
    bbox: Sequence[int],
    image_height: int,
    image_width: int,
    side: int,
) -> Tuple[int, int, int, int]:
    """Calculate strictly square bounding coordinates [x1, y1, x2, y2] centered on bbox.

    Invariants guaranteed:
    1. 0 <= x1 < x2 <= image_width
    2. 0 <= y1 < y2 <= image_height
    3. (x2 - x1) == (y2 - y1) == final_side
    4. Centered on bbox center whenever boundary limits permit.
    5. When boundary limits clip left/right/top/bottom, shifts the entire window
       so the crop remains strictly square without aspect-ratio distortion.

    Args:
        bbox: [x1, y1, x2, y2] bounding box
        image_height: Source image height H (e.g. 2048)
        image_width: Source image width W (e.g. 2048)
        side: Desired crop side length (e.g. 256..512)

    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    final_side = min(int(side), int(image_height), int(image_width))
    if final_side <= 0:
        final_side = min(int(image_height), int(image_width))

    bx1, by1, bx2, by2 = bbox
    cx = (int(bx1) + int(bx2)) // 2
    cy = (int(by1) + int(by2)) // 2

    # Initial centered bounds
    x1 = cx - final_side // 2
    y1 = cy - final_side // 2
    x2 = x1 + final_side
    y2 = y1 + final_side

    # Boundary shifts for X
    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    elif x2 > image_width:
        shift = x2 - image_width
        x1 -= shift
        x2 -= shift

    # Boundary shifts for Y
    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 += shift
    elif y2 > image_height:
        shift = y2 - image_height
        y1 -= shift
        y2 -= shift

    # Final sanity clamp
    x1 = max(0, min(x1, image_width - final_side))
    x2 = x1 + final_side
    y1 = max(0, min(y1, image_height - final_side))
    y2 = y1 + final_side

    return int(x1), int(y1), int(x2), int(y2)
