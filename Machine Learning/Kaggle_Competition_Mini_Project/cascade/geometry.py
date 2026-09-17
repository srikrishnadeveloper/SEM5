"""High-precision adaptive crop and coordinate projection geometry for Cascade 2.0.

Ensures:
1. Strict non-truncation: Large filaments (up to 2048px) are NEVER clipped.
2. Square aspect ratio: Avoids directional distortion during neural patch processing.
3. Exact bijective mapping: Crop coordinates cleanly project back to the 2048x2048 canvas.
"""

from typing import Sequence, Tuple
import numpy as np
import cv2


def compute_adaptive_crop_side(
    bbox: Sequence[int],
    crop_min: int = 256,
    crop_max: int = 2048,
    crop_padding: float = 1.25,
) -> int:
    """Compute adaptive square crop side length based on proposal dimensions.

    Guarantees that filaments of any length (up to 2048) are preserved with sufficient margin.
    """
    x1, y1, x2, y2 = bbox
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    max_dim = max(bw, bh)
    side = int(round(crop_padding * max_dim))
    return int(min(crop_max, max(crop_min, side)))


def square_bounds_from_bbox(
    bbox: Sequence[int],
    image_height: int = 2048,
    image_width: int = 2048,
    side: int = 384,
) -> Tuple[int, int, int, int]:
    """Calculate strictly square bounding coordinates [x1, y1, x2, y2] centered on bbox.

    Invariants guaranteed:
    1. 0 <= x1 < x2 <= image_width
    2. 0 <= y1 < y2 <= image_height
    3. (x2 - x1) == (y2 - y1) == final_side
    4. Centered on bbox center whenever boundary limits permit.
    5. When boundary limits clip left/right/top/bottom, shifts the entire window
       so the crop remains strictly square without aspect-ratio distortion.
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

    # Shift if out of X bounds
    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    elif x2 > image_width:
        shift = x2 - image_width
        x1 -= shift
        x2 -= shift

    # Shift if out of Y bounds
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


def splice_crop_prob_to_canvas(
    crop_prob: np.ndarray,
    crop_bounds: Tuple[int, int, int, int],
    canvas_shape: Tuple[int, int] = (2048, 2048),
) -> np.ndarray:
    """Project a refined crop probability map back to the full 2048x2048 canvas.

    Args:
        crop_prob: 2D float probability map from refiner network (e.g. 384x384).
        crop_bounds: (x1, y1, x2, y2) in canvas coordinates.
        canvas_shape: Full image shape (H, W).

    Returns:
        canvas_prob: 2D float array of shape canvas_shape with the resized crop spliced in.
    """
    H, W = canvas_shape
    canvas_prob = np.zeros((H, W), dtype=np.float32)
    x1, y1, x2, y2 = crop_bounds
    target_w = x2 - x1
    target_h = y2 - y1

    if target_w <= 0 or target_h <= 0:
        return canvas_prob

    # Bilinear resize crop_prob to original un-downsampled crop dimensions
    if crop_prob.shape != (target_h, target_w):
        resized_prob = cv2.resize(crop_prob, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_prob = crop_prob

    canvas_prob[y1:y2, x1:x2] = resized_prob
    return canvas_prob
