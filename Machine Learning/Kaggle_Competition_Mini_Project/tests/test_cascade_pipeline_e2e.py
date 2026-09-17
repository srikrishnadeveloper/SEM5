"""End-to-end integration test for Cascade 2.0 (Crop & Zoom) pipeline."""

import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import cv2
from cascade.geometry import compute_adaptive_crop_side, square_bounds_from_bbox, splice_crop_prob_to_canvas
from cascade.model_refiner import build_crop_refiner
from cascade.infer_cascade import refine_crops_batched, sanitize_zero_overlap, encode_mask_rle, get_solar_limb_mask


def test_cascade_pipeline_e2e():
    H, W = 2048, 2048
    # 1. Create synthetic solar disk with a simulated thin filament
    gray = np.full((H, W), 128, dtype=np.uint8)
    cv2.circle(gray, (1024, 1024), 950, 180, -1)  # solar disk

    # Draw synthetic curved thin filament (thickness 12px)
    pts = np.array([[800, 900], [850, 950], [920, 1000], [1050, 1020]], dtype=np.int32)
    cv2.polylines(gray, [pts], isClosed=False, color=30, thickness=12)

    # 2. Simulated Stage 1 coarse proposal
    # Bounding box around filament
    x1, y1, x2, y2 = 780, 880, 1070, 1040
    candidate_boxes = [[x1, y1, x2, y2]]
    coarse_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.polylines(coarse_mask, [pts], isClosed=False, color=1, thickness=16)
    coarse_masks = [coarse_mask]

    # 3. Initialize refiner model (lightweight for CPU test)
    device = torch.device("cpu")
    model = build_crop_refiner(arch="unet", encoder_name="resnet18", in_channels=3, encoder_weights=None).to(device)
    model.eval()

    # 4. Build 3-channel input
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(gray)
    ch_prior = (coarse_mask > 0).astype(np.uint8) * 255
    img_3ch = np.stack([gray, ch_clahe, ch_prior], axis=-1)

    # 5. Run Stage 2 batched crop refinement
    refined_masks = refine_crops_batched(
        refiner_model=model,
        img_3ch=img_3ch,
        candidate_boxes=candidate_boxes,
        coarse_masks=coarse_masks,
        device=device,
        target_size=256,
        use_tta=False,
        threshold=0.50,
    )

    assert len(refined_masks) == 1
    assert refined_masks[0].shape == (2048, 2048)

    # 6. Apply solar limb mask
    limb_mask = get_solar_limb_mask(2048, 2048, r_frac=0.93)
    clean_mask = refined_masks[0] * limb_mask
    assert clean_mask.shape == (2048, 2048)

    # 7. Zero overlap sanitizer
    final_masks, final_confs = sanitize_zero_overlap([clean_mask], [0.88], min_area=10)

    # 8. Verify RLE encoding
    if len(final_masks) > 0:
        rle = encode_mask_rle(final_masks[0])
        assert isinstance(rle, str)
        assert len(rle) > 0

    print("✅ Cascade 2.0 End-to-End Pipeline Integration Test PASSED!")


if __name__ == "__main__":
    test_cascade_pipeline_e2e()
