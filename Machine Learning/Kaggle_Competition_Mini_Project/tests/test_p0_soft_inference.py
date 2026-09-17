"""
tests/test_p0_soft_inference.py
Unit & integration test for the P0 Champion Soft-TTA & Soft-Ownership Pipeline.
Tests on real MAGFiLO test images using real best.pt weights.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pycocotools import mask as mask_utils
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

WEIGHTS_PATH = Path("C:/Users/srik2/Desktop/Filament_Colab_Run/best.pt")
TEST_IMAGES_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026/test/test_images")


def apply_solar_limb_mask(mask: np.ndarray, r_frac: float = 0.98) -> np.ndarray:
    """Zero out pixels outside the solar disk."""
    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    max_r = int((min(w, h) / 2.0) * r_frac)
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    mask[dist_sq > max_r ** 2] = 0
    return mask


def clean_connected_components(mask: np.ndarray, min_fragment_ratio: float = 0.15) -> np.ndarray:
    """
    Ensure mask topological integrity per MAGFiLO spec.
    Keeps the largest connected component and any secondary component with area >= min_fragment_ratio * largest_area.
    Prunes tiny disconnected boundary specks.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 2:  # 0 is background, 1 is single component
        return mask
    
    # stats[:, cv2.CC_STAT_AREA] has area of each component
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = np.max(areas)
    
    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    for i, area in enumerate(areas, start=1):
        if area == max_area or area >= (min_fragment_ratio * max_area):
            clean_mask[labels == i] = 1
    return clean_mask


def resolve_soft_overlaps(
    candidate_probs: List[np.ndarray],
    confidences: List[float],
    prob_threshold: float = 0.50,
    min_area: int = 50,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Soft-Overlap Ownership Resolver.
    Instead of hard greedy carving (occupied |= m) which punches holes in lower-priority masks,
    for any pixel contested by multiple candidates (P_i > tau and P_j > tau),
    assigns the pixel to argmax_k(P_k), preserving smooth continuous topology.
    """
    if not candidate_probs:
        return [], []

    # Initial binarization and candidate filtering
    valid_candidates = []
    for p, conf in zip(candidate_probs, confidences):
        p_clipped = apply_solar_limb_mask(p.copy())
        bin_m = (p_clipped > prob_threshold).astype(np.uint8)
        area = int(bin_m.sum())
        if area >= min_area:
            valid_candidates.append({
                "prob": p_clipped,
                "conf": float(conf),
                "area": area,
            })

    if not valid_candidates:
        return [], []

    # Sort descending by confidence
    valid_candidates.sort(key=lambda x: x["conf"], reverse=True)

    h, w = valid_candidates[0]["prob"].shape[:2]
    n_cands = len(valid_candidates)

    if n_cands == 1:
        m = (valid_candidates[0]["prob"] > prob_threshold).astype(np.uint8)
        m = clean_connected_components(m)
        if m.sum() >= min_area:
            return [m], [valid_candidates[0]["conf"]]
        return [], []

    # Stack candidate probabilities: shape (N, H, W)
    prob_stack = np.stack([c["prob"] for c in valid_candidates], axis=0)  # (N, H, W)
    
    # Active mask where at least one candidate exceeds threshold
    active_mask = (prob_stack > prob_threshold)  # (N, H, W)
    any_active = np.any(active_mask, axis=0)     # (H, W)

    # Argmax over candidates for each pixel
    # If pixel is active for at least one candidate, winner is argmax(P_k)
    winner_idx = np.argmax(prob_stack, axis=0)   # (H, W)

    final_masks = []
    final_confs = []

    for k in range(n_cands):
        # Pixel belongs to candidate k if k is winner AND P_k(x, y) > prob_threshold
        cand_mask = np.logical_and(winner_idx == k, active_mask[k]).astype(np.uint8)
        # Apply topological cleanup
        cand_mask = clean_connected_components(cand_mask)
        cand_area = int(cand_mask.sum())
        if cand_area >= min_area:
            final_masks.append(cand_mask)
            final_confs.append(valid_candidates[k]["conf"])

    # Strict Zero-Overlap Guarantee verification
    n_kept = len(final_masks)
    for i in range(n_kept):
        for j in range(i + 1, n_kept):
            shared = int(np.logical_and(final_masks[i], final_masks[j]).sum())
            if shared > 0:
                raise AssertionError(f"FATAL: Overlap check failed! {shared} shared pixels.")

    return final_masks, final_confs


def predict_single_view_soft(
    model: YOLO,
    img_np: np.ndarray,
    imgsz: int = 2048,
    conf_thresh: float = 0.20,
    nms_iou: float = 0.45,
    device: str = "cpu",
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Run YOLOv8-seg model and extract full-resolution continuous float probability maps.
    Returns: (list_of_prob_maps_2048, confidences, bboxes)
    """
    h_orig, w_orig = img_np.shape[:2]
    
    # Prepare input tensor: RGB, normalized [0, 1], shape (1, 3, H, W)
    img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img_t = img_t.to(device)

    with torch.no_grad():
        preds = model.model(img_t)
        # preds[0][0]: box+class+coeff tensor, preds[0][1]: prototype masks
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        
        # Run NMS
        det = non_max_suppression(
            preds[0][0],
            conf_thres=conf_thresh,
            iou_thres=nms_iou,
            classes=[0],
            max_det=300,
            nc=1,
        )[0]

    if det is None or len(det) == 0:
        return [], [], []

    # det shape: (N, 6 + 32) = (N, 38)
    bboxes = det[:, :4]       # (N, 4) in xyxy
    scores = det[:, 4].cpu().numpy().tolist()
    coeffs = det[:, 6:]       # (N, 32)

    # Compute raw mask logits: (N, 512, 512)
    c, mh, mw = protos[0].shape
    logits_512 = (coeffs @ protos[0].float().view(c, -1)).view(-1, mh, mw)

    # Bilinear upsample to native 2048x2048
    logits_2048 = F.interpolate(logits_512.unsqueeze(1), size=(h_orig, w_orig), mode="bilinear", align_corners=False).squeeze(1)

    # Convert to continuous probabilities
    probs_2048 = torch.sigmoid(logits_2048)

    # Crop to bounding box
    probs_2048 = ops.crop_mask(probs_2048, bboxes)

    prob_maps = [probs_2048[i].cpu().numpy() for i in range(len(probs_2048))]
    box_coords = [bboxes[i].cpu().numpy() for i in range(len(bboxes))]

    return prob_maps, scores, box_coords


def run_soft_tta_inference(
    model: YOLO,
    img_path: Path,
    conf_thresh: float = 0.20,
    prob_thresh: float = 0.50,
    min_area: int = 50,
    device: str = "cpu",
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run 3-view Soft TTA:
    View 0: Original
    View 1: Horizontal Flip (fliplr)
    View 2: Vertical Flip (flipud)
    Inverts flips and fuses continuous probability maps.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # View 0: Original
    p0, s0, b0 = predict_single_view_soft(model, img_rgb, conf_thresh=conf_thresh, device=device)

    # View 1: Horizontal Flip
    img_hflip = cv2.flip(img_rgb, 1)
    p1, s1, b1 = predict_single_view_soft(model, img_hflip, conf_thresh=conf_thresh, device=device)
    # Unflip horizontally
    p1_unflipped = [cv2.flip(p, 1) for p in p1]

    # View 2: Vertical Flip
    img_vflip = cv2.flip(img_rgb, 0)
    p2, s2, b2 = predict_single_view_soft(model, img_vflip, conf_thresh=conf_thresh, device=device)
    # Unflip vertically
    p2_unflipped = [cv2.flip(p, 0) for p in p2]

    # Fuse instances across views:
    # Use View 0 as anchors, find matching detections in View 1 and View 2 via mask IoU
    fused_probs = []
    fused_confs = []
    used_v1 = set()
    used_v2 = set()

    for i, (m0, c0) in enumerate(zip(p0, s0)):
        bin_m0 = (m0 > prob_thresh).astype(bool)
        area0 = bin_m0.sum()
        if area0 < min_area:
            continue
        
        # Match with v1
        matched_v1 = None
        best_iou1 = 0.30
        for j, m1 in enumerate(p1_unflipped):
            if j in used_v1: continue
            bin_m1 = (m1 > prob_thresh).astype(bool)
            inter = np.logical_and(bin_m0, bin_m1).sum()
            union = np.logical_or(bin_m0, bin_m1).sum()
            iou = inter / union if union > 0 else 0
            if iou > best_iou1:
                best_iou1 = iou
                matched_v1 = j

        # Match with v2
        matched_v2 = None
        best_iou2 = 0.30
        for k, m2 in enumerate(p2_unflipped):
            if k in used_v2: continue
            bin_m2 = (m2 > prob_thresh).astype(bool)
            inter = np.logical_and(bin_m0, bin_m2).sum()
            union = np.logical_or(bin_m0, bin_m2).sum()
            iou = inter / union if union > 0 else 0
            if iou > best_iou2:
                best_iou2 = iou
                matched_v2 = k

        # Fuse probability maps
        maps_to_average = [m0]
        confs_to_average = [c0]
        if matched_v1 is not None:
            maps_to_average.append(p1_unflipped[matched_v1])
            confs_to_average.append(s1[matched_v1])
            used_v1.add(matched_v1)
        if matched_v2 is not None:
            maps_to_average.append(p2_unflipped[matched_v2])
            confs_to_average.append(s2[matched_v2])
            used_v2.add(matched_v2)

        # Average continuous probabilities
        fused_p = np.mean(maps_to_average, axis=0)
        fused_c = np.mean(confs_to_average)
        fused_probs.append(fused_p)
        fused_confs.append(fused_c)

    # Add unmatched high-confidence detections from v1 and v2 (if confirmed by high conf)
    for j, (m1, c1) in enumerate(zip(p1_unflipped, s1)):
        if j not in used_v1 and c1 >= 0.40:
            if (m1 > prob_thresh).sum() >= min_area:
                fused_probs.append(m1)
                fused_confs.append(c1)

    for k, (m2, c2) in enumerate(zip(p2_unflipped, s2)):
        if k not in used_v2 and c2 >= 0.40:
            if (m2 > prob_thresh).sum() >= min_area:
                fused_probs.append(m2)
                fused_confs.append(c2)

    # Apply Soft-Overlap Ownership Resolver
    clean_masks, clean_confs = resolve_soft_overlaps(
        fused_probs, fused_confs, prob_threshold=prob_thresh, min_area=min_area
    )

    return clean_masks, clean_confs


def encode_mask_rle(mask: np.ndarray) -> str:
    """Encode binary mask into Fortran COCO RLE string."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


if __name__ == "__main__":
    print("=== Testing P0 Soft-TTA & Soft-Ownership Pipeline ===")
    assert WEIGHTS_PATH.exists(), f"Weights not found: {WEIGHTS_PATH}"
    assert TEST_IMAGES_DIR.exists(), f"Test dir not found: {TEST_IMAGES_DIR}"

    test_imgs = sorted(list(TEST_IMAGES_DIR.glob("*.jpeg")) + list(TEST_IMAGES_DIR.glob("*.jpg")))
    print(f"Discovered {len(test_imgs)} test images.")
    sample_img = test_imgs[0]
    print(f"Testing on sample image: {sample_img.name}")

    print("Loading model...")
    model = YOLO(str(WEIGHTS_PATH))

    t0 = time.time()
    masks, confs = run_soft_tta_inference(
        model=model,
        img_path=sample_img,
        conf_thresh=0.20,
        prob_thresh=0.45,
        min_area=50,
        device="cpu",
    )
    elapsed = time.time() - t0

    print(f"Inference + Soft-TTA + Soft-Overlap completed in {elapsed:.2f} seconds.")
    print(f"Detected {len(masks)} clean filament instances.")
    for idx, (m, c) in enumerate(zip(masks, confs)):
        area = m.sum()
        rle = encode_mask_rle(m)
        print(f"  Instance {idx+1}: Conf={c:.3f}, Area={area} px, RLE length={len(rle)}")

    # Verify zero overlaps
    n = len(masks)
    total_shared = 0
    for i in range(n):
        for j in range(i + 1, n):
            shared = int(np.logical_and(masks[i], masks[j]).sum())
            total_shared += shared
    print(f"Total pairwise shared pixels: {total_shared} (Must be exactly 0)")
    assert total_shared == 0, "FATAL: Shared pixels detected!"
    print("ALL TESTS PASSED SUCCESSFULLY!")
