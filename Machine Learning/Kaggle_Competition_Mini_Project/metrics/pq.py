"""
Kirillov Panoptic Quality (PQ) scorer for solar filament instance segmentation.

PQ = SQ × RQ
SQ = mean IoU of matched instances (Segmentation Quality)
RQ = TP / (TP + 0.5·FP + 0.5·FN) (Recognition Quality)

Matching: greedy unique pairing with IoU > 0.5 threshold.
Reference: Kirillov et al., "Panoptic Segmentation", CVPR 2019.
"""

import numpy as np
import pycocotools.mask as mask_utils


def encode_mask(mask_hw: np.ndarray) -> str:
    """Encode a 2D binary mask (H, W) to a COCO RLE counts string.

    Always goes through pycocotools Fortran 3D encode.
    Never returns a hardcoded string.

    Args:
        mask_hw: uint8 array of shape (H, W) with values 0 or 1.

    Returns:
        COCO RLE counts string (e.g. for a 2048×2048 zero mask this
        will be whatever pycocotools produces — currently 'PPPP4').
    """
    h, w = mask_hw.shape[:2]
    mask_3d = np.asfortranarray(mask_hw.astype(np.uint8)).reshape((h, w, 1))
    rle = mask_utils.encode(mask_3d)[0]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return counts


def decode_rle(counts_str: str, h: int | tuple = 2048, w: int = 2048) -> np.ndarray:
    """Decode a COCO RLE counts string back to a 2D binary mask (H, W)."""
    if isinstance(h, (tuple, list)):
        h, w = h[0], h[1]
    rle = {"size": [int(h), int(w)], "counts": counts_str}
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def pq_score(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute Kirillov Panoptic Quality between predicted and GT instances.

    Matching is greedy: sort all (pred, gt) pairs by descending IoU,
    accept a pair only if IoU > iou_threshold and neither instance is
    already matched. Each instance can match at most once.

    Args:
        pred_masks: list of binary (H, W) uint8 arrays, one per predicted instance.
        gt_masks:   list of binary (H, W) uint8 arrays, one per GT instance.
        iou_threshold: minimum IoU for a valid match (default 0.5).

    Returns:
        dict with keys: PQ, SQ, RQ, TP, FP, FN, matched_ious.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    # Edge cases
    if n_pred == 0 and n_gt == 0:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0, "TP": 0, "FP": 0, "FN": 0, "matched_ious": []}
    if n_pred == 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": 0, "FN": n_gt, "matched_ious": []}
    if n_gt == 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": n_pred, "FN": 0, "matched_ious": []}

    # Compute full IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_matrix[i, j] = _iou(pm, gm)

    # Greedy matching: sort all pairs by descending IoU
    pairs = []
    for i in range(n_pred):
        for j in range(n_gt):
            if iou_matrix[i, j] > iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    for iou_val, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_ious.append(iou_val)

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    sq = float(np.mean(matched_ious)) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    return {
        "PQ": round(pq, 6),
        "SQ": round(sq, 6),
        "RQ": round(rq, 6),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "matched_ious": matched_ious,
    }


def pq_score_multi(
    pred_masks_list: list[list[np.ndarray]],
    gt_masks_list: list[list[np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute mean PQ / SQ / RQ across multiple images.

    Args:
        pred_masks_list: list of per-image predicted mask lists.
        gt_masks_list:   list of per-image GT mask lists.
        iou_threshold: minimum IoU for matching.

    Returns:
        dict with mean PQ, SQ, RQ, total TP, FP, FN, and per-image results.
    """
    assert len(pred_masks_list) == len(gt_masks_list), "Mismatched image count"

    per_image = []
    total_tp = total_fp = total_fn = 0
    sum_pq = sum_sq = sum_rq = 0.0

    for preds, gts in zip(pred_masks_list, gt_masks_list):
        result = pq_score(preds, gts, iou_threshold)
        per_image.append(result)
        total_tp += result["TP"]
        total_fp += result["FP"]
        total_fn += result["FN"]
        sum_pq += result["PQ"]
        sum_sq += result["SQ"]
        sum_rq += result["RQ"]

    n = len(pred_masks_list)
    return {
        "mean_PQ": round(sum_pq / n, 6) if n > 0 else 0.0,
        "mean_SQ": round(sum_sq / n, 6) if n > 0 else 0.0,
        "mean_RQ": round(sum_rq / n, 6) if n > 0 else 0.0,
        "total_TP": total_tp,
        "total_FP": total_fp,
        "total_FN": total_fn,
        "n_images": n,
        "per_image": per_image,
    }
