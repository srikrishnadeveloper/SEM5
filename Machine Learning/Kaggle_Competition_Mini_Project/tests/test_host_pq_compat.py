"""
Host PQ Compatibility Test:
Compare evaluation of an image with GT filaments when:
- Case A: 0 prediction rows (correct host contract)
- Case B: 1 dummy zero-mask row (old legacy contract)
Demonstrating that Case B adds an extra FP=1 penalty, inflating the denominator.
"""
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.pq import pq_score


def host_style_pq(pred_masks, gt_masks, iou_thresh=0.5):
    """Direct implementation of host get_pq_score logic."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0 and n_pred == 0:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0, "TP": 0, "FP": 0, "FN": 0}
    if n_gt == 0 and n_pred > 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": n_pred, "FN": 0}
    if n_gt > 0 and n_pred == 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": 0, "FN": n_gt}

    # Compute IoU hit matrix
    hit_matrix = np.zeros((n_gt, n_pred), dtype=bool)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            inter = np.logical_and(g > 0, p > 0).sum()
            union = np.logical_or(g > 0, p > 0).sum()
            iou = inter / union if union > 0 else 0.0
            iou_matrix[i, j] = iou
            if iou > iou_thresh:
                hit_matrix[i, j] = True

    # Host greedy matching:
    matched_gt = set()
    matched_pred = set()
    tp_ious = []

    for i in range(n_gt):
        for j in range(n_pred):
            if hit_matrix[i, j] and i not in matched_gt and j not in matched_pred:
                matched_gt.add(i)
                matched_pred.add(j)
                tp_ious.append(iou_matrix[i, j])

    tp = len(tp_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    sq = np.mean(tp_ious) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    pq = sq * rq
    return {"PQ": pq, "SQ": sq, "RQ": rq, "TP": tp, "FP": fp, "FN": fn}


def test_host_pq_compatibility():
    # 2 GT filaments
    gt1 = np.zeros((2048, 2048), dtype=np.uint8)
    gt1[100:200, 100:200] = 1
    gt2 = np.zeros((2048, 2048), dtype=np.uint8)
    gt2[500:600, 500:600] = 1
    gt_masks = [gt1, gt2]

    # Case A: 0 prediction rows (zero detected filaments)
    preds_case_a = []
    res_a = host_style_pq(preds_case_a, gt_masks)
    assert res_a["TP"] == 0
    assert res_a["FN"] == 2
    assert res_a["FP"] == 0, f"Expected FP=0 for 0 prediction rows, got {res_a['FP']}"
    print(f"[PASS] Case A (0 prediction rows): TP={res_a['TP']}, FN={res_a['FN']}, FP={res_a['FP']} (NO FP penalty)")

    # Case B: 1 dummy all-zero mask
    dummy_zero = np.zeros((2048, 2048), dtype=np.uint8)
    preds_case_b = [dummy_zero]
    res_b = host_style_pq(preds_case_b, gt_masks)
    assert res_b["TP"] == 0
    assert res_b["FN"] == 2
    assert res_b["FP"] == 1, f"Expected FP=1 for dummy zero mask, got {res_b['FP']}"
    print(f"[PASS] Case B (1 dummy zero mask): TP={res_b['TP']}, FN={res_b['FN']}, FP={res_b['FP']} (INCURS FP=1 PENALTY!)")

    # Verify our metrics/pq.py produces the exact same result
    our_res_a = pq_score(preds_case_a, gt_masks)
    assert our_res_a["TP"] == res_a["TP"]
    assert our_res_a["FP"] == res_a["FP"]
    assert our_res_a["FN"] == res_a["FN"]
    assert our_res_a["PQ"] == res_a["PQ"]
    print(f"[PASS] Normal instances matching: Host PQ={res_a['PQ']:.4f}, Our PQ={our_res_a['PQ']:.4f}")


if __name__ == "__main__":
    test_host_pq_compatibility()
    print("\n=== ALL HOST PQ COMPATIBILITY TESTS PASSED ===")
