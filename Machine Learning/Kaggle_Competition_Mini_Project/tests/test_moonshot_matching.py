"""
tests/test_moonshot_matching.py — Test Kirillov Matching, Multi-Annotator PQ, and Failure Bins.

Authority: ChatGPT Master
Directive: 17
"""

import numpy as np
import pytest
from metrics.pq import pq_score
from moonshot_2048.match_and_calibrate import (
    evaluate_instances_multi_annotator,
    compute_instance_failure_bins,
    extract_candidate_features,
)


def test_kirillov_pq_perfect_match():
    """Verify PQ = 1.0 when prediction and GT match perfectly with IoU = 1.0."""
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[100:200, 100:200] = 1

    res = pq_score([mask], [mask], iou_threshold=0.5)
    assert res["PQ"] == 1.0
    assert res["SQ"] == 1.0
    assert res["RQ"] == 1.0
    assert res["TP"] == 1
    assert res["FP"] == 0
    assert res["FN"] == 0


def test_kirillov_pq_strict_iou_threshold():
    """Verify that a match with IoU <= 0.50 is rejected (counted as 1 FP and 1 FN)."""
    # Create two overlapping squares with IoU < 0.50
    # Overlap = 100 x 40 = 4000; Union = 100x100 + 100x100 - 4000 = 16000; IoU = 4000/16000 = 0.25 <= 0.50
    m_pred = np.zeros((2048, 2048), dtype=np.uint8)
    m_pred[100:200, 100:200] = 1

    m_gt = np.zeros((2048, 2048), dtype=np.uint8)
    m_gt[100:200, 160:260] = 1

    res = pq_score([m_pred], [m_gt], iou_threshold=0.5)
    assert res["TP"] == 0
    assert res["FP"] == 1
    assert res["FN"] == 1
    assert res["PQ"] == 0.0


def test_multi_annotator_pq_evaluation():
    """Verify that multi-annotator evaluation computes pq_mean across annotators."""
    m_pred = np.zeros((2048, 2048), dtype=np.uint8)
    m_pred[100:200, 100:200] = 1

    # Annotator 1 matches perfectly (PQ=1.0)
    ann1_gt = [m_pred.copy()]
    # Annotator 2 has no annotations (FP=1, FN=0 -> PQ=0.0)
    ann2_gt = []

    res = evaluate_instances_multi_annotator([m_pred], [ann1_gt, ann2_gt], iou_threshold=0.5)
    assert res["pq_mean"] == pytest.approx(0.5, abs=1e-4)
    assert res["pq_max"] == 1.0
    assert len(res["ann_scores"]) == 2


def test_instance_failure_bins():
    """Verify that compute_instance_failure_bins assigns instances to area and radial bins."""
    # Small core instance
    m_small_core = np.zeros((2048, 2048), dtype=np.uint8)
    m_small_core[1020:1030, 1020:1030] = 1  # area = 100 px, center near (1024, 1024)

    # Large limb instance
    m_large_limb = np.zeros((2048, 2048), dtype=np.uint8)
    m_large_limb[100:200, 100:200] = 1  # area = 10000 px, far from center

    bins = compute_instance_failure_bins(
        pred_masks=[m_small_core],
        gt_masks=[m_small_core, m_large_limb],
    )
    # Small core matched (TP)
    assert bins["area"]["small"]["tp"] == 1
    assert bins["radial"]["core"]["tp"] == 1
    # Large limb missed (FN)
    assert bins["area"]["large"]["fn"] == 1
    assert bins["radial"]["limb"]["fn"] == 1


def test_extract_candidate_features():
    """Verify feature extractor computes valid morphology features without NaN."""
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[500:600, 500:700] = 1  # 100 x 200 rectangle
    feats = extract_candidate_features(mask, conf=0.85)

    assert feats["conf"] == 0.85
    assert feats["area"] == 20000.0
    assert feats["perimeter"] > 0
    assert 0.0 <= feats["solidity"] <= 1.0
    assert feats["n_components"] == 1
    assert 0.0 <= feats["radial_dist"] <= 2.0


if __name__ == "__main__":
    test_kirillov_pq_perfect_match()
    test_kirillov_pq_strict_iou_threshold()
    test_multi_annotator_pq_evaluation()
    test_instance_failure_bins()
    test_extract_candidate_features()
    print("test_moonshot_matching passed successfully.")
