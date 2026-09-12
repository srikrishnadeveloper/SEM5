"""
tests/test_moonshot_ensemble.py — Test Topology-Safe Ensemble and Zero-Overlap Sanitizer.

Authority: ChatGPT Master
Directive: 17
"""

import numpy as np
import pytest
from moonshot_2048.ensemble_instances import cluster_instances, resolve_cluster_consensus, ensemble_disk_predictions
from moonshot_2048.predict_native import sanitize_instances_zero_overlap


def test_cluster_instances_and_resolve_best_conf():
    """Verify overlapping instances are clustered and highest confidence candidate is picked."""
    m1 = np.zeros((512, 512), dtype=np.uint8)
    m1[100:200, 100:200] = 1

    m2 = np.zeros((512, 512), dtype=np.uint8)
    m2[105:205, 105:205] = 1

    props = [
        {"mask": m1, "conf": 0.40},
        {"mask": m2, "conf": 0.85},
    ]

    clusters = cluster_instances(props, cluster_iou_thresh=0.3)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2

    consensus_m, consensus_c = resolve_cluster_consensus(clusters[0], method="best_conf")
    assert consensus_c == 0.85
    assert np.array_equal(consensus_m, m2)


def test_sanitize_instances_zero_overlap_strictly_carves():
    """Verify that sanitize_instances_zero_overlap enforces strictly zero shared pixels."""
    # Create two heavily overlapping squares
    m1 = np.zeros((2048, 2048), dtype=np.uint8)
    m1[500:700, 500:700] = 1  # area 40000

    m2 = np.zeros((2048, 2048), dtype=np.uint8)
    m2[600:800, 600:800] = 1  # overlaps m1 in [600:700, 600:700] (10000 px)

    clean_masks, clean_confs = sanitize_instances_zero_overlap(
        masks=[m1, m2],
        confidences=[0.90, 0.70],
        min_area=200,
    )

    assert len(clean_masks) == 2
    # m1 (conf=0.90) must be intact
    assert clean_masks[0].sum() == 40000
    # m2 (conf=0.70) must have had its overlap with m1 carved away: 40000 - 10000 = 30000
    assert clean_masks[1].sum() == 30000

    # Strictly assert 0 shared pixels
    shared_pixels = int(np.logical_and(clean_masks[0], clean_masks[1]).sum())
    assert shared_pixels == 0, f"Found {shared_pixels} overlapping pixels after sanitizer!"


def test_sanitize_instances_drops_tiny_carved_fragments():
    """Verify that when carving leaves a mask with area < min_area, it is dropped."""
    m1 = np.zeros((2048, 2048), dtype=np.uint8)
    m1[500:700, 500:700] = 1

    # m2 is almost entirely covered by m1, leaving only 50 px
    m2 = np.zeros((2048, 2048), dtype=np.uint8)
    m2[500:700, 500:700] = 1
    m2[700:705, 700:710] = 1  # 50 extra pixels

    clean_masks, _ = sanitize_instances_zero_overlap(
        masks=[m1, m2],
        confidences=[0.90, 0.70],
        min_area=200,
    )
    # m2 should be dropped because remaining area (50 px) < min_area (200)
    assert len(clean_masks) == 1
    assert clean_masks[0].sum() == 40000


def test_ensemble_disk_predictions_end_to_end():
    """Verify ensemble of two model predictions produces non-overlapping valid instances."""
    m_v8 = np.zeros((2048, 2048), dtype=np.uint8)
    m_v8[800:900, 800:900] = 1

    m_11 = np.zeros((2048, 2048), dtype=np.uint8)
    m_11[800:900, 800:900] = 1

    pred_v8 = [{"mask": m_v8, "conf": 0.75}]
    pred_11 = [{"mask": m_11, "conf": 0.85}]

    final_masks, final_confs = ensemble_disk_predictions(
        model_predictions=[pred_v8, pred_11],
        method="best_conf",
        min_area=200,
    )
    assert len(final_masks) == 1
    assert final_confs[0] == 0.85
    assert final_masks[0].sum() == 10000


if __name__ == "__main__":
    test_cluster_instances_and_resolve_best_conf()
    test_sanitize_instances_zero_overlap_strictly_carves()
    test_sanitize_instances_drops_tiny_carved_fragments()
    test_ensemble_disk_predictions_end_to_end()
    print("test_moonshot_ensemble passed successfully.")
