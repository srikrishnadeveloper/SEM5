"""
Unit and Contract Verification for Grok Directive R2 (v8_1 Inference Sweep).
"""

import sys
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib
_infer_module = importlib.import_module("v8_1.3_infer_cascade")
arbitrate_instances = _infer_module.arbitrate_instances
get_solar_disk_mask = _infer_module.get_solar_disk_mask
from metrics.pq import pq_score, encode_mask, decode_rle


class TestR2InferenceSweep(unittest.TestCase):
    def setUp(self):
        self.disk_mask = get_solar_disk_mask(2048, 2048, r_frac=0.93)

    def test_overlap_mode_trim(self):
        """Verify trim mode carves overlapping pixels from lower-confidence instances."""
        # Instance A (higher conf): 100x100 box at (500, 500)
        mask_a = np.zeros((2048, 2048), dtype=np.uint8)
        mask_a[500:600, 500:600] = 1

        # Instance B (lower conf): 100x100 box at (550, 550) - 50% overlap with A
        mask_b = np.zeros((2048, 2048), dtype=np.uint8)
        mask_b[550:650, 550:650] = 1

        candidates = [
            {"confidence": 0.90, "mask": mask_a},
            {"confidence": 0.50, "mask": mask_b},
        ]

        accepted = arbitrate_instances(candidates, disk_mask=self.disk_mask, min_area=100, overlap_mode="trim")
        self.assertEqual(len(accepted), 2)
        # In trim mode, accepted B should NOT have any pixels that are in A
        overlap = np.logical_and(accepted[0]["mask"], accepted[1]["mask"]).sum()
        self.assertEqual(overlap, 0, "Trim mode must have zero overlapping pixels!")
        # Remaining area of B should be 10000 - 2500 = 7500
        self.assertEqual(accepted[1]["mask"].sum(), 7500)

    def test_overlap_mode_allow(self):
        """Verify allow mode preserves uncarved masks when IoU <= 0.5."""
        # Instance A: (500:600, 500:600) -> area 10,000
        mask_a = np.zeros((2048, 2048), dtype=np.uint8)
        mask_a[500:600, 500:600] = 1

        # Instance B: (580:680, 580:680) -> overlap is 20x20 = 400. IoU = 400 / 19600 = 0.02 <= 0.5
        mask_b = np.zeros((2048, 2048), dtype=np.uint8)
        mask_b[580:680, 580:680] = 1

        candidates = [
            {"confidence": 0.90, "mask": mask_a},
            {"confidence": 0.50, "mask": mask_b},
        ]

        accepted = arbitrate_instances(candidates, disk_mask=self.disk_mask, min_area=100, overlap_mode="allow")
        self.assertEqual(len(accepted), 2)
        # In allow mode, mask B is uncarved
        self.assertEqual(accepted[1]["mask"].sum(), 10000)

    def test_overlap_mode_allow_suppression(self):
        """Verify allow mode suppresses duplicate detection when IoU > 0.5."""
        # Instance A: (500:600, 500:600) -> area 10,000
        mask_a = np.zeros((2048, 2048), dtype=np.uint8)
        mask_a[500:600, 500:600] = 1

        # Instance B: (510:600, 510:600) -> 90x90 = 8100 overlap. IoU = 8100 / 10000 = 0.81 > 0.5
        mask_b = np.zeros((2048, 2048), dtype=np.uint8)
        mask_b[510:600, 510:600] = 1

        candidates = [
            {"confidence": 0.90, "mask": mask_a},
            {"confidence": 0.50, "mask": mask_b},
        ]

        accepted = arbitrate_instances(candidates, disk_mask=self.disk_mask, min_area=100, overlap_mode="allow")
        self.assertEqual(len(accepted), 1, "Duplicate detection with IoU > 0.5 must be suppressed!")
        self.assertEqual(accepted[0]["confidence"], 0.90)

    def test_min_area_filtering(self):
        """Verify instances below min_area are discarded."""
        mask_small = np.zeros((2048, 2048), dtype=np.uint8)
        mask_small[500:508, 500:508] = 1  # 64 px

        candidates = [{"confidence": 0.80, "mask": mask_small}]
        accepted = arbitrate_instances(candidates, disk_mask=self.disk_mask, min_area=100, overlap_mode="trim")
        self.assertEqual(len(accepted), 0)

        accepted_50 = arbitrate_instances(candidates, disk_mask=self.disk_mask, min_area=50, overlap_mode="trim")
        self.assertEqual(len(accepted_50), 1)

    def test_pq_scorer_contract(self):
        """Verify Kirillov PQ computation and components."""
        m_gt = np.zeros((2048, 2048), dtype=np.uint8)
        m_gt[500:600, 500:600] = 1

        m_pred = np.zeros((2048, 2048), dtype=np.uint8)
        m_pred[500:600, 500:600] = 1

        score = pq_score([m_pred], [m_gt], iou_threshold=0.5)
        self.assertAlmostEqual(score["PQ"], 1.0)
        self.assertAlmostEqual(score["SQ"], 1.0)
        self.assertAlmostEqual(score["RQ"], 1.0)
        self.assertEqual(score["TP"], 1)
        self.assertEqual(score["FP"], 0)
        self.assertEqual(score["FN"], 0)


if __name__ == "__main__":
    unittest.main()
