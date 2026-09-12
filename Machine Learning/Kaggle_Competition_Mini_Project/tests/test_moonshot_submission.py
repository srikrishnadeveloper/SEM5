"""
tests/test_moonshot_submission.py — Test Submission Contract Auditor and Edge Cases.

Authority: ChatGPT Master
Directive: 17
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from metrics.pq import encode_mask
from moonshot_2048.audit_submission import audit_submission_csv


def test_audit_submission_passes_valid_csv():
    """Verify that a compliant submission CSV with zero overlap passes all checks."""
    m1 = np.zeros((2048, 2048), dtype=np.uint8)
    m1[100:200, 100:200] = 1

    m2 = np.zeros((2048, 2048), dtype=np.uint8)
    m2[300:400, 300:400] = 1

    rle1 = encode_mask(m1)
    rle2 = encode_mask(m2)

    df = pd.DataFrame([
        {"filament_id": "test_disk_001_0", "segmentation_rle": rle1},
        {"filament_id": "test_disk_001_1", "segmentation_rle": rle2},
        {"filament_id": "test_disk_002_0", "segmentation_rle": rle1},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_p = Path(tmpdir) / "submission.csv"
        df.to_csv(csv_p, index=False)

        report = audit_submission_csv(csv_p, expected_test_stems=2)
        assert report["contract_passed"] is True
        assert report["zero_area_count"] == 0
        assert report["pairwise_overlap_violations"] == 0
        assert report["duplicate_filament_ids"] == 0


def test_audit_submission_fails_on_pairwise_overlap():
    """Verify that audit_submission_csv fails if two masks on the same disk share pixels."""
    m1 = np.zeros((2048, 2048), dtype=np.uint8)
    m1[100:200, 100:200] = 1

    # Overlapping mask
    m2 = np.zeros((2048, 2048), dtype=np.uint8)
    m2[150:250, 150:250] = 1

    rle1 = encode_mask(m1)
    rle2 = encode_mask(m2)

    df = pd.DataFrame([
        {"filament_id": "test_disk_001_0", "segmentation_rle": rle1},
        {"filament_id": "test_disk_001_1", "segmentation_rle": rle2},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_p = Path(tmpdir) / "submission_overlap.csv"
        df.to_csv(csv_p, index=False)

        with pytest.raises(AssertionError, match="Submission failed contract checks"):
            audit_submission_csv(csv_p, expected_test_stems=1)


def test_audit_submission_fails_on_zero_area_mask():
    """Verify that audit_submission_csv fails if any mask decodes to 0 pixels."""
    m_zero = np.zeros((2048, 2048), dtype=np.uint8)
    rle_zero = encode_mask(m_zero)

    df = pd.DataFrame([
        {"filament_id": "test_disk_001_0", "segmentation_rle": rle_zero},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_p = Path(tmpdir) / "submission_zero.csv"
        df.to_csv(csv_p, index=False)

        with pytest.raises(AssertionError, match="Submission failed contract checks"):
            audit_submission_csv(csv_p, expected_test_stems=1)


def test_audit_submission_fails_on_duplicate_filament_ids():
    """Verify that audit_submission_csv fails if duplicate filament_ids exist."""
    m1 = np.zeros((2048, 2048), dtype=np.uint8)
    m1[100:200, 100:200] = 1
    rle1 = encode_mask(m1)

    df = pd.DataFrame([
        {"filament_id": "duplicate_id", "segmentation_rle": rle1},
        {"filament_id": "duplicate_id", "segmentation_rle": rle1},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_p = Path(tmpdir) / "submission_dup.csv"
        df.to_csv(csv_p, index=False)

        with pytest.raises(AssertionError, match="duplicate filament_ids"):
            audit_submission_csv(csv_p, expected_test_stems=1)


if __name__ == "__main__":
    test_audit_submission_passes_valid_csv()
    test_audit_submission_fails_on_pairwise_overlap()
    test_audit_submission_fails_on_zero_area_mask()
    test_audit_submission_fails_on_duplicate_filament_ids()
    print("test_moonshot_submission passed successfully.")
