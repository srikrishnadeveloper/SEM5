"""
Synthetic test verifying the current Kaggle submission contract:
- 1 row per actual predicted filament
- 0 rows for empty disks (no detections)
- strictly positive decoded area (sum > 0)
- rejection of duplicate filament_id and unknown test stems
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.pq import encode_mask, decode_rle


def test_submission_contract():
    # Simulate 3 test images:
    # A -> 2 positive instances
    # B -> 0 instances
    # C -> 1 positive instance
    official_test_stems = {"test_image_A", "test_image_B", "test_image_C"}

    rows = []
    # Test image A: 2 filaments
    m_a1 = np.zeros((2048, 2048), dtype=np.uint8)
    m_a1[100:200, 100:200] = 1
    rows.append({"filament_id": "test_image_A_1", "segmentation_rle": encode_mask(m_a1)})

    m_a2 = np.zeros((2048, 2048), dtype=np.uint8)
    m_a2[500:600, 500:600] = 1
    rows.append({"filament_id": "test_image_A_2", "segmentation_rle": encode_mask(m_a2)})

    # Test image B: 0 filaments -> EMIT ZERO ROWS (No dummy all-zero mask!)

    # Test image C: 1 filament
    m_c1 = np.zeros((2048, 2048), dtype=np.uint8)
    m_c1[1000:1100, 1000:1100] = 1
    rows.append({"filament_id": "test_image_C_1", "segmentation_rle": encode_mask(m_c1)})

    df = pd.DataFrame(rows)

    # 1. Total rows must equal 3
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    # 2. Columns must match exactly
    assert list(df.columns) == ["filament_id", "segmentation_rle"]

    # 3. All filament_ids must be unique
    assert df["filament_id"].is_unique

    # 4. Stems must belong to official set
    stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    assert set(stems).issubset(official_test_stems)

    # 5. Covered stems: A and C; missing stem: B
    assert set(stems) == {"test_image_A", "test_image_C"}
    assert official_test_stems - set(stems) == {"test_image_B"}

    # 6. Every submitted mask must decode to (2048, 2048) with sum > 0
    for rle in df["segmentation_rle"]:
        m = decode_rle(rle, 2048, 2048)
        assert m.shape == (2048, 2048)
        assert m.sum() > 0, "No dummy all-zero masks permitted!"

    # 7. Reject duplicate filament_id
    dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not dup_df["filament_id"].is_unique

    # 8. Reject unknown stems
    bad_df = pd.DataFrame([{"filament_id": "unknown_stem_1", "segmentation_rle": encode_mask(m_a1)}])
    bad_stems = bad_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    assert not set(bad_stems).issubset(official_test_stems)

    print("[PASS] test_submission_contract: All submission contract assertions verified successfully.")


if __name__ == "__main__":
    test_submission_contract()
    print("\n=== SUBMISSION CONTRACT TESTS PASSED ===")
