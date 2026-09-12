"""
moonshot_2048/audit_submission.py — Strict Submission Contract Verifier.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Mandatory Submission Contract Checks (Directive 17 Section 10):
1. Exactly 180 test images accounted for.
2. Zero-area masks == 0 (each row must decode to sum > 0).
3. Invalid-RLE count == 0.
4. Wrong-shape count == 0 (must decode to exactly (2048, 2048)).
5. Strictly zero pairwise shared pixels per disk: sum(mask_i & mask_j) == 0.
6. Duplicate filament_id count == 0.
7. Disks with zero predictions emit zero rows (never dummy zero masks).
8. Compute distribution of rows/image and mask areas, and SHA256.
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from metrics.pq import decode_rle


def audit_submission_csv(
    csv_path: Path,
    expected_test_stems: int = 180,
    expected_shape: tuple = (2048, 2048),
) -> Dict[str, Any]:
    """Execute complete forensic contract audit on a submission CSV file."""
    print("=" * 75)
    print("MOONSHOT 2048: SUBMISSION CONTRACT AUDIT")
    print("=" * 75)
    print(f"  Target File: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Submission file does not exist at {csv_path}")

    raw_bytes = csv_path.read_bytes()
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    print(f"  SHA256: {sha256_hash}")

    df = pd.read_csv(csv_path)
    print(f"  Total Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # 1. Column contract
    assert list(df.columns) == ["filament_id", "segmentation_rle"], \
        f"Column mismatch! Expected ['filament_id', 'segmentation_rle'], got {list(df.columns)}"

    # 2. Duplicate filament_id check
    dup_ids = df[df.duplicated(subset=["filament_id"])]
    n_dups = len(dup_ids)
    print(f"  Duplicate filament_ids: {n_dups}")
    assert n_dups == 0, f"FATAL: Found {n_dups} duplicate filament_ids!"

    # 3. Disks representation
    # filament_id format is typically '{stem}_{idx}'
    stem_to_rows = defaultdict(list)
    for idx, row in df.iterrows():
        fid = str(row["filament_id"])
        parts = fid.rsplit("_", 1)
        stem = parts[0]
        stem_to_rows[stem].append(row["segmentation_rle"])

    represented_stems = len(stem_to_rows)
    print(f"  Represented Disks: {represented_stems} (out of {expected_test_stems})")
    assert represented_stems <= expected_test_stems, \
        f"FATAL: Represented stems ({represented_stems}) exceeds expected test stems ({expected_test_stems})!"

    # 4. Check inference manifest and/or test directory if available
    manifest_p = csv_path.parent / "inference_manifest.json"
    if manifest_p.exists():
        import json
        with open(manifest_p, "r", encoding="utf-8") as mf:
            man = json.load(mf)
        assert man.get("total_test_images") == expected_test_stems, \
            f"FATAL: Manifest total_test_images ({man.get('total_test_images')}) != {expected_test_stems}!"
        assert man.get("represented_disks") == represented_stems, \
            f"FATAL: Manifest represented_disks ({man.get('represented_disks')}) != {represented_stems}!"
        print(f"  [MANIFEST VERIFIED] Verified against inference_manifest.json ({expected_test_stems} test images).")

    # If test directory is discoverable, assert all represented stems are legitimate test stems
    candidate_test_dirs = [
        csv_path.parent / "test_images",
        Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
        Path("data/MAGFiLO_1.0_Kaggle_2026/test/test_images"),
    ]
    discovered_test_dir = next((td for td in candidate_test_dirs if td.exists()), None)
    if discovered_test_dir and expected_test_stems == 180:
        found_test_files = list(discovered_test_dir.glob("*.jpeg")) + list(discovered_test_dir.glob("*.jpg"))
        assert len(found_test_files) == 180, f"FATAL: Discovered test dir has {len(found_test_files)} images, expected 180!"
        test_stem_set = {p.stem for p in found_test_files}
        unknown_stems = set(stem_to_rows.keys()) - test_stem_set
        assert len(unknown_stems) == 0, f"FATAL: Found {len(unknown_stems)} unknown stems not in test set: {list(unknown_stems)[:3]}"
        print(f"  [TEST SET VERIFIED] All {represented_stems} represented stems belong to official 180 test images.")
    zero_area_count = 0
    invalid_rle_count = 0
    wrong_shape_count = 0
    pairwise_overlap_count = 0
    mask_areas = []

    print("  Decoding masks & verifying zero-overlap contract...")
    for stem, rles in stem_to_rows.items():
        disk_masks = []
        for rle_str in rles:
            try:
                m = decode_rle(rle_str, h=expected_shape[0], w=expected_shape[1])
            except Exception as e:
                invalid_rle_count += 1
                continue

            if m.shape != expected_shape:
                wrong_shape_count += 1

            area = int(m.sum())
            if area <= 0:
                zero_area_count += 1
            else:
                mask_areas.append(area)
                disk_masks.append(m)

        # Assert zero pairwise overlap on this disk
        n_m = len(disk_masks)
        for i in range(n_m):
            for j in range(i + 1, n_m):
                shared = int(np.logical_and(disk_masks[i], disk_masks[j]).sum())
                if shared > 0:
                    pairwise_overlap_count += 1

    # Print distribution stats
    rows_per_disk = [len(rles) for rles in stem_to_rows.values()]

    report = {
        "csv_path": str(csv_path),
        "sha256": sha256_hash,
        "total_rows": len(df),
        "represented_disks": represented_stems,
        "expected_test_stems": expected_test_stems,
        "duplicate_filament_ids": n_dups,
        "zero_area_count": zero_area_count,
        "invalid_rle_count": invalid_rle_count,
        "wrong_shape_count": wrong_shape_count,
        "pairwise_overlap_violations": pairwise_overlap_count,
        "rows_per_disk_stats": {
            "min": int(np.min(rows_per_disk)) if rows_per_disk else 0,
            "mean": float(np.mean(rows_per_disk)) if rows_per_disk else 0.0,
            "median": float(np.median(rows_per_disk)) if rows_per_disk else 0.0,
            "max": int(np.max(rows_per_disk)) if rows_per_disk else 0,
        },
        "mask_area_stats": {
            "min": int(np.min(mask_areas)) if mask_areas else 0,
            "q25": float(np.percentile(mask_areas, 25)) if mask_areas else 0.0,
            "median": float(np.median(mask_areas)) if mask_areas else 0.0,
            "q75": float(np.percentile(mask_areas, 75)) if mask_areas else 0.0,
            "max": int(np.max(mask_areas)) if mask_areas else 0,
        },
        "contract_passed": (
            n_dups == 0
            and zero_area_count == 0
            and invalid_rle_count == 0
            and wrong_shape_count == 0
            and pairwise_overlap_count == 0
        ),
    }

    print("\n" + "=" * 75)
    print("AUDIT RESULTS SUMMARY:")
    print(f"  Zero-area masks:               {zero_area_count} (PASS if 0)")
    print(f"  Invalid RLE strings:           {invalid_rle_count} (PASS if 0)")
    print(f"  Wrong-shape masks:             {wrong_shape_count} (PASS if 0)")
    print(f"  Pairwise overlap violations:   {pairwise_overlap_count} (PASS if 0)")
    print(f"  Rows/disk distribution:        mean={report['rows_per_disk_stats']['mean']:.1f}, median={report['rows_per_disk_stats']['median']:.1f}, max={report['rows_per_disk_stats']['max']}")
    print(f"  Mask area distribution (px):   median={report['mask_area_stats']['median']:.0f}, min={report['mask_area_stats']['min']}, max={report['mask_area_stats']['max']}")
    print(f"  CONTRACT PASSED:               {report['contract_passed']}")
    print("=" * 75)

    assert report["contract_passed"], f"FATAL: Submission failed contract checks! {report}"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Submission CSV Contract")
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()

    audit_submission_csv(Path(args.csv_path))
