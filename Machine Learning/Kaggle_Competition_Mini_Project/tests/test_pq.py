"""
Unit tests for Kirillov PQ scorer.

Six scenarios as mandated by Grok:
1. Perfect copy — PQ must be 1.0
2. Shifted copy — PQ < 1.0, IoU drops
3. Merged pair — two GT, one pred covers both → 1 TP + 1 FN (or 0 TP if IoU < 0.5)
4. Fragmented pair — one GT, two pred fragments → 1 FN + 2 FP (or partial matches)
5. Empty vs empty — PQ = 1.0 by convention
6. Empty vs one blob — PQ = 0.0 (1 FN)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from metrics.pq import pq_score, encode_mask, decode_rle


def _make_blob(h=256, w=256, y1=50, y2=150, x1=50, x2=150):
    """Create a rectangular blob mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_perfect_copy():
    """Pred == GT → PQ = 1.0, SQ = 1.0, RQ = 1.0, TP = 1, FP = 0, FN = 0."""
    gt = _make_blob()
    pred = gt.copy()
    result = pq_score([pred], [gt])
    assert result["PQ"] == 1.0, f"Expected PQ=1.0, got {result['PQ']}"
    assert result["SQ"] == 1.0
    assert result["RQ"] == 1.0
    assert result["TP"] == 1
    assert result["FP"] == 0
    assert result["FN"] == 0
    print("[PASS] test_perfect_copy: PQ=1.0")


def test_shifted_copy():
    """Pred shifted by 30px → IoU < 1.0 but > 0.5, PQ between 0 and 1."""
    gt = _make_blob(y1=50, y2=150, x1=50, x2=150)
    pred = _make_blob(y1=80, y2=180, x1=50, x2=150)  # shift down 30px
    result = pq_score([pred], [gt])
    assert 0.0 < result["PQ"] < 1.0, f"Expected 0 < PQ < 1, got {result['PQ']}"
    assert result["TP"] == 1
    assert result["FP"] == 0
    assert result["FN"] == 0
    print(f"[PASS] test_shifted_copy: PQ={result['PQ']:.4f}, SQ={result['SQ']:.4f}")


def test_merged_pair():
    """Two separate GT filaments, one large pred covers both.

    The single pred can match at most one GT (greedy). So best case:
    TP=1, FN=1, FP=0 if IoU with one GT > 0.5.
    Worst case: TP=0, FN=2, FP=1 if IoU with each GT < 0.5.
    """
    gt_a = _make_blob(y1=20, y2=80, x1=50, x2=150)   # top filament
    gt_b = _make_blob(y1=120, y2=180, x1=50, x2=150)  # bottom filament
    # pred merges both into one big mask
    pred_merged = _make_blob(y1=20, y2=180, x1=50, x2=150)

    result = pq_score([pred_merged], [gt_a, gt_b])
    # The merged pred has IoU < 0.5 with each individual GT (each GT is 60×100=6000,
    # pred is 160×100=16000, intersection with each is 6000, union is 16000 → IoU=0.375)
    # So TP=0, FP=1, FN=2
    assert result["FN"] >= 1, f"Expected FN >= 1 from merge, got {result['FN']}"
    assert result["PQ"] < 0.5, f"Merged pair should have low PQ, got {result['PQ']}"
    print(f"[PASS] test_merged_pair: PQ={result['PQ']:.4f}, TP={result['TP']}, FP={result['FP']}, FN={result['FN']}")


def test_fragmented_pair():
    """One GT filament, two pred fragments that split it.

    Each fragment covers half the GT. IoU of each fragment with GT
    is roughly intersection/union = 5000/(10000+5000-5000)=0.5, borderline.
    We make the split uneven to ensure IoU < 0.5 for fragments.
    """
    gt = _make_blob(y1=50, y2=150, x1=50, x2=150)  # 100×100 = 10000px
    # Split into top half and bottom half
    frag_a = _make_blob(y1=50, y2=100, x1=50, x2=150)  # 50×100 = 5000px
    frag_b = _make_blob(y1=100, y2=150, x1=50, x2=150)  # 50×100 = 5000px

    result = pq_score([frag_a, frag_b], [gt])
    # Each fragment IoU with GT = 5000 / (10000 + 5000 - 5000) = 0.5 exactly
    # IoU > 0.5 threshold is strict >, so these don't match
    assert result["FN"] >= 1, f"Expected FN >= 1 from fragmentation, got {result['FN']}"
    assert result["FP"] >= 1, f"Expected FP >= 1 from fragmentation, got {result['FP']}"
    print(f"[PASS] test_fragmented_pair: PQ={result['PQ']:.4f}, TP={result['TP']}, FP={result['FP']}, FN={result['FN']}")


def test_empty_vs_empty():
    """No predictions, no GT → PQ = 1.0 by convention."""
    result = pq_score([], [])
    assert result["PQ"] == 1.0, f"Expected PQ=1.0 for empty-vs-empty, got {result['PQ']}"
    assert result["TP"] == 0
    assert result["FP"] == 0
    assert result["FN"] == 0
    print("[PASS] test_empty_vs_empty: PQ=1.0")


def test_empty_vs_blob():
    """No predictions, one GT blob → PQ = 0.0, FN = 1."""
    gt = _make_blob()
    result = pq_score([], [gt])
    assert result["PQ"] == 0.0, f"Expected PQ=0.0, got {result['PQ']}"
    assert result["FN"] == 1
    assert result["FP"] == 0
    print("[PASS] test_empty_vs_blob: PQ=0.0, FN=1")


def test_blob_vs_empty():
    """One prediction, no GT → PQ = 0.0, FP = 1."""
    pred = _make_blob()
    result = pq_score([pred], [])
    assert result["PQ"] == 0.0, f"Expected PQ=0.0, got {result['PQ']}"
    assert result["FP"] == 1
    assert result["FN"] == 0
    print("[PASS] test_blob_vs_empty: PQ=0.0, FP=1")


if __name__ == "__main__":
    test_perfect_copy()
    test_shifted_copy()
    test_merged_pair()
    test_fragmented_pair()
    test_empty_vs_empty()
    test_empty_vs_blob()
    test_blob_vs_empty()
    print("\n=== ALL 7 PQ TESTS PASSED ===")
