import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from cascade.cleanup import fill_mask_holes, prune_mask_specks, sanitize_filament_morphology, sanitize_zero_overlap_with_morphology


def test_sanitize_zero_overlap_with_morphology():
    # Mask 1 (conf=0.9): large region (50:150, 50:150) of area 10,000 with a 10x10 hole
    m1 = np.zeros((300, 300), dtype=np.uint8)
    m1[50:150, 50:150] = 1
    m1[95:105, 95:105] = 0  # 100 px hole

    # Mask 2 (conf=0.6): rectangle crossing m1: (80:120, 20:200)
    # The part overlapping m1 will be carved out.
    # Left part outside m1: (80:120, 20:50) -> 40 * 30 = 1200 px (main body)
    # Right part outside m1: (80:85, 150:156) -> 5 * 6 = 30 px (tiny fragment)
    m2 = np.zeros((300, 300), dtype=np.uint8)
    m2[80:120, 20:50] = 1   # 1200 px
    m2[80:120, 50:150] = 1  # 4000 px overlapping m1
    m2[80:85, 150:156] = 1  # 30 px fragment

    masks = [m1, m2]
    confs = [0.9, 0.6]

    clean_masks, clean_confs = sanitize_zero_overlap_with_morphology(
        masks, confs, min_area=200, min_secondary_area=150, fill_holes=True, max_hole_area=200
    )

    assert len(clean_masks) == 2
    out_m1, out_m2 = clean_masks

    # 1. Strictly 0 shared pixels
    assert (out_m1 & out_m2).sum() == 0

    # 2. Mask 1 hole was filled
    assert out_m1.sum() == 10000
    assert out_m1[100, 100] == 1

    # 3. Mask 2 had its overlapping region carved out AND its 30px fragment pruned
    assert out_m2.sum() == 1200
    assert out_m2[82, 153] == 0  # 30px fragment was eliminated!
    assert out_m2[100, 35] == 1  # Main body preserved!



def test_fill_mask_holes():
    # Create 100x100 square with a 10x10 hole in the center
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    mask[45:55, 45:55] = 0  # Hole of area 100
    
    assert mask.sum() == (60 * 60) - 100
    filled = fill_mask_holes(mask, max_hole_area=200)
    assert filled.sum() == 60 * 60  # Hole filled!
    assert filled[50, 50] == 1



def test_prune_mask_specks():
    # Create main body of area 1000 and two tiny specks of area 30 and 40
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:120, 20:30] = 1  # 100 * 10 = 1000 px
    mask[150:156, 150:155] = 1  # 6 * 5 = 30 px speck
    mask[170:175, 170:178] = 1  # 5 * 8 = 40 px speck
    
    clean, n_pruned = prune_mask_specks(mask, min_secondary_area=150)
    assert n_pruned == 2
    assert clean.sum() == 1000
    assert clean[152, 152] == 0
    assert clean[172, 172] == 0
    assert clean[50, 25] == 1


def test_preserve_substantial_bifurcation():
    # Create main body (1000 px) and substantial secondary component (400 px)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:120, 20:30] = 1  # 1000 px
    mask[130:170, 130:140] = 1  # 400 px (40% of main)
    
    clean, n_pruned = prune_mask_specks(mask, min_secondary_area=150, min_ratio_to_largest=0.20)
    assert n_pruned == 0
    assert clean.sum() == 1400  # Both preserved!


def test_empty_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    clean = sanitize_filament_morphology(mask)
    assert clean.sum() == 0


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    test_fill_mask_holes()
    test_prune_mask_specks()
    test_preserve_substantial_bifurcation()
    test_empty_mask()
    test_sanitize_zero_overlap_with_morphology()
    print("[SUCCESS] All cleanup and zero-overlap unit tests passed successfully!")


