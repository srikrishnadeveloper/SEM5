"""
moonshot_2048/run_ensemble_v8_moonshot.py — Cross-Architecture Instance Ensemble.

Combines:
  1. Moonshot Native 2048 YOLOv8l-seg (0.360 LB)
  2. V8.1 True Instance Cascade (0.350 LB)
Both models were trained by the student.
"""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from metrics.pq import decode_rle, encode_mask, _iou
from moonshot_2048.audit_submission import audit_submission_csv

def run_ensemble(
    moon_csv: Path,
    v8_csv: Path,
    out_csv: Path,
    iou_match_thresh: float = 0.25,
    min_area_standalone: int = 350,
    min_area_final: int = 80,
):
    print("=" * 75)
    print("RUNNING CROSS-ARCHITECTURE ENSEMBLE (MOONSHOT 0.360 + V8.1 0.350)")
    print("=" * 75)
    
    df_moon = pd.read_csv(moon_csv)
    df_v8 = pd.read_csv(v8_csv)
    
    print(f"  Model A (Moonshot Native 2048): {len(df_moon)} rows")
    print(f"  Model B (V8.1 True Cascade):    {len(df_v8)} rows")
    
    moon_by_stem = defaultdict(list)
    for _, r in df_moon.iterrows():
        stem = str(r["filament_id"]).rsplit("_", 1)[0]
        moon_by_stem[stem].append(r["segmentation_rle"])
        
    v8_by_stem = defaultdict(list)
    for _, r in df_v8.iterrows():
        stem = str(r["filament_id"]).rsplit("_", 1)[0]
        v8_by_stem[stem].append(r["segmentation_rle"])
        
    all_stems = sorted(list(set(list(moon_by_stem.keys()) + list(v8_by_stem.keys()))))
    print(f"  Total Unique Disks:             {len(all_stems)}")
    
    out_rows = []
    total_filaments = 0
    reinforce_count = 0
    v8_recovered_count = 0
    zero_pred_disks = 0
    
    for disk_idx, stem in enumerate(all_stems):
        rles_moon = moon_by_stem.get(stem, [])
        rles_v8 = v8_by_stem.get(stem, [])
        
        m_moon = [decode_rle(r) for r in rles_moon]
        m_v8 = [decode_rle(r) for r in rles_v8]
        
        matched_v8_indices = set()
        consensus_candidates = []
        
        # 1. Process Moonshot masks & match against V8.1
        for m in m_moon:
            best_iou = 0.0
            best_v_idx = -1
            for v_idx, v in enumerate(m_v8):
                if v_idx in matched_v8_indices:
                    continue
                iou_val = _iou(m, v)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_v_idx = v_idx
            
            if best_iou >= iou_match_thresh and best_v_idx >= 0:
                # Confirmed by BOTH architectures: union high-confidence coverage
                consensus = (m | m_v8[best_v_idx]).astype(np.uint8)
                matched_v8_indices.add(best_v_idx)
                consensus_candidates.append({"mask": consensus, "priority": 3, "area": int(consensus.sum())})
                reinforce_count += 1
            else:
                # Moonshot alone
                consensus_candidates.append({"mask": m, "priority": 2, "area": int(m.sum())})
                
        # 2. Add high-confidence V8.1 filaments missed by Moonshot
        for v_idx, v in enumerate(m_v8):
            if v_idx not in matched_v8_indices:
                max_ov = max([_iou(v, c["mask"]) for c in consensus_candidates]) if consensus_candidates else 0.0
                if max_ov < 0.10 and v.sum() >= min_area_standalone:
                    consensus_candidates.append({"mask": v, "priority": 1, "area": int(v.sum())})
                    v8_recovered_count += 1
                    
        # 3. Sort candidates: priority desc, then area desc
        consensus_candidates.sort(key=lambda x: (x["priority"], x["area"]), reverse=True)
        
        # 4. Strict Greedy Zero-Overlap Carve
        occupied = np.zeros((2048, 2048), dtype=np.uint8)
        clean_masks = []
        for c in consensus_candidates:
            clean_m = c["mask"] & ~occupied
            if clean_m.sum() >= min_area_final:
                occupied |= clean_m
                clean_masks.append(clean_m)
                
        # 5. Strict Zero-Overlap Assertion
        n_clean = len(clean_masks)
        for i in range(n_clean):
            for j in range(i + 1, n_clean):
                assert int(np.logical_and(clean_masks[i], clean_masks[j]).sum()) == 0, \
                    f"Overlap violation on disk {stem} between {i} and {j}!"
                    
        if not clean_masks:
            zero_pred_disks += 1
            continue
            
        for inst_idx, mask in enumerate(clean_masks):
            rle_str = encode_mask(mask)
            out_rows.append({
                "filament_id": f"{stem}_{inst_idx + 1}",
                "segmentation_rle": rle_str,
            })
            total_filaments += 1
            
        if (disk_idx + 1) % 40 == 0 or (disk_idx + 1) == len(all_stems):
            print(f"  [{disk_idx + 1}/{len(all_stems)}] disks processed | Filaments so far: {total_filaments}")
            
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(out_rows)
        
    print("\n" + "=" * 75)
    print("ENSEMBLE COMPLETE:")
    print(f"  Total Filaments Emitted:     {total_filaments}")
    print(f"  Dual-Model Reinforced Masks: {reinforce_count}")
    print(f"  Recovered Missed Filaments:  {v8_recovered_count}")
    print(f"  Zero-Detection Disks:        {zero_pred_disks}")
    print(f"  Saved To:                    {out_csv}")
    print("=" * 75)
    
    # Run audit
    audit_submission_csv(out_csv, expected_test_stems=180)
    return out_csv

if __name__ == "__main__":
    m_csv = Path("submissions/moonshot_submission.csv")
    v_csv = Path("submissions/v8_1_kaggle_production_submission.csv")
    out_csv = Path("submissions/ensemble_moonshot_v8.csv")
    run_ensemble(m_csv, v_csv, out_csv)
