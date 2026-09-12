"""Scan source tree and notebooks for stale patterns and report exact occurrences."""
from pathlib import Path
import json

PATTERNS = [
    "batch 4, accum 2",
    "Master: Grok",
    "frozen by Grok",
    "Letterboxed input size",
    "pycocotools.__version__",
    "BATCH_SIZE = 16",
    "$data_dir",
    "w / 1024",
    "h / 1024",
    "* scale_x",
    "* scale_y",
    "unique_stems == 180",
    "category_id == 4",
    "if raw is None: continue",
]

TARGET_FILES = [
    Path("v8_1/config.py"),
    Path("v8_1/geometry.py"),
    Path("v8_1/1_train_yolo.py"),
    Path("v8_1/2_train_crop_refiner.py"),
    Path("v8_1/3_infer_cascade.py"),
    Path("scripts/convert_coco_to_yolo.py"),
    Path("metrics/pq.py"),
    Path("notebooks/build_v8_1_kaggle_nb.py"),
    Path("notebooks/V8_1_Kaggle_Production.ipynb"),
    Path("C:/Users/srik2/Desktop/Filament_Colab_Run/V8_1_Kaggle_Production.ipynb"),
    Path("C:/Users/srik2/Desktop/V8_1_Kaggle_Production.ipynb"),
]

print("=" * 80)
print("STATIC STALE-PATTERN SCAN REPORT")
print("=" * 80)

total_matches = 0
for pat in PATTERNS:
    print(f"\nPattern: {repr(pat)}")
    matches = 0
    for tf in TARGET_FILES:
        if not tf.exists():
            continue
        try:
            content = tf.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            if pat in line:
                matches += 1
                total_matches += 1
                print(f"  [{tf.as_posix()}:{idx}] {line.strip()[:100]}")
    if matches == 0:
        print("  --> NO MATCHES FOUND (Clean)")

print("\n" + "=" * 80)
print(f"TOTAL STALE MATCHES: {total_matches}")
print("=" * 80)
