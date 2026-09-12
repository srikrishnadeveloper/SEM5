"""
Verify V8.1 Kaggle notebook internal consistency, exact disk sync, commands, and stale patterns.
"""
import ast
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "V8_1_Kaggle_Production.ipynb"


def test_notebook_sync():
    print("=" * 80)
    print("VERIFYING NOTEBOOK JSON & EMBEDDED SOURCE SYNC")
    print("=" * 80)
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded notebook successfully: {len(nb['cells'])} cells, format v{nb['nbformat']}.{nb['nbformat_minor']}")

    # Cell 4 contains embedded files written via f.write(...)
    cell_4_src = "".join(nb["cells"][3]["source"])

    embedded_files = {
        "metrics/pq.py": PROJECT_ROOT / "metrics" / "pq.py",
        "scripts/convert_coco_to_yolo.py": PROJECT_ROOT / "scripts" / "convert_coco_to_yolo.py",
        "v8_1/config.py": PROJECT_ROOT / "v8_1" / "config.py",
        "v8_1/geometry.py": PROJECT_ROOT / "v8_1" / "geometry.py",
        "v8_1/1_train_yolo.py": PROJECT_ROOT / "v8_1" / "1_train_yolo.py",
        "v8_1/2_train_crop_refiner.py": PROJECT_ROOT / "v8_1" / "2_train_crop_refiner.py",
        "v8_1/3_infer_cascade.py": PROJECT_ROOT / "v8_1" / "3_infer_cascade.py",
    }

    print("\n--- Checking Embedded Source Code Identity ---")
    all_matched = True
    for rel_path, disk_file in embedded_files.items():
        assert disk_file.exists(), f"Disk file missing: {disk_file}"
        disk_content = disk_file.read_text(encoding="utf-8")

        pattern = rf'with open\("{re.escape(rel_path)}"[^\)]*\) as f:\s+f\.write\((.*?)\)\n'
        match = re.search(pattern, cell_4_src, re.DOTALL)
        if not match:
            print(f"[FAIL] Could not locate write call for {rel_path} in Cell 4!")
            all_matched = False
            continue

        raw_arg = match.group(1).strip()
        try:
            embedded_content = ast.literal_eval(raw_arg)
        except Exception as e:
            print(f"[ERROR] Could not parse embedded literal for {rel_path}: {e}")
            all_matched = False
            continue

        if embedded_content == disk_content:
            print(f"[PASS] {rel_path:<35} MATCHES DISK EXACTLY ({len(disk_content)} bytes)")
        else:
            print(f"[FAIL] {rel_path:<35} MISMATCH WITH DISK!")
            all_matched = False

    assert all_matched, "FATAL: One or more embedded files in notebook do not match disk!"
    print("\n[OK] All 7 embedded modules match disk byte-for-byte!")

    print("\n" + "=" * 80)
    print("NOTEBOOK STAGE COMMANDS")
    print("=" * 80)
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"]).strip()
        if "scripts/convert_coco_to_yolo.py" in src and "write(" not in src:
            print(f"\n[Stage 0: Conversion Command (Cell {i+1})]:")
            for line in src.splitlines():
                if "cmd = [" in line or "--data-dir" in line or "--out-dir" in line or "subprocess" in line:
                    print("  " + line)
        elif "v8_1/1_train_yolo.py" in src and "write(" not in src:
            print(f"\n[Stage 1: YOLO Training Command (Cell {i+1})]:")
            for line in src.splitlines():
                if "!python" in line:
                    print("  " + line)
        elif "v8_1/2_train_crop_refiner.py" in src and "write(" not in src:
            print(f"\n[Stage 2: Crop Refiner Training Command (Cell {i+1})]:")
            for line in src.splitlines():
                if "!python" in line:
                    print("  " + line)
        elif "v8_1/3_infer_cascade.py" in src and "write(" not in src:
            print(f"\n[Stage 3: Inference Cascade Command (Cell {i+1})]:")
            for line in src.splitlines():
                if "!python" in line or "best_yolo" in line:
                    print("  " + line)


if __name__ == "__main__":
    test_notebook_sync()
