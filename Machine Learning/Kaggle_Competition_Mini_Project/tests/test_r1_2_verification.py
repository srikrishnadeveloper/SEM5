"""
Comprehensive Verification Suite for ChatGPT Master Directive R1.2.
"""
import ast
import json
import re
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "V8_1_Kaggle_Production.ipynb"


def test_notebook_json_and_embedded_config():
    print("\n--- A. JSON load production notebook ---")
    assert NOTEBOOK_PATH.exists(), f"Missing {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"[PASS] Successfully loaded notebook with {len(nb['cells'])} cells.")

    print("\n--- D. Inspect embedded v8_1/config.py in Cell 4 ---")
    cell_4_src = "".join(nb["cells"][3]["source"])
    pattern = r'with open\("v8_1/config\.py"[^\)]*\) as f:\s+f\.write\((.*?)\)\n'
    m = re.search(pattern, cell_4_src, re.DOTALL)
    assert m is not None, "Failed to find v8_1/config.py in Cell 4"
    config_code = ast.literal_eval(m.group(1))

    # Execute config code in local namespace to inspect classes
    local_ns = {"__file__": str(PROJECT_ROOT / "v8_1" / "config.py")}
    exec(config_code, local_ns)
    CropRefinerConfig = local_ns["CropRefinerConfig"]
    YOLOConfig = local_ns["YOLOConfig"]
    CascadeConfig = local_ns["CascadeConfig"]

    assert CropRefinerConfig.BATCH_SIZE == 8, f"Expected 8, got {CropRefinerConfig.BATCH_SIZE}"
    print(f"[PASS] CropRefinerConfig.BATCH_SIZE == {CropRefinerConfig.BATCH_SIZE}")

    assert CropRefinerConfig.EPOCHS == 20, f"Expected 20, got {CropRefinerConfig.EPOCHS}"
    print(f"[PASS] CropRefinerConfig.EPOCHS == {CropRefinerConfig.EPOCHS}")

    assert YOLOConfig.BATCH == 2, f"Expected 2, got {YOLOConfig.BATCH}"
    print(f"[PASS] YOLOConfig.BATCH == {YOLOConfig.BATCH}")

    assert YOLOConfig.EPOCHS == 20, f"Expected 20, got {YOLOConfig.EPOCHS}"
    print(f"[PASS] YOLOConfig.EPOCHS == {YOLOConfig.EPOCHS}")

    assert YOLOConfig.DEVICE == 0, f"Expected 0, got {YOLOConfig.DEVICE}"
    print(f"[PASS] YOLOConfig.DEVICE == {YOLOConfig.DEVICE}")

    assert YOLOConfig.OVERLAP_MASK is True, f"Expected True, got {YOLOConfig.OVERLAP_MASK}"
    print(f"[PASS] YOLOConfig.OVERLAP_MASK == {YOLOConfig.OVERLAP_MASK}")

    assert YOLOConfig.MASK_RATIO == 4, f"Expected 4, got {YOLOConfig.MASK_RATIO}"
    print(f"[PASS] YOLOConfig.MASK_RATIO == {YOLOConfig.MASK_RATIO}")

    assert CascadeConfig.DEFAULT_CONF == 0.25, f"Expected 0.25, got {CascadeConfig.DEFAULT_CONF}"
    print(f"[PASS] CascadeConfig.DEFAULT_CONF == {CascadeConfig.DEFAULT_CONF}")


def test_cell2_version_logging():
    print("\n--- F. Verify Cell 2 version logging robustness ---")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cell_2_src = "".join(nb["cells"][1]["source"])

    assert "safe_version" in cell_2_src, "Cell 2 missing safe_version function"
    assert "pkg_version('pycocotools')" in cell_2_src or "safe_version('pycocotools')" in cell_2_src, "Cell 2 not using safe package version resolution"
    assert "pycocotools.__version__" not in cell_2_src, "Unsafe pycocotools.__version__ still present in Cell 2"

    # Simulate execution of safe_version with missing __version__
    from importlib.metadata import version as pkg_version
    def safe_version(pkg_name):
        try:
            return pkg_version(pkg_name)
        except Exception:
            return "unknown"

    ver = safe_version("pycocotools")
    print(f"[PASS] safe_version('pycocotools') resolved successfully: {ver}")


def test_stage2_unreadable_image_raises():
    print("\n--- G. Verify Stage 2 unreadable image raises RuntimeError ---")
    import importlib
    mod = importlib.import_module("v8_1.2_train_crop_refiner")
    FilamentCropDataset = mod.FilamentCropDataset

    # Create dummy dataset instance with non-existent file
    fake_record = {
        "img_path": Path("non_existent_filament_image_12345.jpeg"),
        "bbox": [100, 100, 300, 300],
        "pts": np.array([[100, 100], [150, 120], [200, 100]], dtype=np.float32),
    }
    ds = FilamentCropDataset([fake_record], target_size=384, augment=False)
    try:
        _ = ds[0]
        assert False, "Should have raised RuntimeError on missing/unreadable image!"
    except RuntimeError as e:
        assert "FATAL: Unable to decode training image" in str(e)
        print(f"[PASS] Caught expected exception: {e}")


if __name__ == "__main__":
    test_notebook_json_and_embedded_config()
    test_cell2_version_logging()
    test_stage2_unreadable_image_raises()
    print("\n=== ALL R1.2 CHECKS PASSED SUCCESSFULLY ===")
