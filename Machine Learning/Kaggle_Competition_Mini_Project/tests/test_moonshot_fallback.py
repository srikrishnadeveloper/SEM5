"""
tests/test_moonshot_fallback.py — Test Resolution/Batch Fallback Chain, Manifest, and NMS Parity.

Authority: ChatGPT Master
Directives: 17 & 18
"""

import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import torch

from moonshot_2048.config import MoonshotConfig
from moonshot_2048.train_yolov8l import train_yolov8l_with_fallback, run_holdout_validation_sweep


def test_fallback_sequence_all_transitions():
    """Verify that OOM errors step through:
    (2048, 2) -> (2048, 1) -> (1792, 1) -> (1536, 1)."""
    attempted_specs = []

    def mock_train(*args, **kwargs):
        imgsz = kwargs.get("imgsz")
        batch = kwargs.get("batch")
        attempted_specs.append((imgsz, batch))
        if len(attempted_specs) < 4:
            # Simulate CUDA OOM error on first 3 attempts
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test")
        # 4th attempt succeeds

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_run_dir = Path(tmpdir) / "runs"
        yaml_path = Path(tmpdir) / "data.yaml"
        yaml_path.write_text("dummy: data")

        # Fake weights file to satisfy existence check
        fake_weights = tmp_run_dir / "moonshot_v8l_1536_b1" / "weights" / "best.pt"
        fake_weights.parent.mkdir(parents=True, exist_ok=True)
        fake_weights.write_bytes(b"mock_weights_bytes_12345")

        with patch("torch.cuda.is_available", return_value=True), \
             patch("ultralytics.YOLO") as MockYOLO, \
             patch.object(MoonshotConfig, "RUNS_DIR", tmp_run_dir):
            mock_instance = MockYOLO.return_value
            mock_instance.train = mock_train

            best_ckpt, manifest = train_yolov8l_with_fallback(
                data_yaml=yaml_path,
                epochs=1,
                device=0,
                dry_run=False,
            )

            # Assert all 4 approved fallback transitions were attempted in exact order
            expected_specs = [(2048, 2), (2048, 1), (1792, 1), (1536, 1)]
            assert attempted_specs == expected_specs, f"Unexpected fallback attempts: {attempted_specs}"
            assert manifest["status"] == "SUCCESS"
            assert manifest["attempt_spec"] == "1536/batch1"
            assert manifest["trained_imgsz"] == 1536
            assert manifest["trained_batch"] == 1
            assert Path(manifest["checkpoint_path"]).exists()
            assert manifest["checkpoint_sha256"] is not None


def test_non_oom_exception_propagates_immediately():
    """Verify that a non-OOM exception (e.g. ValueError or FileNotFoundError) is NOT caught as OOM."""
    call_count = 0

    def mock_train(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid dataset format or corrupted YAML!")

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "data.yaml"
        yaml_path.write_text("dummy: data")

        with patch("torch.cuda.is_available", return_value=True), \
             patch("ultralytics.YOLO") as MockYOLO:
            mock_instance = MockYOLO.return_value
            mock_instance.train = mock_train

            with pytest.raises(ValueError, match="Invalid dataset format"):
                train_yolov8l_with_fallback(
                    data_yaml=yaml_path,
                    epochs=1,
                    dry_run=False,
                )

            # Assert execution stopped immediately without attempting any fallback
            assert call_count == 1


def test_checkpoint_identity_and_manifest_written():
    """Verify that experiment_manifest.json is saved to the run directory and matches best.pt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_run_dir = Path(tmpdir) / "runs"
        yaml_path = Path(tmpdir) / "data.yaml"
        yaml_path.write_text("dummy: data")

        fake_weights = tmp_run_dir / "moonshot_v8l_2048_b2" / "weights" / "best.pt"
        fake_weights.parent.mkdir(parents=True, exist_ok=True)
        fake_weights.write_bytes(b"first_attempt_success_weights")

        with patch("torch.cuda.is_available", return_value=True), \
             patch("ultralytics.YOLO") as MockYOLO, \
             patch.object(MoonshotConfig, "RUNS_DIR", tmp_run_dir):
            mock_instance = MockYOLO.return_value
            mock_instance.train = MagicMock()

            best_ckpt, manifest = train_yolov8l_with_fallback(
                data_yaml=yaml_path,
                epochs=60,
                dry_run=False,
            )

            manifest_file = tmp_run_dir / "moonshot_v8l_2048_b2" / "experiment_manifest.json"
            assert manifest_file.exists()
            with open(manifest_file, "r") as mf:
                saved_manifest = json.load(mf)

            assert saved_manifest["checkpoint_path"] == manifest["checkpoint_path"]
            assert saved_manifest["checkpoint_sha256"] == manifest["checkpoint_sha256"]
            assert saved_manifest["trained_imgsz"] == 2048
            assert saved_manifest["trained_batch"] == 2


def test_deployment_nms_parity():
    """Verify that validation uses deployment-matched NMS (iou=0.00)."""
    assert MoonshotConfig.NMS_IOU == 0.00
    mock_model = MagicMock()
    mock_model.predict.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock minimal MAGFiLO structure
        magfilo_dir = Path(tmpdir) / "MAGFiLO"
        train_dir = magfilo_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "train_images").mkdir(parents=True, exist_ok=True)
        coco_json = train_dir / "annotations.json"
        coco_json.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))

        yolo_data_dir = Path(tmpdir) / "yolo"
        (yolo_data_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Run sweep
        results = run_holdout_validation_sweep(
            model=mock_model,
            magfilo_dir=magfilo_dir,
            yolo_data_dir=yolo_data_dir,
            conf_list=[0.30],
            min_area_list=[200],
            nms_iou=0.00,
        )
        assert len(results) == 1
        assert results[0]["nms_iou"] == 0.00


if __name__ == "__main__":
    test_fallback_sequence_all_transitions()
    test_non_oom_exception_propagates_immediately()
    test_checkpoint_identity_and_manifest_written()
    test_deployment_nms_parity()
    print("test_moonshot_fallback passed successfully.")
