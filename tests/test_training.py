"""Tests for the training pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gamevision.training.collector import CollectorConfig, DataCollector
from gamevision.training.dataset import DatasetManager, SplitResult, augment_image


# --- DataCollector ---


class TestCollectorConfig:
    def test_defaults(self):
        cfg = CollectorConfig()
        assert cfg.mode == "classification"
        assert "1" in cfg.class_names
        assert cfg.capture_width == 640

    def test_custom(self):
        cfg = CollectorConfig(
            output_dir="test_out",
            class_names={"a": "player", "b": "npc"},
            mode="detection",
        )
        assert cfg.mode == "detection"
        assert len(cfg.class_names) == 2


class TestDataCollector:
    def test_setup_classification_dirs(self, tmp_path: Path):
        cfg = CollectorConfig(
            output_dir=str(tmp_path / "data"),
            class_names={"1": "player", "2": "no_player"},
        )
        collector = DataCollector(cfg)
        assert (tmp_path / "data" / "player").is_dir()
        assert (tmp_path / "data" / "no_player").is_dir()

    def test_setup_detection_dirs(self, tmp_path: Path):
        cfg = CollectorConfig(
            output_dir=str(tmp_path / "data"),
            mode="detection",
        )
        collector = DataCollector(cfg)
        assert (tmp_path / "data" / "images").is_dir()
        assert (tmp_path / "data" / "labels").is_dir()

    def test_save_classification(self, tmp_path: Path):
        cfg = CollectorConfig(output_dir=str(tmp_path / "data"))
        collector = DataCollector(cfg)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        path = collector.save_classification(frame, "positive")
        assert path.exists()
        assert path.parent.name == "positive"

    def test_save_detection(self, tmp_path: Path):
        cfg = CollectorConfig(output_dir=str(tmp_path / "data"), mode="detection")
        collector = DataCollector(cfg)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        path = collector.save_detection(frame, "0 0.5 0.5 0.2 0.3")
        assert path.exists()
        label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
        assert label_path.exists()
        assert "0 0.5 0.5 0.2 0.3" in label_path.read_text()

    def test_get_stats(self, tmp_path: Path):
        cfg = CollectorConfig(output_dir=str(tmp_path / "data"))
        collector = DataCollector(cfg)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        collector.save_classification(frame, "positive")
        collector.save_classification(frame, "positive")
        collector.save_classification(frame, "negative")
        stats = collector.get_stats()
        assert stats["positive"] == 2
        assert stats["negative"] == 1


# --- DatasetManager ---


def _create_classification_dataset(root: Path, classes: dict[str, int]) -> Path:
    """Helper: create a fake classification dataset."""
    for class_name, count in classes.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img_{i:03d}.png"), img)
    return root


def _create_detection_dataset(root: Path, count: int) -> Path:
    """Helper: create a fake detection dataset."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"img_{i:03d}.png"), img)
        (root / "labels" / f"img_{i:03d}.txt").write_text(f"0 0.5 0.5 0.2 0.3\n")
    return root


class TestDatasetManager:
    def test_get_classes_classification(self, tmp_path: Path):
        root = _create_classification_dataset(tmp_path, {"cat": 5, "dog": 3})
        dm = DatasetManager(root, mode="classification")
        assert dm.get_classes() == ["cat", "dog"]

    def test_get_classes_detection(self, tmp_path: Path):
        root = _create_detection_dataset(tmp_path, 3)
        dm = DatasetManager(root, mode="detection")
        assert "0" in dm.get_classes()

    def test_count_classification(self, tmp_path: Path):
        root = _create_classification_dataset(tmp_path, {"a": 10, "b": 5})
        dm = DatasetManager(root, mode="classification")
        counts = dm.count()
        assert counts["a"] == 10
        assert counts["b"] == 5

    def test_count_detection(self, tmp_path: Path):
        root = _create_detection_dataset(tmp_path, 7)
        dm = DatasetManager(root, mode="detection")
        counts = dm.count()
        assert counts["images"] == 7
        assert counts["labels"] == 7

    def test_split_classification(self, tmp_path: Path):
        root = _create_classification_dataset(tmp_path / "raw", {"player": 20, "npc": 10})
        dm = DatasetManager(root, mode="classification")
        result = dm.split(output_dir=tmp_path / "split", val_ratio=0.2, seed=42)
        assert isinstance(result, SplitResult)
        assert result.train_count + result.val_count == 30
        assert result.val_count >= 2  # at least some go to val
        assert (tmp_path / "split" / "train" / "player").is_dir()
        assert (tmp_path / "split" / "val" / "player").is_dir()

    def test_split_detection(self, tmp_path: Path):
        root = _create_detection_dataset(tmp_path / "raw", 10)
        dm = DatasetManager(root, mode="detection")
        result = dm.split(output_dir=tmp_path / "split", val_ratio=0.3, seed=42)
        assert result.train_count + result.val_count == 10
        # Check that labels were copied too
        train_labels = list((tmp_path / "split" / "train" / "labels").glob("*.txt"))
        assert len(train_labels) == result.train_count

    def test_validate_ok(self, tmp_path: Path):
        root = _create_classification_dataset(tmp_path, {"a": 15, "b": 15})
        dm = DatasetManager(root, mode="classification")
        issues = dm.validate()
        assert issues == []

    def test_validate_missing_labels(self, tmp_path: Path):
        root = _create_detection_dataset(tmp_path, 5)
        # Remove one label
        labels = list((root / "labels").glob("*.txt"))
        labels[0].unlink()
        dm = DatasetManager(root, mode="detection")
        issues = dm.validate()
        assert any("without labels" in i for i in issues)

    def test_validate_empty_class(self, tmp_path: Path):
        root = tmp_path / "ds"
        (root / "empty_class").mkdir(parents=True)
        _create_classification_dataset(root, {"good": 15})
        dm = DatasetManager(root, mode="classification")
        issues = dm.validate()
        assert any("no images" in i for i in issues)

    def test_generate_yolo_yaml(self, tmp_path: Path):
        root = _create_detection_dataset(tmp_path / "ds", 5)
        dm = DatasetManager(root, mode="detection")
        yaml_path = dm.generate_yolo_yaml(
            tmp_path / "dataset.yaml",
            class_names=["player", "npc"],
        )
        assert yaml_path.exists()
        text = yaml_path.read_text()
        assert "player" in text
        assert "npc" in text

    def test_not_found(self):
        with pytest.raises(FileNotFoundError):
            DatasetManager("/nonexistent/path")

    def test_get_image_paths(self, tmp_path: Path):
        root = _create_classification_dataset(tmp_path, {"a": 3, "b": 2})
        dm = DatasetManager(root, mode="classification")
        paths = dm.get_image_paths()
        assert len(paths) == 5


# --- Augmentation ---


class TestAugmentation:
    def test_augment_returns_multiple(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        results = augment_image(img, seed=42)
        assert len(results) >= 4  # original + flipped + brightness + noise
        assert all(r.shape == img.shape for r in results)

    def test_augment_no_flip(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        results = augment_image(img, flip_h=False, flip_v=False, seed=42)
        assert len(results) == 3  # original + brightness + noise

    def test_augment_deterministic(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        r1 = augment_image(img, seed=123)
        r2 = augment_image(img, seed=123)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_augment_clipping(self):
        # White image — brightness increase should still be clipped to 255
        img = np.full((50, 50, 3), 250, dtype=np.uint8)
        results = augment_image(img, brightness_range=(1.5, 1.5), seed=42)
        assert results[0].max() <= 255
