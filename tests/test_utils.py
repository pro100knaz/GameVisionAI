"""Tests for utility modules."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gamevision.utils.config import load_config
from gamevision.utils.logger import get_logger
from gamevision.utils.timer import Timer


class TestTimer:
    def test_measures_time(self):
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.ms > 5  # at least 5ms
        assert t.elapsed > 0.005

    def test_repr_after(self):
        with Timer("test") as t:
            pass
        assert "ms" in repr(t)

    def test_repr_before(self):
        t = Timer("test")
        assert "running" in repr(t)


class TestConfig:
    def test_load_yaml(self, tmp_path: Path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model: yolov8n.pt\nconfidence: 0.5\n")
        cfg = load_config(cfg_file)
        assert cfg["model"] == "yolov8n.pt"
        assert cfg["confidence"] == 0.5

    def test_load_json(self, tmp_path: Path):
        cfg_file = tmp_path / "test.json"
        cfg_file.write_text(json.dumps({"model": "yolov8s.pt"}))
        cfg = load_config(cfg_file)
        assert cfg["model"] == "yolov8s.pt"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestLogger:
    def test_returns_logger(self):
        log = get_logger("test")
        assert log.name == "test"
