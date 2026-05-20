"""Tests for detection base classes."""

from __future__ import annotations

import numpy as np
import pytest

from gamevision.detection.base import BBox, Detection, Detector


class TestBBox:
    def test_dimensions(self):
        box = BBox(10, 20, 110, 70)
        assert box.width == 100
        assert box.height == 50

    def test_center(self):
        box = BBox(0, 0, 100, 100)
        assert box.center == (50, 50)

    def test_area(self):
        box = BBox(0, 0, 10, 20)
        assert box.area == 200

    def test_to_xywh(self):
        box = BBox(10, 20, 50, 80)
        assert box.to_xywh() == (10, 20, 40, 60)

    def test_immutable(self):
        box = BBox(0, 0, 100, 100)
        with pytest.raises(AttributeError):
            box.x1 = 50


class TestDetection:
    def test_repr(self):
        det = Detection(
            bbox=BBox(0, 0, 100, 100),
            class_name="person",
            confidence=0.95,
        )
        s = repr(det)
        assert "person" in s
        assert "0.95" in s

    def test_metadata_default(self):
        det = Detection(bbox=BBox(0, 0, 1, 1), class_name="x", confidence=0.5)
        assert det.metadata == {}


class FakeDetector(Detector):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [
            Detection(
                bbox=BBox(0, 0, 50, 50),
                class_name="person",
                confidence=0.9,
            )
        ]

    def warmup(self) -> None:
        pass


class TestDetector:
    def test_fake_detector(self):
        det = FakeDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].class_name == "person"

    def test_detector_is_abstract(self):
        with pytest.raises(TypeError):
            Detector()
