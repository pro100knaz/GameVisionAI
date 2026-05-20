"""Tests for capture module."""

from __future__ import annotations

import numpy as np
import pytest

from gamevision.capture.base import CaptureBackend


class FakeBackend(CaptureBackend):
    """Fake backend for testing base class logic."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self._w = width
        self._h = height

    def grab_full(self, monitor: int = 0) -> np.ndarray:
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def grab_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_screen_size(self, monitor: int = 0) -> tuple[int, int]:
        return self._w, self._h


class TestCaptureBackend:
    def test_grab_center_default(self):
        backend = FakeBackend(1920, 1080)
        frame = backend.grab_center()
        assert frame.shape == (300, 300, 3)

    def test_grab_center_custom_size(self):
        backend = FakeBackend(1920, 1080)
        frame = backend.grab_center(width=640, height=480)
        assert frame.shape == (480, 640, 3)

    def test_grab_full(self):
        backend = FakeBackend(1920, 1080)
        frame = backend.grab_full()
        assert frame.shape == (1080, 1920, 3)

    def test_grab_region(self):
        backend = FakeBackend()
        frame = backend.grab_region(100, 100, 200, 150)
        assert frame.shape == (150, 200, 3)

    def test_screen_size(self):
        backend = FakeBackend(2560, 1440)
        assert backend.get_screen_size() == (2560, 1440)
