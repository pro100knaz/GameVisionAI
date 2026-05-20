"""MSS-based screen capture backend (cross-platform)."""

from __future__ import annotations

import numpy as np
from mss import mss

from gamevision.capture.base import CaptureBackend


class MSSBackend(CaptureBackend):
    """Screen capture using the `mss` library.

    Fast, cross-platform (Windows/Linux/macOS).
    """

    def __init__(self) -> None:
        self._sct = mss()

    def grab_full(self, monitor: int = 0) -> np.ndarray:
        mon = self._sct.monitors[monitor + 1]  # mss uses 1-indexed for real monitors
        shot = self._sct.grab(mon)
        return np.array(shot)[:, :, :3]  # BGRA -> BGR

    def grab_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        region = {"left": x, "top": y, "width": w, "height": h}
        shot = self._sct.grab(region)
        return np.array(shot)[:, :, :3]  # BGRA -> BGR

    def get_screen_size(self, monitor: int = 0) -> tuple[int, int]:
        mon = self._sct.monitors[monitor + 1]
        return mon["width"], mon["height"]
