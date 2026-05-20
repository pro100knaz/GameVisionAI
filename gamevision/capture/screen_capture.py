"""High-level screen capture facade."""

from __future__ import annotations

from typing import Literal

import numpy as np

from gamevision.capture.base import CaptureBackend


def _create_backend(backend: str) -> CaptureBackend:
    """Factory for capture backends."""
    if backend == "mss":
        from gamevision.capture.mss_backend import MSSBackend

        return MSSBackend()
    raise ValueError(f"Unknown capture backend: {backend!r}. Available: 'mss'")


class ScreenCapture:
    """High-level screen capture API.

    Args:
        backend: Capture backend name. Currently supports 'mss'.

    Example:
        >>> cap = ScreenCapture(backend="mss")
        >>> frame = cap.grab_center(width=300, height=300)
        >>> frame.shape
        (300, 300, 3)
    """

    def __init__(self, backend: Literal["mss"] = "mss") -> None:
        self._backend = _create_backend(backend)

    @property
    def backend(self) -> CaptureBackend:
        """Access the underlying capture backend."""
        return self._backend

    def grab_full(self, monitor: int = 0) -> np.ndarray:
        """Capture the full screen."""
        return self._backend.grab_full(monitor)

    def grab_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Capture a specific screen region."""
        return self._backend.grab_region(x, y, w, h)

    def grab_center(self, width: int = 300, height: int = 300, monitor: int = 0) -> np.ndarray:
        """Capture a region centered on the screen."""
        return self._backend.grab_center(width, height, monitor)

    def get_screen_size(self, monitor: int = 0) -> tuple[int, int]:
        """Get screen resolution."""
        return self._backend.get_screen_size(monitor)
