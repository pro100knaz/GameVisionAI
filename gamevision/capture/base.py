"""Abstract base class for screen capture backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CaptureBackend(ABC):
    """Abstract screen capture backend.

    All backends must return BGR numpy arrays (OpenCV convention).
    """

    @abstractmethod
    def grab_full(self, monitor: int = 0) -> np.ndarray:
        """Capture the full screen.

        Args:
            monitor: Monitor index (0 = primary).

        Returns:
            BGR image as numpy array (H, W, 3).
        """

    @abstractmethod
    def grab_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Capture a specific screen region.

        Args:
            x: Left coordinate.
            y: Top coordinate.
            w: Width in pixels.
            h: Height in pixels.

        Returns:
            BGR image as numpy array (H, W, 3).
        """

    @abstractmethod
    def get_screen_size(self, monitor: int = 0) -> tuple[int, int]:
        """Get screen resolution.

        Args:
            monitor: Monitor index.

        Returns:
            (width, height) tuple.
        """

    def grab_center(self, width: int = 300, height: int = 300, monitor: int = 0) -> np.ndarray:
        """Capture a region centered on the screen.

        Args:
            width: Capture width.
            height: Capture height.
            monitor: Monitor index.

        Returns:
            BGR image as numpy array (H, W, 3).
        """
        sw, sh = self.get_screen_size(monitor)
        x = (sw - width) // 2
        y = (sh - height) // 2
        return self.grab_region(x, y, width, height)
