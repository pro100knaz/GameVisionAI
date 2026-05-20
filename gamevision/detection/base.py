"""Base classes and data structures for detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, w, h) format."""
        return (self.x1, self.y1, self.width, self.height)


@dataclass(frozen=True)
class Detection:
    """A single detection result.

    Attributes:
        bbox: Bounding box.
        class_name: Detected class name.
        confidence: Confidence score [0, 1].
        class_id: Numeric class ID from the model.
        metadata: Extra info from the detector.
    """

    bbox: BBox
    class_name: str
    confidence: float
    class_id: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        cx, cy = self.bbox.center
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"center=({cx:.0f},{cy:.0f}), size={self.bbox.width:.0f}x{self.bbox.height:.0f})"
        )


class Detector(ABC):
    """Abstract detector interface.

    All detectors take a BGR numpy image and return a list of Detections.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the model (run a dummy inference)."""
