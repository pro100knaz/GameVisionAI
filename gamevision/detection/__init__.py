"""Detection models and pipeline."""

from gamevision.detection.base import BBox, Detection, Detector

__all__ = ["BBox", "Detection", "Detector", "YOLODetector"]


def __getattr__(name: str):
    if name == "YOLODetector":
        from gamevision.detection.yolo import YOLODetector

        return YOLODetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
