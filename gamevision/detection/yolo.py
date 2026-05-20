"""YOLO detector using the ultralytics library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from gamevision.detection.base import BBox, Detection, Detector
from gamevision.utils.logger import get_logger

log = get_logger(__name__)


class YOLODetector(Detector):
    """YOLOv8/v11 object detector.

    Args:
        model: Path to YOLO weights (.pt) or model name (e.g., "yolov8n.pt").
        classes: Filter to only these class names. None = all classes.
        confidence: Minimum confidence threshold.
        device: Inference device ("cpu", "cuda", "cuda:0", etc.). None = auto.
        imgsz: Input image size for the model.

    Example:
        >>> det = YOLODetector(model="yolov8n.pt", classes=["person"])
        >>> results = det.detect(frame)
    """

    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        classes: list[str] | None = None,
        confidence: float = 0.25,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        self._model_path = str(model)
        self._filter_classes = classes
        self._confidence = confidence
        self._device = device
        self._imgsz = imgsz

        log.info("Loading YOLO model: %s", self._model_path)
        self._model = YOLO(self._model_path)

        if self._device:
            self._model.to(self._device)

        # Build class name -> id mapping for filtering
        self._class_names: dict[int, str] = self._model.names  # {0: 'person', 1: 'bicycle', ...}
        self._filter_ids: list[int] | None = None
        if self._filter_classes:
            name_to_id = {v: k for k, v in self._class_names.items()}
            self._filter_ids = [name_to_id[c] for c in self._filter_classes if c in name_to_id]
            missing = set(self._filter_classes) - set(name_to_id.keys())
            if missing:
                log.warning("Classes not found in model: %s", missing)

        log.info(
            "YOLO ready (classes=%s, conf=%.2f, device=%s)",
            self._filter_classes or "all",
            self._confidence,
            self._device or "auto",
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self._confidence,
            classes=self._filter_ids,
            imgsz=self._imgsz,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self._class_names.get(cls_id, str(cls_id))

                detections.append(
                    Detection(
                        bbox=BBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        class_name=cls_name,
                        confidence=conf,
                        class_id=cls_id,
                    )
                )

        return detections

    def warmup(self) -> None:
        """Run a dummy inference to warm up the model."""
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self.detect(dummy)
        log.info("YOLO warmup complete")

    @property
    def class_names(self) -> dict[int, str]:
        """Model's class name mapping."""
        return self._class_names
