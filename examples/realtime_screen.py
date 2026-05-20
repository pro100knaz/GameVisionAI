"""Real-time screen detection with FPS counter."""

from __future__ import annotations

import time

from gamevision.capture import ScreenCapture
from gamevision.detection import YOLODetector
from gamevision.utils.logger import get_logger
from gamevision.utils.timer import Timer

log = get_logger(__name__)


def main() -> None:
    cap = ScreenCapture(backend="mss")
    detector = YOLODetector(model="yolov8n.pt", classes=["person"], confidence=0.3)
    detector.warmup()

    log.info("Starting real-time detection (Ctrl+C to stop)")

    frame_count = 0
    fps_start = time.perf_counter()

    try:
        while True:
            with Timer("pipeline") as t:
                frame = cap.grab_center(width=640, height=640)
                detections = detector.detect(frame)

            frame_count += 1
            elapsed = time.perf_counter() - fps_start

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                log.info(
                    "FPS: %.1f | Pipeline: %.1fms | Detections: %d",
                    fps,
                    t.ms,
                    len(detections),
                )
                frame_count = 0
                fps_start = time.perf_counter()

    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()
