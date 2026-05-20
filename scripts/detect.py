#!/usr/bin/env python3
"""Run detection on screen capture.

Usage:
    python scripts/detect.py                          # detect persons in center 640x640
    python scripts/detect.py --width 800 --height 600 # custom capture size
    python scripts/detect.py --classes person car      # filter classes
    python scripts/detect.py --model yolov8s.pt        # different model
    python scripts/detect.py --save                    # save annotated screenshot
"""

from __future__ import annotations

import argparse
import sys

import cv2

from gamevision.capture import ScreenCapture
from gamevision.detection import YOLODetector
from gamevision.utils.logger import get_logger
from gamevision.utils.timer import Timer

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen detection with YOLO")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
    p.add_argument("--classes", nargs="+", default=["person"], help="Class filter")
    p.add_argument("--confidence", type=float, default=0.25, help="Min confidence")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=640, help="Capture height")
    p.add_argument("--device", default=None, help="Inference device (cpu/cuda)")
    p.add_argument("--save", action="store_true", help="Save annotated screenshot")
    p.add_argument("--loops", type=int, default=1, help="Number of detection loops")
    return p.parse_args()


def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    for det in detections:
        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def main() -> None:
    args = parse_args()

    # Initialize
    cap = ScreenCapture(backend="mss")
    detector = YOLODetector(
        model=args.model,
        classes=args.classes,
        confidence=args.confidence,
        device=args.device,
    )
    detector.warmup()

    screen_w, screen_h = cap.get_screen_size()
    log.info("Screen: %dx%d, Capture: %dx%d", screen_w, screen_h, args.width, args.height)

    for i in range(args.loops):
        # Capture
        with Timer("capture") as t_cap:
            frame = cap.grab_center(width=args.width, height=args.height)
        log.info("Capture: %.1fms", t_cap.ms)

        # Detect
        with Timer("detect") as t_det:
            detections = detector.detect(frame)
        log.info("Detection: %.1fms (%d objects)", t_det.ms, len(detections))

        # Results
        for det in detections:
            log.info("  %s", det)

        if not detections:
            log.info("  No detections")

        # Save
        if args.save and i == args.loops - 1:
            annotated = draw_detections(frame.copy(), detections)
            cv2.imwrite("detection_result.png", annotated)
            log.info("Saved: detection_result.png")

    log.info("Total pipeline: capture=%.1fms + detect=%.1fms = %.1fms", t_cap.ms, t_det.ms, t_cap.ms + t_det.ms)


if __name__ == "__main__":
    main()
