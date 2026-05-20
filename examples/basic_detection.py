"""Basic detection example — simplest possible usage of GameVisionAI."""

from gamevision.capture import ScreenCapture
from gamevision.detection import YOLODetector
from gamevision.utils.timer import Timer

# 1. Create capture and detector
cap = ScreenCapture(backend="mss")
detector = YOLODetector(model="yolov8n.pt", classes=["person"], confidence=0.3)
detector.warmup()

# 2. Capture center of screen
frame = cap.grab_center(width=640, height=640)
print(f"Captured frame: {frame.shape}")

# 3. Run detection
with Timer("detect") as t:
    detections = detector.detect(frame)

print(f"Found {len(detections)} objects in {t.ms:.1f}ms:")
for det in detections:
    cx, cy = det.bbox.center
    print(f"  {det.class_name} (conf={det.confidence:.2f}) at ({cx:.0f}, {cy:.0f})")
