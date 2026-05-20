"""Screenshot collector with hotkey labeling for dataset creation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

from gamevision.capture import ScreenCapture
from gamevision.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for the data collector.

    Attributes:
        output_dir: Root directory for saved screenshots.
        capture_width: Width of the capture region.
        capture_height: Height of the capture region.
        mode: 'classification' saves into class folders, 'detection' saves images + placeholder labels.
        class_names: Mapping of hotkey -> class name (e.g., {"1": "player", "2": "npc"}).
        image_format: Output image format.
        auto_interval: If set, auto-capture every N seconds (no hotkey needed).
    """

    output_dir: str = "datasets/collected"
    capture_width: int = 640
    capture_height: int = 640
    mode: Literal["classification", "detection"] = "classification"
    class_names: dict[str, str] = field(default_factory=lambda: {"1": "positive", "2": "negative"})
    image_format: str = "png"
    auto_interval: float | None = None


class DataCollector:
    """Collects labeled screenshots for training.

    In classification mode, saves images into class-specific folders:
        output_dir/positive/img_001.png
        output_dir/negative/img_002.png

    In detection mode, saves images and empty label files (for manual annotation):
        output_dir/images/img_001.png
        output_dir/labels/img_001.txt

    Args:
        config: Collector configuration.
        capture: ScreenCapture instance. Created automatically if None.

    Example:
        >>> collector = DataCollector(CollectorConfig(class_names={"1": "player", "2": "no_player"}))
        >>> collector.start()  # press 1/2 to label, q to quit
    """

    def __init__(
        self,
        config: CollectorConfig | None = None,
        capture: ScreenCapture | None = None,
    ) -> None:
        self.config = config or CollectorConfig()
        self._capture = capture  # lazy — created on first capture_frame() if None
        self._output_dir = Path(self.config.output_dir)
        self._count = 0
        self._setup_dirs()

    def _setup_dirs(self) -> None:
        """Create output directory structure."""
        if self.config.mode == "classification":
            for class_name in self.config.class_names.values():
                (self._output_dir / class_name).mkdir(parents=True, exist_ok=True)
        else:
            (self._output_dir / "images").mkdir(parents=True, exist_ok=True)
            (self._output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def _gen_filename(self) -> str:
        """Generate a unique filename based on timestamp."""
        ts = int(time.time() * 1000)
        return f"img_{ts}_{self._count:04d}"

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the screen center."""
        if self._capture is None:
            self._capture = ScreenCapture(backend="mss")
        return self._capture.grab_center(
            width=self.config.capture_width,
            height=self.config.capture_height,
        )

    def save_classification(self, frame: np.ndarray, class_name: str) -> Path:
        """Save a frame with a classification label.

        Args:
            frame: BGR image.
            class_name: Class folder name.

        Returns:
            Path to the saved image.
        """
        fname = f"{self._gen_filename()}.{self.config.image_format}"
        path = self._output_dir / class_name / fname
        cv2.imwrite(str(path), frame)
        self._count += 1
        return path

    def save_detection(self, frame: np.ndarray, label: str = "") -> Path:
        """Save a frame for detection (YOLO format).

        Args:
            frame: BGR image.
            label: Optional YOLO-format label string (class cx cy w h per line).

        Returns:
            Path to the saved image.
        """
        fname = self._gen_filename()
        img_path = self._output_dir / "images" / f"{fname}.{self.config.image_format}"
        lbl_path = self._output_dir / "labels" / f"{fname}.txt"

        cv2.imwrite(str(img_path), frame)
        lbl_path.write_text(label, encoding="utf-8")
        self._count += 1
        return img_path

    def start(self) -> None:
        """Start interactive collection loop.

        Press the configured hotkeys to label and save screenshots.
        Press 'q' to quit. Press 's' to skip (capture without saving).

        Requires a display — shows a preview window.
        """
        log.info("Data Collector started (mode=%s)", self.config.mode)
        log.info("Output: %s", self._output_dir.resolve())
        log.info("Hotkeys: %s | q=quit | s=skip", self.config.class_names)

        try:
            while True:
                frame = self.capture_frame()

                # Show preview
                preview = cv2.resize(frame, (400, 400))
                cv2.putText(
                    preview,
                    f"Collected: {self._count} | Keys: {self.config.class_names}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("DataCollector", preview)

                key = cv2.waitKey(100) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("s"):
                    continue

                key_char = chr(key) if 32 <= key < 127 else ""
                if key_char in self.config.class_names:
                    class_name = self.config.class_names[key_char]
                    if self.config.mode == "classification":
                        path = self.save_classification(frame, class_name)
                    else:
                        path = self.save_detection(frame)
                    log.info("Saved #%d: %s -> %s", self._count, class_name, path.name)

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            log.info("Collection done. Total: %d images", self._count)

    def get_stats(self) -> dict[str, int]:
        """Return counts per class in the output directory."""
        stats: dict[str, int] = {}
        if self.config.mode == "classification":
            for class_name in self.config.class_names.values():
                class_dir = self._output_dir / class_name
                if class_dir.exists():
                    stats[class_name] = len(list(class_dir.glob(f"*.{self.config.image_format}")))
        else:
            img_dir = self._output_dir / "images"
            if img_dir.exists():
                stats["images"] = len(list(img_dir.glob(f"*.{self.config.image_format}")))
            lbl_dir = self._output_dir / "labels"
            if lbl_dir.exists():
                stats["labels"] = len(list(lbl_dir.glob("*.txt")))
        return stats
