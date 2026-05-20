"""Model export: PyTorch -> ONNX -> TensorRT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gamevision.utils.logger import get_logger

log = get_logger(__name__)

SUPPORTED_FORMATS = ("onnx", "torchscript", "engine", "openvino", "coreml")


@dataclass
class ExportConfig:
    """Export configuration.

    Attributes:
        model_path: Path to trained YOLO weights (.pt).
        format: Export format ('onnx', 'torchscript', 'engine', 'openvino', 'coreml').
        imgsz: Input image size.
        half: Use FP16 (half precision).
        dynamic: Dynamic input shapes (ONNX).
        simplify: Simplify ONNX graph.
        opset: ONNX opset version.
        device: Export device.
    """

    model_path: str = "best.pt"
    format: str = "onnx"
    imgsz: int = 640
    half: bool = False
    dynamic: bool = False
    simplify: bool = True
    opset: int = 17
    device: str | None = None


class Exporter:
    """Export trained YOLO models to deployment formats.

    Args:
        config: Export configuration.

    Example:
        >>> exporter = Exporter(ExportConfig(model_path="runs/train/exp/weights/best.pt"))
        >>> onnx_path = exporter.export()
        >>> exporter.validate(onnx_path)
    """

    def __init__(self, config: ExportConfig | None = None) -> None:
        self.config = config or ExportConfig()

    def export(self) -> Path:
        """Export the model.

        Returns:
            Path to the exported model file.
        """
        from ultralytics import YOLO

        cfg = self.config
        if cfg.format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {cfg.format!r}. Supported: {SUPPORTED_FORMATS}"
            )

        log.info("Exporting %s -> %s (imgsz=%d, half=%s)", cfg.model_path, cfg.format, cfg.imgsz, cfg.half)

        model = YOLO(cfg.model_path)
        export_path = model.export(
            format=cfg.format,
            imgsz=cfg.imgsz,
            half=cfg.half,
            dynamic=cfg.dynamic,
            simplify=cfg.simplify,
            opset=cfg.opset,
            device=cfg.device,
        )

        result = Path(export_path)
        log.info("Export complete: %s (%.1f MB)", result, result.stat().st_size / 1024 / 1024)
        return result

    def validate(self, exported_path: str | Path) -> dict[str, float]:
        """Validate an exported model by running inference on a dummy input.

        Args:
            exported_path: Path to the exported model.

        Returns:
            Dict with validation info (inference time, output shape, etc.).
        """
        import numpy as np
        from ultralytics import YOLO

        path = Path(exported_path)
        if not path.exists():
            raise FileNotFoundError(f"Exported model not found: {path}")

        log.info("Validating exported model: %s", path)

        model = YOLO(str(path))
        dummy = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)

        from gamevision.utils.timer import Timer

        with Timer("validation_inference") as t:
            results = model.predict(dummy, verbose=False)

        info = {
            "inference_ms": t.ms,
            "file_size_mb": path.stat().st_size / 1024 / 1024,
        }

        log.info("Validation: inference=%.1fms, size=%.1fMB", info["inference_ms"], info["file_size_mb"])
        return info
