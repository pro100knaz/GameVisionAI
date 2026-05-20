"""Fine-tuning engine for YOLO models on game screenshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gamevision.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        base_model: Pretrained model path or name (e.g., 'yolov8n.pt').
        dataset: Path to dataset YAML (YOLO format) or dataset directory.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size (-1 for auto).
        lr0: Initial learning rate.
        project: Output project directory.
        name: Run name within project.
        device: Training device ('cpu', 'cuda', '0', etc.).
        resume: Resume from last checkpoint.
        patience: Early stopping patience (0 = disabled).
        extra: Additional kwargs passed to YOLO train().
    """

    base_model: str = "yolov8n.pt"
    dataset: str = "dataset.yaml"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    project: str = "runs/train"
    name: str = "exp"
    device: str | None = None
    resume: bool = False
    patience: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Fine-tune YOLO models on custom game datasets.

    Uses ultralytics YOLO transfer learning — starts from a pretrained model
    and fine-tunes on your game-specific data.

    Args:
        config: Training configuration.

    Example:
        >>> trainer = Trainer(TrainConfig(
        ...     base_model="yolov8n.pt",
        ...     dataset="datasets/split/dataset.yaml",
        ...     epochs=50,
        ... ))
        >>> results = trainer.train()
        >>> print(f"mAP50: {trainer.get_metrics()['mAP50']}")
    """

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()
        self._model = None
        self._results = None

    def train(self) -> Any:
        """Run the training loop.

        Returns:
            Ultralytics training results object.
        """
        from ultralytics import YOLO

        cfg = self.config
        log.info(
            "Starting training: model=%s, dataset=%s, epochs=%d, imgsz=%d",
            cfg.base_model,
            cfg.dataset,
            cfg.epochs,
            cfg.imgsz,
        )

        self._model = YOLO(cfg.base_model)
        self._results = self._model.train(
            data=cfg.dataset,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            lr0=cfg.lr0,
            project=cfg.project,
            name=cfg.name,
            device=cfg.device,
            resume=cfg.resume,
            patience=cfg.patience,
            **cfg.extra,
        )

        log.info("Training complete. Results in: %s/%s", cfg.project, cfg.name)
        return self._results

    def get_metrics(self) -> dict[str, float]:
        """Get training metrics after training completes.

        Returns:
            Dict with keys like 'mAP50', 'mAP50-95', 'precision', 'recall'.
        """
        if self._results is None:
            raise RuntimeError("No training results. Call train() first.")

        metrics = self._results.results_dict
        return {
            "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            "precision": metrics.get("metrics/precision(B)", 0.0),
            "recall": metrics.get("metrics/recall(B)", 0.0),
        }

    def get_best_weights(self) -> Path:
        """Get the path to the best model weights.

        Returns:
            Path to best.pt.
        """
        best = Path(self.config.project) / self.config.name / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"Best weights not found: {best}")
        return best

    @property
    def model(self):
        """Access the underlying YOLO model (available after train())."""
        return self._model
