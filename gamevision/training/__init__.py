"""Training pipeline: data collection, dataset management, fine-tuning, export."""

from gamevision.training.collector import CollectorConfig, DataCollector
from gamevision.training.dataset import DatasetManager, SplitResult, augment_image

__all__ = [
    "CollectorConfig",
    "DataCollector",
    "DatasetManager",
    "SplitResult",
    "augment_image",
    "ExportConfig",
    "Exporter",
    "TrainConfig",
    "Trainer",
]


def __getattr__(name: str):
    if name in ("TrainConfig", "Trainer"):
        from gamevision.training import trainer

        return getattr(trainer, name)
    if name in ("ExportConfig", "Exporter"):
        from gamevision.training import exporter

        return getattr(exporter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
