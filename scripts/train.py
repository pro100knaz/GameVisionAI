#!/usr/bin/env python3
"""Train (fine-tune) a YOLO model on game screenshots.

Usage:
    python scripts/train.py --dataset datasets/split/dataset.yaml
    python scripts/train.py --dataset dataset.yaml --model yolov8s.pt --epochs 100
    python scripts/train.py --dataset dataset.yaml --export onnx
"""

from __future__ import annotations

import argparse

from gamevision.training.trainer import TrainConfig, Trainer
from gamevision.training.exporter import ExportConfig, Exporter
from gamevision.utils.logger import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO on game screenshots")
    p.add_argument("--model", default="yolov8n.pt", help="Base pretrained model")
    p.add_argument("--dataset", required=True, help="Path to dataset YAML")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--device", default=None, help="Device (cpu/cuda/0)")
    p.add_argument("--project", default="runs/train", help="Project output dir")
    p.add_argument("--name", default="exp", help="Run name")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--export", default=None, help="Export format after training (onnx, engine, etc.)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Train
    config = TrainConfig(
        base_model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )

    trainer = Trainer(config)
    trainer.train()

    metrics = trainer.get_metrics()
    log.info("Metrics: %s", metrics)

    # Export if requested
    if args.export:
        best_weights = trainer.get_best_weights()
        export_config = ExportConfig(
            model_path=str(best_weights),
            format=args.export,
            imgsz=args.imgsz,
        )
        exporter = Exporter(export_config)
        exported = exporter.export()
        log.info("Exported to: %s", exported)


if __name__ == "__main__":
    main()
