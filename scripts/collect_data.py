#!/usr/bin/env python3
"""Run the data collector for screenshot labeling.

Usage:
    python scripts/collect_data.py                              # default (player/no_player)
    python scripts/collect_data.py --output datasets/my_game    # custom output dir
    python scripts/collect_data.py --mode detection             # YOLO detection format
    python scripts/collect_data.py --classes 1=player 2=npc 3=mob  # custom class mapping
    python scripts/collect_data.py --width 800 --height 600     # custom capture size
"""

from __future__ import annotations

import argparse

from gamevision.training.collector import CollectorConfig, DataCollector
from gamevision.utils.logger import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect labeled screenshots for training")
    p.add_argument("--output", default="datasets/collected", help="Output directory")
    p.add_argument("--mode", choices=["classification", "detection"], default="classification")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=640, help="Capture height")
    p.add_argument(
        "--classes",
        nargs="+",
        default=["1=player", "2=no_player"],
        help="Class mapping as KEY=NAME pairs",
    )
    return p.parse_args()


def parse_class_mapping(pairs: list[str]) -> dict[str, str]:
    """Parse 'KEY=NAME' pairs into a dict."""
    mapping = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid class mapping: {pair!r}. Use KEY=NAME format.")
        key, name = pair.split("=", 1)
        mapping[key] = name
    return mapping


def main() -> None:
    args = parse_args()
    class_names = parse_class_mapping(args.classes)

    config = CollectorConfig(
        output_dir=args.output,
        capture_width=args.width,
        capture_height=args.height,
        mode=args.mode,
        class_names=class_names,
    )

    collector = DataCollector(config)
    log.info("Class mapping: %s", class_names)
    collector.start()


if __name__ == "__main__":
    main()
