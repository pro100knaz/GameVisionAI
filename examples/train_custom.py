"""Example: Full training pipeline for a custom game.

Steps:
1. Collect screenshots with hotkey labeling
2. Split into train/val sets
3. Generate YOLO dataset config
4. Fine-tune YOLO model
5. Export to ONNX
"""

from gamevision.training import (
    CollectorConfig,
    DataCollector,
    DatasetManager,
)

# --- Step 1: Collect data ---
# Run this interactively, press 1 for player, 2 for no_player, q to quit
config = CollectorConfig(
    output_dir="datasets/my_game",
    mode="classification",
    class_names={"1": "player", "2": "no_player"},
    capture_width=640,
    capture_height=640,
)
collector = DataCollector(config)
print("Press 1=player, 2=no_player, q=quit")
# collector.start()  # uncomment to run

# --- Step 2: Split dataset ---
dm = DatasetManager("datasets/my_game", mode="classification")
issues = dm.validate()
if issues:
    print(f"Dataset issues: {issues}")
else:
    result = dm.split(output_dir="datasets/my_game_split", val_ratio=0.2)
    print(f"Split: {result.train_count} train, {result.val_count} val")

# --- Step 3: Train (requires ultralytics + torch) ---
# from gamevision.training import Trainer, TrainConfig
# trainer = Trainer(TrainConfig(
#     base_model="yolov8n.pt",
#     dataset="datasets/my_game_split/dataset.yaml",
#     epochs=50,
# ))
# trainer.train()

# --- Step 4: Export ---
# from gamevision.training import Exporter, ExportConfig
# exporter = Exporter(ExportConfig(
#     model_path="runs/train/exp/weights/best.pt",
#     format="onnx",
# ))
# exporter.export()
