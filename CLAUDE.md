# GameVisionAI — Project Instructions

## Purpose
Universal game entity detection library using computer vision + ML.
Detect players, NPCs, mobs, UI elements on screen in real-time for any game.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     GameVisionAI                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐       │
│  │ Capture  │──▶│ Preprocessor │──▶│   Detector    │       │
│  │ Engine   │   │              │   │ (YOLO/Custom) │       │
│  └──────────┘   └──────────────┘   └───────┬───────┘       │
│                                             │                │
│                                    ┌────────▼────────┐      │
│                                    │  Post-Processor │      │
│                                    │ (NMS, Tracking) │      │
│                                    └────────┬────────┘      │
│                                             │                │
│                              ┌──────────────▼──────────┐    │
│                              │  Results / API Output   │    │
│                              └─────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Training Toolkit                                     │   │
│  │  - Data collector (auto-screenshot + hotkey labeling) │   │
│  │  - Dataset manager (train/val split, augmentation)    │   │
│  │  - Fine-tuner (transfer learning on game screenshots) │   │
│  │  - Model exporter (PyTorch → ONNX → TensorRT)        │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Module Overview

### 1. Capture Engine (`gamevision/capture/`)
Screen capture abstraction. Multiple backends:
- `mss` — cross-platform, fast, default
- `dxgi` — Windows DXGI Desktop Duplication (lowest latency)
- `opencv` — fallback via cv2.VideoCapture

API:
```python
from gamevision.capture import ScreenCapture
cap = ScreenCapture(backend="mss")
frame = cap.grab_center(width=300, height=300)  # center crop
frame = cap.grab_region(x=100, y=100, w=400, h=400)  # arbitrary region
frame = cap.grab_full()  # full screen
```

### 2. Detection Pipeline (`gamevision/detection/`)
Pluggable detector interface. Implementations:
- `YOLODetector` — ultralytics YOLOv8/v11 (pretrained + fine-tuned)
- `ClassifierDetector` — MobileNetV2/ResNet18 binary/multi-class
- `TemplateDetector` — OpenCV template matching (no ML, fast)
- `EnsembleDetector` — combine multiple detectors, vote

API:
```python
from gamevision.detection import YOLODetector
det = YOLODetector(model="yolov8n.pt", classes=["player"])
results = det.detect(frame)
# results: [Detection(bbox, class_name, confidence, ...)]
```

### 3. Training Toolkit (`gamevision/training/`)
End-to-end training pipeline:
- `DataCollector` — captures screenshots with hotkey labeling (press 1=player, 2=no_player)
- `DatasetManager` — organize, augment, split train/val
- `Trainer` — fine-tune pretrained models on game data
- `Exporter` — PyTorch → ONNX → TensorRT

API:
```python
from gamevision.training import DataCollector, Trainer
# Collect data
collector = DataCollector(output_dir="datasets/pw_players")
collector.start()  # press hotkeys to label screenshots

# Train
trainer = Trainer(base_model="yolov8n.pt", dataset="datasets/pw_players")
trainer.train(epochs=50)
trainer.export("onnx")
```

### 4. Model Zoo (`models/`)
Pre-configured model configs:
- `person_yolov8n` — COCO pretrained, detects real humans
- `game_player_binary` — fine-tuned binary classifier (player/not)
- Custom models added via training toolkit

### 5. Inference Server (`gamevision/server/`)
FastAPI-based real-time detection server:
- REST: `POST /detect` with image → JSON results
- WebSocket: real-time stream of detections
- Configurable: model, region, confidence threshold

### 6. Overlay (`gamevision/overlay/`)
Debug visualization:
- Draw bounding boxes on screen
- Show confidence scores
- FPS counter
- Transparent overlay window (win32 or pygame)

### 7. SDK / Client Libraries
- Python client (native)
- C# client (for integration with VTxPlayground or game bots)

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.11+ | ML ecosystem, fast prototyping |
| ML Framework | PyTorch + ultralytics | Best YOLO support, transfer learning |
| Inference | ONNX Runtime | Fast, cross-platform, GPU/CPU |
| Screen Capture | mss + dxgi | Low latency |
| Image Processing | OpenCV + Pillow | Standard |
| API Server | FastAPI + uvicorn | Async, fast, auto-docs |
| Overlay | pygame / win32gui | Transparent window |
| Package Manager | uv or pip | Modern Python tooling |

## Project Structure

```
GameVisionAI/
├── CLAUDE.md                 # This file
├── README.md                 # User-facing docs
├── pyproject.toml            # Project metadata + dependencies
├── requirements.txt          # Pinned dependencies
│
├── gamevision/               # Main package
│   ├── __init__.py
│   ├── capture/              # Screen capture backends
│   │   ├── __init__.py
│   │   ├── base.py           # CaptureBackend ABC
│   │   ├── mss_backend.py    # mss implementation
│   │   └── dxgi_backend.py   # DXGI Desktop Duplication (Windows)
│   │
│   ├── detection/            # Detection models
│   │   ├── __init__.py
│   │   ├── base.py           # Detector ABC, Detection dataclass
│   │   ├── yolo.py           # YOLOv8/v11 wrapper
│   │   ├── classifier.py     # Binary/multi-class CNN
│   │   └── template.py       # OpenCV template matching
│   │
│   ├── training/             # Training pipeline
│   │   ├── __init__.py
│   │   ├── collector.py      # Screenshot collector with labeling
│   │   ├── dataset.py        # Dataset management + augmentation
│   │   ├── trainer.py        # Fine-tuning engine
│   │   └── exporter.py       # Model export (ONNX, TensorRT)
│   │
│   ├── tracking/             # Multi-object tracking (future)
│   │   ├── __init__.py
│   │   └── sort.py           # SORT/DeepSORT tracker
│   │
│   ├── server/               # API server
│   │   ├── __init__.py
│   │   ├── app.py            # FastAPI application
│   │   └── ws.py             # WebSocket real-time stream
│   │
│   ├── overlay/              # Debug overlay
│   │   ├── __init__.py
│   │   └── renderer.py       # Transparent overlay window
│   │
│   └── utils/                # Shared utilities
│       ├── __init__.py
│       ├── config.py          # YAML/JSON config loader
│       ├── logger.py          # Structured logging
│       └── timer.py           # Performance timing
│
├── models/                    # Model configs + weights
│   ├── yolov8n_coco.yaml     # Pretrained COCO config
│   └── player_binary.yaml    # Binary classifier config
│
├── datasets/                  # Training data (gitignored)
│   └── .gitkeep
│
├── scripts/                   # CLI scripts
│   ├── collect_data.py        # Run data collector
│   ├── train.py               # Run training
│   ├── detect.py              # Run detection on image/screen
│   └── serve.py               # Start API server
│
├── tests/                     # Tests
│   ├── test_capture.py
│   ├── test_detection.py
│   └── test_training.py
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md
│   ├── QUICKSTART.md
│   └── TRAINING_GUIDE.md
│
└── examples/                  # Usage examples
    ├── basic_detection.py     # Simplest usage
    ├── realtime_screen.py     # Real-time screen detection
    ├── train_custom.py        # Train on your game
    └── api_client.py          # Use the API server
```

## Development Rules

### Code Quality
- Type hints on all public functions
- Docstrings (Google style) on all classes and public methods
- Abstract base classes for pluggable components (Capture, Detector, Tracker)
- dataclasses for data structures (Detection, BBox, TrackingResult)
- Pydantic for API models and config validation

### Architecture Principles
- **Plugin architecture** — new detectors/capture backends added without modifying core
- **Config-driven** — YAML configs for models, regions, thresholds
- **Separation of concerns** — capture, detection, tracking, visualization are independent
- **Async-ready** — capture and detection can run in separate threads/processes
- **GPU optional** — everything works on CPU, GPU accelerates inference

### Git Workflow
- Feature branches + PR (same as VTxPlayground)
- Conventional commits (feat:, fix:, docs:)
- Tests before merge

### Performance Targets
| Metric | Target |
|--------|--------|
| Screen capture (300×300 region) | < 5ms |
| YOLO inference (GPU) | < 10ms |
| YOLO inference (CPU) | < 50ms |
| Binary classifier (CPU) | < 20ms |
| End-to-end pipeline | < 30ms GPU / < 100ms CPU |
| Memory usage | < 500MB |

## Implementation Order

### Phase 1: Foundation (MVP)
1. Project setup (pyproject.toml, deps, structure)
2. Capture engine (mss backend)
3. YOLO detector (pretrained, "person" class)
4. Basic script: capture center → detect → print result
5. README with quickstart

### Phase 2: Training Pipeline
6. Data collector (hotkey screenshot labeling)
7. Dataset manager (split, augment)
8. Fine-tuner (YOLO on game screenshots)
9. Model exporter (ONNX)
10. Training guide doc

### Phase 3: Real-time + API
11. Real-time detection loop (threaded capture + detect)
12. Debug overlay (bounding boxes on screen)
13. FastAPI server
14. WebSocket real-time stream

### Phase 4: Advanced
15. Multi-object tracking (SORT)
16. Binary classifier detector
17. Ensemble detector
18. C# client SDK
19. TensorRT optimization
20. HP bar / name plate OCR

## Key Dependencies

```
ultralytics>=8.0    # YOLOv8
torch>=2.0          # PyTorch
onnxruntime-gpu     # ONNX inference (or onnxruntime for CPU)
opencv-python>=4.8  # Image processing
mss>=9.0            # Screen capture
fastapi>=0.100      # API server
uvicorn>=0.20       # ASGI server
pydantic>=2.0       # Config validation
pillow>=10.0        # Image handling
numpy>=1.24         # Array ops
pyyaml>=6.0         # Config files
```

## Integration with VTxPlayground

GameVisionAI can work alongside VTxPlayground:
- VTxPlayground provides kernel-level memory access to game structures
- GameVisionAI provides visual detection (what's on screen)
- Combined: visual targeting + memory-based action execution

Future: C# SDK in GameVisionAI calls the FastAPI server, integrated into PWEasyOffsetResearch workflow.
