"""Dataset management: splitting, augmentation, validation."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from gamevision.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SplitResult:
    """Result of a train/val split operation."""

    train_count: int
    val_count: int
    output_dir: Path


class DatasetManager:
    """Manages datasets for training: splitting, augmenting, validating.

    Supports two layouts:
    - Classification: root/{class_name}/*.png
    - Detection (YOLO): root/images/*.png + root/labels/*.txt

    Args:
        root_dir: Path to the dataset root directory.
        mode: Dataset mode ('classification' or 'detection').

    Example:
        >>> dm = DatasetManager("datasets/collected", mode="classification")
        >>> dm.split(output_dir="datasets/split", val_ratio=0.2)
        >>> dm.validate()
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: Literal["classification", "detection"] = "classification",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.mode = mode

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

    def get_classes(self) -> list[str]:
        """Get class names from the dataset.

        For classification: subfolder names.
        For detection: unique class IDs from label files.
        """
        if self.mode == "classification":
            return sorted(
                d.name for d in self.root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
            )
        else:
            classes: set[str] = set()
            labels_dir = self.root_dir / "labels"
            if labels_dir.exists():
                for lbl_file in labels_dir.glob("*.txt"):
                    text = lbl_file.read_text(encoding="utf-8").strip()
                    for line in text.splitlines():
                        parts = line.strip().split()
                        if parts:
                            classes.add(parts[0])
            return sorted(classes)

    def get_image_paths(self) -> list[Path]:
        """Get all image paths in the dataset."""
        extensions = {"*.png", "*.jpg", "*.jpeg", "*.bmp"}
        paths: list[Path] = []

        if self.mode == "classification":
            for class_dir in self.root_dir.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith("."):
                    for ext in extensions:
                        paths.extend(class_dir.glob(ext))
        else:
            img_dir = self.root_dir / "images"
            if img_dir.exists():
                for ext in extensions:
                    paths.extend(img_dir.glob(ext))

        return sorted(paths)

    def count(self) -> dict[str, int]:
        """Count images per class."""
        counts: dict[str, int] = {}
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

        if self.mode == "classification":
            for class_dir in self.root_dir.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith("."):
                    total = sum(len(list(class_dir.glob(ext))) for ext in extensions)
                    counts[class_dir.name] = total
        else:
            img_dir = self.root_dir / "images"
            if img_dir.exists():
                total = sum(len(list(img_dir.glob(ext))) for ext in extensions)
                counts["images"] = total
            lbl_dir = self.root_dir / "labels"
            if lbl_dir.exists():
                counts["labels"] = len(list(lbl_dir.glob("*.txt")))

        return counts

    def split(
        self,
        output_dir: str | Path,
        val_ratio: float = 0.2,
        seed: int = 42,
        copy: bool = True,
    ) -> SplitResult:
        """Split dataset into train and val sets.

        Args:
            output_dir: Where to write the split dataset.
            val_ratio: Fraction of data for validation (0.0-1.0).
            seed: Random seed for reproducibility.
            copy: If True, copy files. If False, move them.

        Returns:
            SplitResult with counts.
        """
        output = Path(output_dir)
        rng = random.Random(seed)
        transfer = shutil.copy2 if copy else shutil.move
        train_count = 0
        val_count = 0

        if self.mode == "classification":
            for class_dir in self.root_dir.iterdir():
                if not class_dir.is_dir() or class_dir.name.startswith("."):
                    continue

                images = self._list_images(class_dir)
                rng.shuffle(images)
                split_idx = max(1, int(len(images) * (1 - val_ratio)))

                train_imgs = images[:split_idx]
                val_imgs = images[split_idx:]

                train_dir = output / "train" / class_dir.name
                val_dir = output / "val" / class_dir.name
                train_dir.mkdir(parents=True, exist_ok=True)
                val_dir.mkdir(parents=True, exist_ok=True)

                for img in train_imgs:
                    transfer(str(img), str(train_dir / img.name))
                for img in val_imgs:
                    transfer(str(img), str(val_dir / img.name))

                train_count += len(train_imgs)
                val_count += len(val_imgs)

        else:
            images = self._list_images(self.root_dir / "images")
            rng.shuffle(images)
            split_idx = max(1, int(len(images) * (1 - val_ratio)))

            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]

            for subset_name, subset_imgs in [("train", train_imgs), ("val", val_imgs)]:
                img_out = output / subset_name / "images"
                lbl_out = output / subset_name / "labels"
                img_out.mkdir(parents=True, exist_ok=True)
                lbl_out.mkdir(parents=True, exist_ok=True)

                for img in subset_imgs:
                    transfer(str(img), str(img_out / img.name))
                    lbl_file = self.root_dir / "labels" / f"{img.stem}.txt"
                    if lbl_file.exists():
                        transfer(str(lbl_file), str(lbl_out / lbl_file.name))

            train_count = len(train_imgs)
            val_count = len(val_imgs)

        log.info("Split: %d train, %d val -> %s", train_count, val_count, output)
        return SplitResult(train_count=train_count, val_count=val_count, output_dir=output)

    def validate(self) -> list[str]:
        """Validate dataset integrity. Returns list of issues found."""
        issues: list[str] = []

        if not self.root_dir.exists():
            issues.append(f"Root directory does not exist: {self.root_dir}")
            return issues

        if self.mode == "classification":
            classes = self.get_classes()
            if not classes:
                issues.append("No class subdirectories found")
            for cls in classes:
                count = len(self._list_images(self.root_dir / cls))
                if count == 0:
                    issues.append(f"Class '{cls}' has no images")
                elif count < 10:
                    issues.append(f"Class '{cls}' has only {count} images (recommend >= 10)")
        else:
            img_dir = self.root_dir / "images"
            lbl_dir = self.root_dir / "labels"
            if not img_dir.exists():
                issues.append("Missing 'images' directory")
            if not lbl_dir.exists():
                issues.append("Missing 'labels' directory")

            if img_dir.exists() and lbl_dir.exists():
                images = {p.stem for p in self._list_images(img_dir)}
                labels = {p.stem for p in lbl_dir.glob("*.txt")}
                missing_labels = images - labels
                orphan_labels = labels - images
                if missing_labels:
                    issues.append(f"{len(missing_labels)} images without labels")
                if orphan_labels:
                    issues.append(f"{len(orphan_labels)} orphan label files")

        if issues:
            for issue in issues:
                log.warning("Validation: %s", issue)
        else:
            log.info("Dataset validation passed")

        return issues

    def generate_yolo_yaml(
        self,
        output_path: str | Path,
        class_names: list[str],
        train_path: str = "train/images",
        val_path: str = "val/images",
    ) -> Path:
        """Generate a YOLO dataset YAML config file.

        Args:
            output_path: Where to write the YAML file.
            class_names: List of class names (index = class ID).
            train_path: Relative path to training images.
            val_path: Relative path to validation images.

        Returns:
            Path to the generated YAML file.
        """
        import yaml

        output_path = Path(output_path)
        config = {
            "path": str(self.root_dir.resolve()),
            "train": train_path,
            "val": val_path,
            "names": {i: name for i, name in enumerate(class_names)},
        }
        output_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        log.info("Generated YOLO YAML: %s", output_path)
        return output_path

    @staticmethod
    def _list_images(directory: Path) -> list[Path]:
        """List all image files in a directory."""
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        images: list[Path] = []
        for ext in extensions:
            images.extend(directory.glob(ext))
        return sorted(images)


def augment_image(
    image: np.ndarray,
    flip_h: bool = True,
    flip_v: bool = False,
    brightness_range: tuple[float, float] = (0.8, 1.2),
    noise_sigma: float = 10.0,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Apply basic augmentations to an image.

    Returns the original plus augmented variants.

    Args:
        image: BGR input image.
        flip_h: Include horizontal flip.
        flip_v: Include vertical flip.
        brightness_range: Random brightness multiplier range.
        noise_sigma: Gaussian noise standard deviation.
        seed: Random seed.

    Returns:
        List of augmented images (always includes the original).
    """
    rng = np.random.RandomState(seed)
    results = [image]

    if flip_h:
        results.append(cv2.flip(image, 1))

    if flip_v:
        results.append(cv2.flip(image, 0))

    # Brightness
    factor = rng.uniform(*brightness_range)
    bright = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    results.append(bright)

    # Gaussian noise
    noise = rng.normal(0, noise_sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    results.append(noisy)

    return results
