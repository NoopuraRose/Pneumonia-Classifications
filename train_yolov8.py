"""Prepare the image folders and train a YOLOv8 classification model."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "train"
DATASET_DIR = PROJECT_DIR / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_dataset(
    source_dir: Path = SOURCE_DIR,
    dataset_dir: Path = DATASET_DIR,
    seed: int = 42,
    force: bool = False,
) -> None:
    """Create YOLO classification train/val/test folders from class folders."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {source_dir}")

    classes = sorted(path.name for path in source_dir.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError("Expected at least two class folders inside the train directory")

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        if not force:
            print(f"Using existing prepared dataset: {dataset_dir}")
            return
        shutil.rmtree(dataset_dir)

    rng = random.Random(seed)
    for class_name in classes:
        files = sorted(
            path
            for path in (source_dir / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No supported images found in {source_dir / class_name}")

        rng.shuffle(files)
        train_end = max(1, int(len(files) * 0.8))
        val_end = max(train_end + 1, int(len(files) * 0.9))
        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split_name, split_files in splits.items():
            target_dir = dataset_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_files:
                shutil.copy2(image_path, target_dir / image_path.name)

        print(
            f"{class_name}: {len(splits['train'])} train, "
            f"{len(splits['val'])} val, {len(splits['test'])} test"
        )



def train_model(epochs: int = 3, imgsz: int = 224, batch: int = 16) -> Path:
    """Train and validate YOLOv8n-cls, returning the best checkpoint path."""
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(PROJECT_DIR / "runs" / "classify"),
        name="pneumonia",
        exist_ok=True,
        seed=42,
    )
    model.val(data=str(DATASET_DIR), split="test")
    return PROJECT_DIR / "runs" / "classify" / "pneumonia" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-split", action="store_true", help="Recreate the prepared dataset"
    )
    args = parser.parse_args()

    prepare_dataset(seed=args.seed, force=args.force_split)
    best_model = train_model(args.epochs, args.imgsz, args.batch)
    print(f"Best model saved to: {best_model}")


if __name__ == "__main__":
    main()