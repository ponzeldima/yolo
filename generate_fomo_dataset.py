from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

# === Constants ===
SOURCE_DATASET_DIR = Path("dataset_test_244")
OUTPUT_ROOT = Path("datasets/fomo")
IMAGE_SIZE = 96
GRID_SIZE = 12
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def parse_data_yaml(dataset_dir: Path) -> list[str]:
    """Read class names from YOLO data.yaml if available."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return []

    names: list[str] = []
    nc: int | None = None

    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("nc:"):
            try:
                nc = int(line.split(":", 1)[1].strip())
            except ValueError:
                nc = None

        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    names = [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                names = []

    if not names and nc is not None and nc > 0:
        names = [f"class_{idx}" for idx in range(nc)]

    return names


def find_images(images_dir: Path) -> list[Path]:
    image_files: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(pattern))
    return sorted(image_files)


def parse_yolo_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return YOLO rows as (cls, cx, cy, w, h), normalized in [0, 1]."""
    if not label_path.exists():
        return []

    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            cls_idx = int(parts[0])
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue

        rows.append((cls_idx, cx, cy, bw, bh))
    return rows


def to_fomo_heatmap(
    yolo_rows: Iterable[tuple[int, float, float, float, float]],
    num_source_classes: int,
    grid_size: int,
) -> np.ndarray:
    """
    Create FOMO target map with shape [grid, grid, classes+1].
    Channel 0 is background, channels 1..N are object classes.
    """
    heatmap = np.zeros((grid_size, grid_size, num_source_classes + 1), dtype=np.float32)

    for cls_idx, cx, cy, _, _ in yolo_rows:
        if cls_idx < 0 or cls_idx >= num_source_classes:
            continue

        cell_x = int(np.clip(cx * grid_size, 0, grid_size - 1))
        cell_y = int(np.clip(cy * grid_size, 0, grid_size - 1))
        heatmap[cell_y, cell_x, cls_idx + 1] = 1.0

    object_presence = np.sum(heatmap[:, :, 1:], axis=-1)
    heatmap[:, :, 0] = (object_presence == 0).astype(np.float32)
    return heatmap


def save_resized_image(src_path: Path, dst_path: Path, image_size: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        rgb = img.convert("RGB")
        resized = rgb.resize((image_size, image_size), Image.BILINEAR)
        resized.save(dst_path)


def convert_split(
    dataset_dir: Path,
    split_name: str,
    out_dir: Path,
    class_names: list[str],
    image_size: int,
    grid_size: int,
) -> dict[str, int]:
    split_input_dir = dataset_dir / split_name
    images_dir = split_input_dir / "images"
    labels_dir = split_input_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        return {"images": 0, "labels": 0}

    split_output_images = out_dir / split_name / "images"
    split_output_labels = out_dir / split_name / "labels"
    split_output_images.mkdir(parents=True, exist_ok=True)
    split_output_labels.mkdir(parents=True, exist_ok=True)

    image_files = find_images(images_dir)
    if not image_files:
        return {"images": 0, "labels": 0}

    num_source_classes = len(class_names)
    if num_source_classes == 0:
        raise ValueError("Could not detect classes from data.yaml. Set valid names/nc first.")

    labels_written = 0

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        yolo_rows = parse_yolo_label_file(label_path)
        heatmap = to_fomo_heatmap(yolo_rows, num_source_classes, grid_size)

        output_image_path = split_output_images / f"{image_path.stem}.jpg"
        output_label_path = split_output_labels / f"{image_path.stem}.npy"

        save_resized_image(image_path, output_image_path, image_size)
        np.save(output_label_path, heatmap)
        labels_written += 1

    return {"images": len(image_files), "labels": labels_written}


def resolve_splits(dataset_dir: Path) -> list[str]:
    preferred = ["train", "valid", "val", "test"]
    return [name for name in preferred if (dataset_dir / name).exists()]


def main() -> None:
    dataset_dir = SOURCE_DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset folder not found: {dataset_dir}")

    source_name = dataset_dir.name
    output_dir = OUTPUT_ROOT / f"{source_name}-fomo"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_data_yaml(dataset_dir)
    if not class_names:
        raise ValueError("Could not read class names from data.yaml")

    found_splits = resolve_splits(dataset_dir)
    if not found_splits:
        raise ValueError("No train/valid/val/test split folders found in source dataset")

    split_stats: dict[str, dict[str, int]] = {}
    for split_name in found_splits:
        normalized_name = "valid" if split_name == "val" else split_name
        stats = convert_split(
            dataset_dir=dataset_dir,
            split_name=split_name,
            out_dir=output_dir,
            class_names=class_names,
            image_size=IMAGE_SIZE,
            grid_size=GRID_SIZE,
        )
        split_stats[normalized_name] = stats

    fomo_meta = {
        "source_dataset": str(dataset_dir),
        "output_dataset": str(output_dir),
        "image_size": IMAGE_SIZE,
        "grid_size": GRID_SIZE,
        "num_classes_total": len(class_names) + 1,
        "class_names": ["background", *class_names],
        "splits": split_stats,
        "label_format": {
            "type": "npy",
            "shape": [GRID_SIZE, GRID_SIZE, len(class_names) + 1],
            "channels": "0=background, 1..N=classes",
        },
    }

    meta_path = output_dir / "fomo_dataset.json"
    meta_path.write_text(json.dumps(fomo_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("FOMO dataset generated")
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    for split_name, stats in split_stats.items():
        print(f"{split_name}: images={stats['images']}, labels={stats['labels']}")


if __name__ == "__main__":
    main()
