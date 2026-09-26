from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]

# PoC parameters.
MODEL_PATH = ROOT_DIR / "train_models" / "runs" / "detect" / "v8n_320" / "weights" / "best.pt"
DATASET_PATH = ROOT_DIR / "datasets" / "own_1080_IMG_3157"
OUTPUT_DATASET_PATH = ROOT_DIR / "datasets" / "own_1080_IMG_3157_crops"
CAMERA_FOV_DEGREES = 54
ERROR_DEGREES = 8
MODE = "oracle"  # "oracle" or "random"
RANDOM_SEED = 42
IMAGE_SIZE = 320

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []

    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        class_id, center_x, center_y, width, height = values
        labels.append(
            (int(class_id), float(center_x), float(center_y), float(width), float(height))
        )
    return labels


def crop_labels(
    labels: list[tuple[int, float, float, float, float]],
    crop_left: int,
    crop_top: int,
    crop_size: int,
    image_width: int,
    image_height: int,
) -> list[str]:
    transformed = []
    for class_id, center_x, center_y, width, height in labels:
        box_left = (center_x - width / 2) * image_width
        box_top = (center_y - height / 2) * image_height
        box_right = (center_x + width / 2) * image_width
        box_bottom = (center_y + height / 2) * image_height

        box_left = max(box_left, crop_left)
        box_top = max(box_top, crop_top)
        box_right = min(box_right, crop_left + crop_size)
        box_bottom = min(box_bottom, crop_top + crop_size)
        if box_right <= box_left or box_bottom <= box_top:
            continue

        new_center_x = ((box_left + box_right) / 2 - crop_left) / crop_size
        new_center_y = ((box_top + box_bottom) / 2 - crop_top) / crop_size
        new_width = (box_right - box_left) / crop_size
        new_height = (box_bottom - box_top) / crop_size
        transformed.append(
            f"{class_id} {new_center_x:.6f} {new_center_y:.6f} "
            f"{new_width:.6f} {new_height:.6f}"
        )
    return transformed


def create_cropped_dataset() -> tuple[int, int]:
    source_images = DATASET_PATH / "valid" / "images"
    source_labels = DATASET_PATH / "valid" / "labels"
    output_images = OUTPUT_DATASET_PATH / "valid" / "images"
    output_labels = OUTPUT_DATASET_PATH / "valid" / "labels"

    if MODE not in {"oracle", "random"}:
        raise ValueError("MODE must be 'oracle' or 'random'")
    if CAMERA_FOV_DEGREES <= 0 or ERROR_DEGREES <= 0:
        raise ValueError("CAMERA_FOV_DEGREES and ERROR_DEGREES must be positive")
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(
            f"Expected valid/images and valid/labels in {DATASET_PATH}"
        )

    if OUTPUT_DATASET_PATH.exists():
        shutil.rmtree(OUTPUT_DATASET_PATH)
    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)

    random_generator = random.Random(RANDOM_SEED)
    image_count = 0
    label_count = 0

    for image_path in sorted(source_images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        labels = read_labels(source_labels / f"{image_path.stem}.txt")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size
            crop_size = round(
                image_width * 2 * ERROR_DEGREES / CAMERA_FOV_DEGREES
            )
            crop_size = min(crop_size, image_width, image_height)

            if labels:
                target_x = labels[0][1] * image_width
                target_y = labels[0][2] * image_height
            else:
                target_x = image_width / 2
                target_y = image_height / 2

            if MODE == "random":
                pixels_per_degree_x = image_width / CAMERA_FOV_DEGREES
                pixels_per_degree_y = image_height / CAMERA_FOV_DEGREES
                target_x += random_generator.uniform(
                    -ERROR_DEGREES, ERROR_DEGREES
                ) * pixels_per_degree_x
                target_y += random_generator.uniform(
                    -ERROR_DEGREES, ERROR_DEGREES
                ) * pixels_per_degree_y

            crop_left = round(target_x - crop_size / 2)
            crop_top = round(target_y - crop_size / 2)
            crop_left = max(0, min(crop_left, image_width - crop_size))
            crop_top = max(0, min(crop_top, image_height - crop_size))

            cropped = image.crop(
                (crop_left, crop_top, crop_left + crop_size, crop_top + crop_size)
            )
            cropped.save(output_images / image_path.name)
            transformed_labels = crop_labels(
                labels, crop_left, crop_top, crop_size, image_width, image_height
            )
            (output_labels / f"{image_path.stem}.txt").write_text(
                "\n".join(transformed_labels), encoding="utf-8"
            )

        image_count += 1
        label_count += len(transformed_labels)

    source_yaml = DATASET_PATH / "data.yaml"
    dataset_config = source_yaml.read_text(encoding="utf-8")
    dataset_config = re.sub(
        r"(?m)^(train|val|test):.*$", r"\1: valid/images", dataset_config
    )
    (OUTPUT_DATASET_PATH / "data.yaml").write_text(
        dataset_config, encoding="utf-8"
    )
    return image_count, label_count


def main() -> None:
    image_count, label_count = create_cropped_dataset()
    print(f"Cropped dataset: {OUTPUT_DATASET_PATH}")
    print(f"Mode: {MODE}, crop FOV: {2 * ERROR_DEGREES} degrees")
    print(f"Images: {image_count}, labels: {label_count}")

    metrics = YOLO(str(MODEL_PATH)).val(
        data=str(OUTPUT_DATASET_PATH / "data.yaml"),
        split="val",
        imgsz=IMAGE_SIZE,
    )
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
