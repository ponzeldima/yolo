from __future__ import annotations

import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_NAME = "own_1080"
NEW_DATASET_NAME = f"{DATASET_NAME}_filtered"
IMAGE_PREFIX = "IMG_3157"


def create_filtered_dataset(
	source_dataset: Path,
	output_dataset: Path,
	filename_prefix: str,
) -> tuple[int, int]:
	source_images = source_dataset / "valid" / "images"
	source_labels = source_dataset / "valid" / "labels"
	output_images = output_dataset / "valid" / "images"
	output_labels = output_dataset / "valid" / "labels"

	if not source_images.is_dir():
		raise FileNotFoundError(f"Images directory not found: {source_images}")
	if not source_labels.is_dir():
		raise FileNotFoundError(f"Labels directory not found: {source_labels}")

	output_images.mkdir(parents=True, exist_ok=True)
	output_labels.mkdir(parents=True, exist_ok=True)

	copied_images = 0
	copied_labels = 0
	for image_path in sorted(source_images.iterdir()):
		if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
			continue
		if not image_path.name.startswith(filename_prefix):
			continue

		shutil.copy2(image_path, output_images / image_path.name)
		copied_images += 1

		label_path = source_labels / f"{image_path.stem}.txt"
		if label_path.is_file():
			shutil.copy2(label_path, output_labels / label_path.name)
			copied_labels += 1

	source_yaml = source_dataset / "data.yaml"
	shutil.copy2(source_yaml, output_dataset / "data.yaml")

	return copied_images, copied_labels


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	source_dataset = script_dir.parent / "datasets" / DATASET_NAME
	output_dataset = source_dataset.parent / NEW_DATASET_NAME
	copied_images, copied_labels = create_filtered_dataset(
		source_dataset, output_dataset, IMAGE_PREFIX
	)
	print(f"Dataset saved to: {output_dataset}")
	print(f"Images copied: {copied_images}")
	print(f"Labels copied: {copied_labels}")


if __name__ == "__main__":
	main()
