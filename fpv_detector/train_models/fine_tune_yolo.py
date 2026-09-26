from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = Path("datasets")
RUNS_DIR = Path("runs") / "detect"

# Change these four values for the next fine-tuning session.
OLD_DATASET_YAML = DATASETS_DIR / "fpv_public_datasts_310826" / "data.yaml"
NEW_DATASET_YAML = DATASETS_DIR / "dataset_3" / "data.yaml"
BASE_WEIGHTS = RUNS_DIR / "v8n_640_fpv_public_datasts_310826" / "weights" / "best.pt"
FINE_TUNE_RUN_NAME = "v8n_640_fpv_public_datasts_310826_plus_v8n_640_dataset_3"

# All new train images are used. This is the maximum number sampled from the old train split.
OLD_TRAIN_SAMPLE_SIZE = 3000
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO on a new dataset while replaying sampled old images."
    )
    parser.add_argument("--old-data", type=Path, default=OLD_DATASET_YAML)
    parser.add_argument("--new-data", type=Path, default=NEW_DATASET_YAML)
    parser.add_argument("--weights", type=Path, default=BASE_WEIGHTS)
    parser.add_argument("--run-name", default=FINE_TUNE_RUN_NAME)
    parser.add_argument("--old-images", type=int, default=OLD_TRAIN_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create the mixed dataset manifests without starting training.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return config


def class_names(config: dict, yaml_path: Path) -> list[str]:
    names = config.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            return [str(names[index]) for index in range(len(names))]
        except KeyError as error:
            raise ValueError(
                f"Class ids in {yaml_path} must be consecutive and start at 0."
            ) from error
    raise ValueError(f"Missing 'names' in {yaml_path}")


def resolve_image_directory(config: dict, yaml_path: Path, split: str) -> Path:
    split_path = config.get(split)
    if not isinstance(split_path, str):
        raise ValueError(f"Missing string '{split}' path in {yaml_path}")

    configured_root = Path(config.get("path", yaml_path.parent))
    if not configured_root.is_absolute():
        configured_root = yaml_path.parent / configured_root

    raw_path = Path(split_path)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend((configured_root / raw_path, yaml_path.parent / raw_path))

    # Some existing YAML files use '../train/images' while train/ is beside data.yaml.
    if len(raw_path.parts) >= 2 and raw_path.parts[-1].lower() == "images":
        candidates.append(yaml_path.parent / raw_path.parts[-2] / "images")
    candidates.append(yaml_path.parent / split / "images")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find '{split}' images for {yaml_path}. Checked:\n  {checked}")


def image_paths(image_directory: Path) -> list[Path]:
    images = sorted(
        path.resolve()
        for path in image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in {image_directory}")
    return images


def write_manifest(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")


def prepare_mixed_dataset(arguments: argparse.Namespace) -> Path:
    if arguments.old_images < 0:
        raise ValueError("--old-images must be zero or greater")

    old_config = load_yaml(arguments.old_data)
    new_config = load_yaml(arguments.new_data)
    old_names = class_names(old_config, arguments.old_data)
    new_names = class_names(new_config, arguments.new_data)
    if old_names != new_names:
        raise ValueError(
            "The old and new datasets use different classes or class-id order:\n"
            f"old: {old_names}\nnew: {new_names}"
        )

    old_images = image_paths(resolve_image_directory(old_config, arguments.old_data, "train"))
    new_images = image_paths(resolve_image_directory(new_config, arguments.new_data, "train"))
    new_validation_images = image_paths(
        resolve_image_directory(new_config, arguments.new_data, "val")
    )

    sample_size = min(arguments.old_images, len(old_images))
    old_sample = random.Random(arguments.seed).sample(old_images, sample_size)
    mixed_images = new_images + old_sample
    random.Random(arguments.seed).shuffle(mixed_images)

    output_directory = DATASETS_DIR / "fine_tune_mix" / arguments.run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    train_manifest = output_directory / "train_images.txt"
    validation_manifest = output_directory / "val_images.txt"
    mixed_yaml = output_directory / "data.yaml"
    write_manifest(train_manifest, mixed_images)
    write_manifest(validation_manifest, new_validation_images)
    mixed_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str(train_manifest.resolve()),
                "val": str(validation_manifest.resolve()),
                "names": {index: name for index, name in enumerate(old_names)},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print(f"Mixed dataset YAML: {mixed_yaml}")
    print(f"New train images: {len(new_images)}")
    print(f"Old train images sampled: {sample_size} of {len(old_images)}")
    print(f"Total train images: {len(mixed_images)}")
    print(f"New validation images: {len(new_validation_images)}")
    return mixed_yaml


def main() -> None:
    arguments = parse_arguments()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    mixed_yaml = prepare_mixed_dataset(arguments)
    if arguments.prepare_only:
        return

    last_weights = RUNS_DIR / arguments.run_name / "weights" / "last.pt"
    if last_weights.is_file():
        print(f"Resuming interrupted fine-tuning from: {last_weights}")
        YOLO(str(last_weights)).train(resume=True)
        return

    if not arguments.weights.is_file():
        raise FileNotFoundError(f"Base model weights were not found: {arguments.weights}")

    print(f"Fine-tuning from: {arguments.weights}")
    YOLO(str(arguments.weights)).train(
        data=str(mixed_yaml),
        epochs=50,
        imgsz=640,
        batch=64,
        device="cuda",
        cache=True,
        # project=str(RUNS_DIR),
        name=arguments.run_name,
        save=True,
        plots=True,
        workers=0,
        degrees=15.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )


if __name__ == "__main__":
    main()