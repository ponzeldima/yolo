from pathlib import Path

import numpy as np
import tensorflow as tf


DATASET_ROOT = Path("datasets/minst")
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"


def _save_split_images(images: np.ndarray, labels: np.ndarray, split_dir: Path):
    """Save MNIST arrays as PNG files grouped by class folders."""
    split_dir.mkdir(parents=True, exist_ok=True)

    class_counts = {digit: 0 for digit in range(10)}

    for image, label in zip(images, labels):
        class_dir = split_dir / str(int(label))
        class_dir.mkdir(parents=True, exist_ok=True)

        image_id = class_counts[int(label)]
        image_name = f"{int(label)}_{image_id:05d}.png"
        image_path = class_dir / image_name

        # save_img expects HxW or HxWxC, so we add an explicit channel axis for grayscale.
        image_for_save = image.astype(np.uint8)[..., np.newaxis]
        tf.keras.utils.save_img(image_path, image_for_save, scale=False)

        class_counts[int(label)] += 1


def download_and_export_mnist():
    """Download MNIST and export it to datasets/minst as image files."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    _save_split_images(x_train, y_train, TRAIN_DIR)
    _save_split_images(x_test, y_test, TEST_DIR)

    print(f"Saved train images to: {TRAIN_DIR}")
    print(f"Saved test images to: {TEST_DIR}")
    print("MNIST export complete")


if __name__ == "__main__":
    download_and_export_mnist()
