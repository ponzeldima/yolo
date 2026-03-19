import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_and_prepare_image(image_path: Path) -> np.ndarray:
    """Load image as 28x28 grayscale tensor and normalize to [0, 1]."""
    image = tf.keras.utils.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    image_arr = tf.keras.utils.img_to_array(image)
    image_arr = image_arr.astype("float32") / 255.0

    # Model expects batch dimension: (1, 28, 28, 1)
    return np.expand_dims(image_arr, axis=0)


def predict_digit(model_path: Path, image_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(model_path)
    input_tensor = load_and_prepare_image(image_path)

    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    top3_idx = np.argsort(probabilities)[-3:][::-1]

    print(f"Image: {image_path}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f}")
    print("Top-3 classes:")
    for idx in top3_idx:
        print(f"  {int(idx)} -> {float(probabilities[idx]):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict MNIST digit from one image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("mnist_dense_from_project.keras"),
        help="Path to trained Keras model.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to image file (png/jpg).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_digit(model_path=args.model, image_path=args.image)
