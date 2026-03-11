import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, Softmax
from tensorflow.keras.models import Model

FOMO_DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "fomo" / "dataset_test_244-fomo"
BATCH_SIZE = 16
EPOCHS = 200


class EpochMetricsPrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss", 0.0)
        train_acc = logs.get("accuracy", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        val_precision = logs.get("val_precision", 0.0)
        val_recall = logs.get("val_recall", 0.0)

        print(
            f"\n[Epoch {epoch + 1}] "
            f"loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"val_precision={val_precision:.4f}, val_recall={val_recall:.4f}"
        )


def read_fomo_meta(dataset_dir: Path) -> dict:
    meta_path = dataset_dir / "fomo_dataset.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Не знайдено файл метаданих: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_npy_label(label_path_tensor):
    label_path = label_path_tensor
    if isinstance(label_path, np.ndarray):
        label_path = label_path.item()
    if isinstance(label_path, bytes):
        label_path = label_path.decode("utf-8")

    label = np.load(label_path).astype(np.float32)
    return label


def create_split_dataset(
    dataset_dir: Path,
    split_name: str,
    image_size: int,
    grid_size: int,
    classes_count: int,
    batch_size: int,
    training: bool,
):
    images_dir = dataset_dir / split_name / "images"
    labels_dir = dataset_dir / split_name / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Не знайдено split '{split_name}' у {dataset_dir}")

    image_files = sorted(
        [*images_dir.glob("*.jpg"), *images_dir.glob("*.jpeg"), *images_dir.glob("*.png")]
    )
    label_by_name = {path.stem: path for path in labels_dir.glob("*.npy")}

    pairs = [(img, label_by_name[img.stem]) for img in image_files if img.stem in label_by_name]
    if not pairs:
        raise ValueError(f"У split '{split_name}' не знайдено пар image/label")

    image_paths, label_paths = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in image_paths], [str(p) for p in label_paths]))

    def _map_fn(image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape((image_size, image_size, 3))

        label = tf.numpy_function(_load_npy_label, [label_path], tf.float32)
        label.set_shape((grid_size, grid_size, classes_count))
        return image, label

    if training:
        ds = ds.shuffle(buffer_size=min(len(pairs), 1024), reshuffle_each_iteration=True)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(pairs)


def create_model(image_size: int, classes_count: int, grid_size: int) -> Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=0.35,
        include_top=False,
        weights="imagenet",
    )

    # Беремо шар зі spatial-розміром 12x12 для input 96x96, щоб збігався з FOMO-мітками.
    feature_layer = base_model.get_layer("block_6_expand_relu").output
    x = Conv2D(classes_count, (1, 1), padding="same")(feature_layer)
    x = Softmax(axis=-1)(x)

    model = Model(inputs=base_model.input, outputs=x)

    output_h, output_w = model.output_shape[1], model.output_shape[2]
    if output_h != grid_size or output_w != grid_size:
        raise ValueError(
            f"Розмір виходу моделі {output_h}x{output_w} не збігається з grid {grid_size}x{grid_size}"
        )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    meta = read_fomo_meta(FOMO_DATASET_DIR)
    image_size = int(meta["image_size"])
    grid_size = int(meta["grid_size"])
    classes_count = int(meta["num_classes_total"])

    train_dataset, train_count = create_split_dataset(
        dataset_dir=FOMO_DATASET_DIR,
        split_name="train",
        image_size=image_size,
        grid_size=grid_size,
        classes_count=classes_count,
        batch_size=BATCH_SIZE,
        training=True,
    )
    valid_dataset, valid_count = create_split_dataset(
        dataset_dir=FOMO_DATASET_DIR,
        split_name="valid",
        image_size=image_size,
        grid_size=grid_size,
        classes_count=classes_count,
        batch_size=BATCH_SIZE,
        training=False,
    )

    print(f"Train samples: {train_count}")
    print(f"Valid samples: {valid_count}")
    print(f"Classes: {meta['class_names']}")

    fomo_model = create_model(image_size=image_size, classes_count=classes_count, grid_size=grid_size)
    fomo_model.summary()

    fomo_model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        callbacks=[EpochMetricsPrinter()],
    )

    val_metrics = fomo_model.evaluate(valid_dataset, return_dict=True, verbose=0)
    print("\n=== Validation Metrics (final) ===")
    print(f"val_loss: {val_metrics.get('loss', 0.0):.4f}")
    print(f"val_accuracy: {val_metrics.get('accuracy', 0.0):.4f}")
    print(f"val_precision: {val_metrics.get('precision', 0.0):.4f}")
    print(f"val_recall: {val_metrics.get('recall', 0.0):.4f}")

    out_model_path = FOMO_DATASET_DIR / "fomo_model.keras"
    fomo_model.save(out_model_path)
    print(f"Модель збережено: {out_model_path}")


if __name__ == "__main__":
    main()