from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout


DATASET_ROOT = Path("datasets/minst")
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
MODEL_SAVE_PATH = Path("mnist_dense_from_project.keras")

def build_keras_mobilenet_v1_classifier(
    input_shape=(28, 28, 1),
    num_classes: int = 10,
    alpha: float = 1.0,
    dropout_rate: float = 0.2,
    use_imagenet_weights: bool = True,
) -> Model:
    """MobileNetV1 classifier using tf.keras.applications.MobileNet as a backbone."""
    weights = "imagenet" if use_imagenet_weights else None

    backbone = tf.keras.applications.MobileNet(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        pooling="avg",
        weights=weights,
    )

    inputs = Input(shape=input_shape, name="input_image")
    x = backbone(inputs)
    x = Dropout(dropout_rate, name="dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="keras_mobilenet_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    model.summary()

    return model

def build_dense_mnist_model(
    input_shape=(28, 28, 1),
    num_classes: int = 10,
    hidden_units=(256, 128),
    dropout_rate: float = 0.2,
) -> Model:
    """Build a simple MLP for MNIST (no convolution layers)."""
    inputs = Input(shape=input_shape, name="mnist_image")
    x = Flatten(name="flatten")(inputs)

    for idx, units in enumerate(hidden_units, start=1):
        x = Dense(units, activation="relu", name=f"dense_{idx}")(x)
        x = Dropout(dropout_rate, name=f"dropout_{idx}")(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="mnist_dense_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model


def make_dataset_from_directory(path: Path, batch_size: int):
    """Create tf.data dataset from class subfolders with grayscale images."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=(28, 28),
        batch_size=batch_size,
        shuffle=True,
    )

    # Normalize image pixels to [0, 1].
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    return dataset.prefetch(tf.data.AUTOTUNE)


def train_model_from_project_images(epochs: int = 10, batch_size: int = 128):
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Expected dataset folders not found: {TRAIN_DIR} and/or {TEST_DIR}. "
            "Run download_mnist_to_project.py first."
        )

    train_dataset = make_dataset_from_directory(TRAIN_DIR, batch_size=batch_size)
    test_dataset = make_dataset_from_directory(TEST_DIR, batch_size=batch_size)

    # model = build_dense_mnist_model()
    model = build_keras_mobilenet_v1_classifier()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            mode="max",
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    model.save(MODEL_SAVE_PATH)

    print(f"Saved model to: {MODEL_SAVE_PATH}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    return model, history


if __name__ == "__main__":
    train_model_from_project_images()
