import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout


def load_mnist_data():
    """Load MNIST and normalize pixel values to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_dense_mnist_model(
    input_shape=(28, 28),
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
    return model


def train_mnist_dense_model(
    epochs: int = 10,
    batch_size: int = 128,
    model_save_path: str = "mnist_dense.keras",
):
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    model = build_dense_mnist_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            mode="max",
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(model_save_path)

    print(f"Saved model to: {model_save_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    return model, history


if __name__ == "__main__":
    train_mnist_dense_model()
