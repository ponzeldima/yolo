import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    DepthwiseConv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
)


# Minimal MobileNetV1 building block: depthwise conv + pointwise conv.
def _depthwise_pointwise_block(x, pointwise_filters: int, stride: int, block_id: int):
    x = DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"conv_dw_{block_id}",
    )(x)
    x = BatchNormalization(name=f"conv_dw_{block_id}_bn")(x)
    x = ReLU(max_value=6.0, name=f"conv_dw_{block_id}_relu")(x)

    x = Conv2D(
        pointwise_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"conv_pw_{block_id}",
    )(x)
    x = BatchNormalization(name=f"conv_pw_{block_id}_bn")(x)
    x = ReLU(max_value=6.0, name=f"conv_pw_{block_id}_relu")(x)
    return x


def build_custom_mobilenet_v1_classifier(
    input_shape=(224, 224, 3),
    num_classes: int = 2,
    alpha: float = 1.0,
    dropout_rate: float = 0.2,
) -> Model:
    """Custom MobileNetV1-style classifier created manually with Keras layers."""
    def make_divisible(v: int, divisor: int = 8) -> int:
        return max(divisor, int(v + divisor / 2) // divisor * divisor)

    def c(filters: int) -> int:
        return make_divisible(int(filters * alpha))

    inputs = Input(shape=input_shape, name="input_image")

    x = Conv2D(c(32), 3, strides=2, padding="same", use_bias=False, name="conv1")(inputs)
    x = BatchNormalization(name="conv1_bn")(x)
    x = ReLU(max_value=6.0, name="conv1_relu")(x)

    x = _depthwise_pointwise_block(x, c(64), stride=1, block_id=1)
    x = _depthwise_pointwise_block(x, c(128), stride=2, block_id=2)
    x = _depthwise_pointwise_block(x, c(128), stride=1, block_id=3)
    x = _depthwise_pointwise_block(x, c(256), stride=2, block_id=4)
    x = _depthwise_pointwise_block(x, c(256), stride=1, block_id=5)
    x = _depthwise_pointwise_block(x, c(512), stride=2, block_id=6)

    for i in range(7, 12):
        x = _depthwise_pointwise_block(x, c(512), stride=1, block_id=i)

    x = _depthwise_pointwise_block(x, c(1024), stride=2, block_id=12)
    x = _depthwise_pointwise_block(x, c(1024), stride=1, block_id=13)

    x = GlobalAveragePooling2D(name="global_pool")(x)
    x = Dropout(dropout_rate, name="dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="custom_mobilenet_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model


def build_keras_mobilenet_v1_classifier(
    input_shape=(224, 224, 3),
    num_classes: int = 2,
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
    return model


if __name__ == "__main__":
    custom_model = build_custom_mobilenet_v1_classifier()
    keras_model = build_keras_mobilenet_v1_classifier()

    print("Custom MobileNetV1 compiled")
    custom_model.summary()

    print("\nKeras MobileNetV1 compiled")
    keras_model.summary()
