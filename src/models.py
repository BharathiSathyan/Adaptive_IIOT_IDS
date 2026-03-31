import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(name, input_shape, num_classes, lr, dropout, dense_units):

    if name == "densenet":
        base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model")

    for layer in base.layers[:-20]:
        layer.trainable = False
    for layer in base.layers[-20:]:
        layer.trainable = True

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)

    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model