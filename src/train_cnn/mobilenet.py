import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/images"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5


# -----------------------------
# Data Generators
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

num_classes = train_gen.num_classes


# -----------------------------
# Model Builder (for tuner)
# -----------------------------
def build_model(hp):

    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Tune number of frozen layers
    freeze_layers = hp.Int("freeze_layers", min_value=50, max_value=100, step=25)

    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Tune dense units
    x = Dense(
        units=hp.Choice("dense_units", [128, 256, 512]),
        activation="relu"
    )(x)

    # Tune dropout
    x = Dropout(
        rate=hp.Choice("dropout", [0.3, 0.4, 0.5])
    )(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Tune learning rate
    lr = hp.Choice("learning_rate", [1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Keras Tuner Setup
# -----------------------------
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=2,              # number of combinations to try
    executions_per_trial=1,
    directory="tuner_results",
    project_name="nfiiot_ids"
)


print("\nStarting Hyperparameter Search...\n")

tuner.search(
    train_gen,
    validation_data=val_gen,
    epochs=3   # keep small for tuning phase
)


# -----------------------------
# Get Best Model
# -----------------------------
best_model = tuner.get_best_models(num_models=1)[0]

best_hp = tuner.get_best_hyperparameters(1)[0]

print("\nBest Hyperparameters:")
for param in best_hp.values:
    print(param, ":", best_hp.get(param))


# -----------------------------
# Train Final Model (FULL TRAINING)
# -----------------------------
print("\nTraining final model...\n")

history = best_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)


# -----------------------------
# Evaluate Model
# -----------------------------
print("\nEvaluating model...\n")

val_gen.reset()
y_true = val_gen.classes

y_pred = best_model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=val_gen.class_indices.keys()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))


# -----------------------------
# Save Model
# -----------------------------
best_model.save("models.mobilenet_tuned.keras")

print("\nModel saved successfully.")