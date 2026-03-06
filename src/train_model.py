import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "data/processed/nf_ton_iotv2_subset.csv"
MODEL_PATH = "models/ids_model.h5"


# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded")
print("Shape:", df.shape)


# -----------------------------
# Split Features / Target
# -----------------------------

X = df.drop("Attack", axis=1)
y = df["Attack"]

# convert to float for tensorflow
X = X.astype("float32")


# -----------------------------
# Encode Labels
# -----------------------------

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))

print("\nAttack classes:")
print(encoder.classes_)


# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# -----------------------------
# Compute Class Weights
# -----------------------------

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(zip(np.unique(y_train), class_weights))

print("\nClass weights:")
print(class_weights)


# -----------------------------
# Build Neural Network
# -----------------------------

model = Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# -----------------------------
# Train Model
# -----------------------------

print("\nTraining model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)


# -----------------------------
# Evaluate Model
# -----------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_))


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred_classes)

print("\nConfusion Matrix:")
print(cm)


# -----------------------------
# Save Model
# -----------------------------

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)