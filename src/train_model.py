import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------------------
# SUPPRESS WARNINGS
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: all, 1: INFO, 2: WARNING, 3: ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# -----------------------------
# IMPORTS
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout

# -----------------------------
# CONFIG (FAST TEST MODE)
# -----------------------------
DATA_PATH = "data/processed/nf_ton_iotv2_subset.csv"

IMG_SIZE = 64
EPOCHS = 6
BATCH_SIZE = 32
SAMPLE_SIZE = 5000

print("\n🚀 Starting Pipeline...\n")

start_time = time.time()

# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ Processed dataset not found. Run preprocessing first.")

df = pd.read_csv(DATA_PATH)

# reduce size for testing
df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)

if "Attack" not in df.columns:
    raise ValueError("❌ Target column 'Attack' missing!")

X = df.drop("Attack", axis=1).astype("float32")
y = df["Attack"]

print(f"✅ Dataset loaded: {X.shape}")

# -----------------------------
# LABEL ENCODING
# -----------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

print("✅ Classes:", list(encoder.classes_))

# -----------------------------
# TABULAR → IMAGE
# -----------------------------
def tabular_to_image(X):
    imgs = []

    for row in X.values:
        row = (row - np.min(row)) / (np.max(row) - np.min(row) + 1e-8)

        size = int(np.sqrt(len(row)))
        row = row[:size*size]

        img = row.reshape(size, size)
        img = tf.image.resize(img[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()
        img = np.repeat(img, 3, axis=-1)

        imgs.append(img)

    return np.array(imgs)

print("\n🔄 Converting tabular data → images...")
X_img = tabular_to_image(X)

print("✅ Image shape:", X_img.shape)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_img, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print("✅ Train shape:", X_train.shape)
print("✅ Test shape:", X_test.shape)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {k: round(float(v), 4) for k, v in enumerate(class_weights)}

# CLIP EXTREME WEIGHTS
for k in class_weights:
    class_weights[k] = min(class_weights[k], 5.0)

print("\n⚖️ Class Weights:", class_weights)

# -----------------------------
# MODEL BUILDER
# -----------------------------
def build_model(name):

    if name == "mobilenet":
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        base_model = DenseNet121(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )

    for layer in base_model.layers[:-20]:
      layer.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=2,
        restore_best_weights=True
    )
]

# -----------------------------
# TRAIN MOBILE NET
# -----------------------------
print("\n🧠 Training MobileNet...")
model1 = build_model("mobilenet")

model1.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------------
# TRAIN DENSE NET
# -----------------------------
print("\n🧠 Training DenseNet...")
model2 = build_model("densenet")

model2.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(model, name):

    pred = model.predict(X_test)
    pred_cls = np.argmax(pred, axis=1)

    acc = accuracy_score(y_test, pred_cls)

    print(f"\n📊 {name} Accuracy: {acc:.4f}")

    print(classification_report(
        y_test,
        pred_cls,
        target_names=encoder.classes_,
        zero_division=0   # 🔥 avoids warning
    ))

    return pred, acc

# -----------------------------
# EVALUATE
# -----------------------------
pred1, acc1 = evaluate(model1, "MobileNet")
pred2, acc2 = evaluate(model2, "DenseNet")

# -----------------------------
# ENSEMBLE
# -----------------------------
print("\n🔗 Running Ensemble...")

ensemble_pred = (pred1 + pred2) / 2
ensemble_cls = np.argmax(ensemble_pred, axis=1)

ensemble_acc = accuracy_score(y_test, ensemble_cls)

print(f"\n🚀 Ensemble Accuracy: {ensemble_acc:.4f}")

print(classification_report(
    y_test,
    ensemble_cls,
    target_names=encoder.classes_,
    zero_division=0
))

# -----------------------------
# TIME
# -----------------------------
end_time = time.time()

print("\n⏱ Total Execution Time:", round(end_time - start_time, 2), "seconds")

print("\n✅ Pipeline Step 2 Completed Successfully!")