import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: all, 1: INFO, 2: WARNING, 3: ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from models import build_model
from utils import evaluate, ensemble


DATA_PATH = "data/processed/nf_ton_iotv2_subset.csv"
IMG_SIZE = 64

def tabular_to_image(X):
    imgs = []
    for row in X.values:
        row = (row - row.min()) / (row.max() - row.min() + 1e-8)
        img = np.tile(row, (len(row), 1))
        img = tf.image.resize(img[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()
        img = np.repeat(img, 3, axis=-1)
        imgs.append(img)
    return np.array(imgs)

def train_single(model_name, X_train, y_train, X_val, y_val, class_weights):

    lr = random.choice([1e-3, 3e-4, 1e-4])
    dropout = random.choice([0.3, 0.4, 0.5])
    dense = random.choice([128, 256])

    print(f"\nTraining {model_name} with lr={lr}, dropout={dropout}, dense={dense}")

    model = build_model(model_name, (IMG_SIZE, IMG_SIZE, 3), len(np.unique(y_train)), lr, dropout, dense)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    path = f"checkpoints/{model_name}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.CSVLogger(f"logs/{model_name}.csv"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    if os.path.exists(path):
        print("Resuming from checkpoint:", path)
        model.load_weights(path)

    start = time.time()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("Training time:", round(time.time() - start, 2))

    model.save(f"models/{model_name}.keras")

    return model

def tabular_to_sequence(X, seq_len=10):
    sequences = []
    for i in range(len(X) - seq_len):
        sequences.append(X.iloc[i:i+seq_len].values)
    return np.array(sequences)

def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Attack", axis=1).astype("float32")
    y = df["Attack"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_img = tabular_to_image(X)

    X_train, X_test, y_train, y_test = train_test_split(X_img, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    weights = {i: float(w) for i, w in enumerate(weights)}

    models = ["densenet", "mobilenet"]
    trained = []
    preds = []

    for m in models:
        print(f"\n==============================")
        print(f"STARTING MODEL: {m.upper()}")
        print(f"==============================")
        model = train_single(m, X_train, y_train, X_val, y_val, weights)
        trained.append(model)

    from utils import evaluate

    # collect accuracies
    accs = []

    for model, name in zip(trained, models):
        p, acc = evaluate(model, X_test, y_test, encoder, name)
        preds.append(p)
        accs.append(acc)
            
   # dynamic weights
    weights = np.array(accs) / np.sum(accs)
    print("\nDynamic Ensemble Weights:", weights)

    # ensemble
    final = ensemble(preds, weights)
    final_cls = np.argmax(final, axis=1)

    print("\nEnsemble Accuracy:", np.mean(final_cls == y_test))

    

if __name__ == "__main__":
    main()