import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from lstm_baseline import train_lstm


DATA_PATH = "data/processed/nf_ton_iotv2_subset.csv"


# NON-OVERLAPPING + SAFE
def tabular_to_sequence(X, y, seq_len=10):
    X_seq, y_seq = [], []

    for i in range(0, len(X) - seq_len, seq_len):
        X_seq.append(X.iloc[i:i+seq_len].values)
        y_seq.append(y[i + seq_len - 1])

    return np.array(X_seq), np.array(y_seq)


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Attack", axis=1).astype("float32")
    y = df["Attack"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # SPLIT FIRST (NO LEAKAGE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    print("Converting to sequences...")

    X_train, y_train = tabular_to_sequence(X_train, y_train)
    X_val, y_val = tabular_to_sequence(X_val, y_val)
    X_test, y_test = tabular_to_sequence(X_test, y_test)

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    # class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights = {i: float(w) for i, w in enumerate(weights)}

    # train
    model = train_lstm(X_train, y_train, X_val, y_val, weights)

    # evaluation
    preds = model.predict(X_test)
    preds_cls = np.argmax(preds, axis=1)

    acc = np.mean(preds_cls == y_test)
    print("\nLSTM Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, preds_cls))


if __name__ == "__main__":
    main()