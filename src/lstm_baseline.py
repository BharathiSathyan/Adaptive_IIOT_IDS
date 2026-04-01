import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os


def build_lstm(input_shape, num_classes, lr, dropout, units):

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),

        LSTM(units // 2),
        Dropout(dropout),

        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_lstm(X_train, y_train, X_val, y_val, class_weights):

    # SAFE hyperparameters 
    lr = 1e-3
    dropout = 0.3
    units = 64

    print(f"\nTraining LSTM with lr={lr}, dropout={dropout}, units={units}")

    model = build_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=len(np.unique(y_train)),
        lr=lr,
        dropout=dropout,
        units=units
    )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    checkpoint_path = "checkpoints/lstm.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger("logs/lstm.csv")
    ]

    #  SAFE resume (no crash if architecture changes)
    try:
        if os.path.exists(checkpoint_path):
            print("Trying to resume from checkpoint...")
            model.load_weights(checkpoint_path)
    except:
        print("Checkpoint incompatible. Training fresh model.")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,              # keep small for your system
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    model.save("models/lstm.keras")

    return model