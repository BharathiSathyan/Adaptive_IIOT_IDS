"""
Preprocessing pipeline for NF-TON-IoTv2 dataset.

Steps implemented:
1. Load dataset
2. Remove duplicates
3. Remove unnecessary identifier features
4. Stratified sampling per attack class
5. One-hot encode protocol feature
6. Min-max normalization
7. Save processed subset dataset

The subset dataset will be used for the remaining project steps:
- tabular → image transformation
- CNN training
- GA optimization
- ensemble learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "data/raw/nf_ton_iotv2.csv"
OUTPUT_PATH = "data/processed/nf_ton_iotv2_subset.csv"

# number of samples to keep per attack class
SAMPLES_PER_CLASS = 20000


# -----------------------------
# Step 1 — Load Dataset
# -----------------------------

def load_dataset(path):
    """
    Load dataset from CSV or parquet.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print("Dataset loaded")
    print("Shape:", df.shape)

    return df


# -----------------------------
# Step 2 — Clean Dataset
# -----------------------------

def clean_dataset(df):
    """
    Remove duplicates and unnecessary features.
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove identifier columns that do not help learning
    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

    df = df.drop(columns=drop_cols, errors="ignore")

    print("Dataset cleaned")
    print("Shape after cleaning:", df.shape)

    return df


# -----------------------------
# Step 3 — Stratified Sampling
# -----------------------------

def stratified_sampling(df, label_column="label", samples_per_class=20000):
    """
    Create a balanced subset dataset by sampling the same number
    of rows from each attack class.

    This ensures:
    - all attack classes are present
    - dataset remains manageable for CNN training
    """

    sampled_frames = []

    classes = df[label_column].unique()

    for c in classes:

        class_df = df[df[label_column] == c]

        if len(class_df) > samples_per_class:
            class_df = class_df.sample(samples_per_class, random_state=42)

        sampled_frames.append(class_df)

    df_sampled = pd.concat(sampled_frames).reset_index(drop=True)

    print("Stratified sampling completed")
    print("New dataset size:", df_sampled.shape)

    print("\nClass distribution:")
    print(df_sampled[label_column].value_counts())

    return df_sampled


# -----------------------------
# Step 4 — Encode Protocol
# -----------------------------

def encode_protocol(df):
    """
    Convert categorical PROTOCOL feature into one-hot encoding.
    """

    if "PROTOCOL" in df.columns:
        df = pd.get_dummies(df, columns=["PROTOCOL"])

    print("Protocol encoding completed")

    return df


# -----------------------------
# Step 5 — Normalize Features
# -----------------------------

def normalize_features(df, label_column="label"):
    """
    Apply Min-Max normalization to all numeric features.
    """

    scaler = MinMaxScaler()

    X = df.drop(label_column, axis=1)
    y = df[label_column]

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    df_normalized = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    print("Normalization completed")

    return df_normalized


# -----------------------------
# Step 6 — Save Processed Dataset
# -----------------------------

def save_dataset(df, path):
    """
    Save processed dataset.
    """

    df.to_csv(path, index=False)

    print("Processed dataset saved to:", path)


# -----------------------------
# Main Pipeline
# -----------------------------

def main():

    # Load dataset
    df = load_dataset(DATA_PATH)

    # Clean dataset
    df = clean_dataset(df)

    # Stratified sampling
    df = stratified_sampling(df, label_column="label", samples_per_class=SAMPLES_PER_CLASS)

    # Encode protocol
    df = encode_protocol(df)

    # Normalize dataset
    df = normalize_features(df, label_column="label")

    # Save processed dataset
    save_dataset(df, OUTPUT_PATH)

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()