"""
Memory-safe preprocessing pipeline for NF-ToN-IoTv2 dataset.

Steps:
1. Load small random fraction of dataset
2. Clean dataset
3. Stratified sampling per attack class
4. One-hot encode protocol feature
5. Min-max normalization
6. Save processed dataset
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "data/raw/NF-ToN-IoT-V2.parquet"
OUTPUT_PATH = "data/processed/nf_ton_iotv2_subset.csv"

# how much of the raw dataset to initially load
INITIAL_SAMPLE_FRACTION = 0.1

# final samples per attack class
SAMPLES_PER_CLASS = 5000


# -----------------------------
# Step 1 — Load Dataset (Sampled)
# -----------------------------
def load_dataset(path):

    print("Loading dataset...")

    df = pd.read_parquet(path)

    # take small fraction first to reduce memory
    df = df.sample(frac=INITIAL_SAMPLE_FRACTION, random_state=42)

    print("Dataset sampled")
    print("Shape:", df.shape)

    return df


# -----------------------------
# Step 2 — Clean Dataset
# -----------------------------
def clean_dataset(df):

    # remove binary label
    df = df.drop(columns=["Label"], errors="ignore")

    # drop identifier columns if present
    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    df = df.drop(columns=drop_cols, errors="ignore")

    print("Dataset cleaned")
    print("Shape:", df.shape)

    return df


# -----------------------------
# Step 3 — Stratified Sampling
# -----------------------------
def stratified_sampling(df, label_column="Attack", samples_per_class=5000):

    sampled_frames = []

    for attack_class in df[label_column].unique():

        class_df = df[df[label_column] == attack_class]

        if len(class_df) > samples_per_class:
            class_df = class_df.sample(samples_per_class, random_state=42)

        sampled_frames.append(class_df)

    df_sampled = pd.concat(sampled_frames).reset_index(drop=True)

    print("\nStratified sampling completed")
    print("New dataset size:", df_sampled.shape)

    print("\nClass distribution:")
    print(df_sampled[label_column].value_counts())

    return df_sampled


# -----------------------------
# Step 4 — Encode Protocol
# -----------------------------
def encode_protocol(df):

    if "PROTOCOL" in df.columns:
        df = pd.get_dummies(df, columns=["PROTOCOL"])

    print("Protocol encoding completed")

    return df


# -----------------------------
# Step 5 — Normalize Features
# -----------------------------
def normalize_features(df, label_column="Attack"):

    scaler = MinMaxScaler()

    X = df.drop(label_column, axis=1)
    y = df[label_column]

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    df_normalized = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    print("Normalization completed")

    return df_normalized


# -----------------------------
# Step 6 — Save Dataset
# -----------------------------
def save_dataset(df, path):

    df.to_csv(path, index=False)

    print("Processed dataset saved to:", path)


# -----------------------------
# Main Pipeline
# -----------------------------
def main():

    df = load_dataset(DATA_PATH)

    df = clean_dataset(df)

    df = stratified_sampling(
        df,
        label_column="Attack",
        samples_per_class=SAMPLES_PER_CLASS
    )

    df = encode_protocol(df)

    df = normalize_features(
        df,
        label_column="Attack"
    )

    save_dataset(df, OUTPUT_PATH)

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()