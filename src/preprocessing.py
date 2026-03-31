import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "data/raw/NF-ToN-IoT-V2.parquet"
OUTPUT_PATH = "data/processed/nf_ton_iotv2_subset.csv"

SAMPLES_PER_CLASS = 3000  

def load_dataset():
    print("\nLoading dataset...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sample(n=200000, random_state=42)
    print("Samples Shape:", df.shape)
    return df

def clean_dataset(df):
    print("\nCleaning dataset...")

    df = df.drop_duplicates()

    df = df.replace(["?", "-", "NaN"], np.nan)
    df = df.fillna(-1)

    df = df.drop(columns=["Label"], errors="ignore")

    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    df = df.drop(columns=drop_cols, errors="ignore")

    print("After cleaning:", df.shape)
    return df

def stratified_sampling(df):
    print("\nApplying stratified sampling...")

    frames = []
    for cls in df["Attack"].unique():
        subset = df[df["Attack"] == cls]

        if len(subset) >= SAMPLES_PER_CLASS:
            subset = subset.sample(SAMPLES_PER_CLASS, random_state=42)

        frames.append(subset)

    df = pd.concat(frames).reset_index(drop=True)

    print("After sampling:", df.shape)
    print(df["Attack"].value_counts())
    return df

def encode(df):
    if "PROTOCOL" in df.columns:
        df = pd.get_dummies(df, columns=["PROTOCOL"])
    return df

def normalize(df):
    print("\nNormalizing...")

    X = df.drop("Attack", axis=1)
    y = df["Attack"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    return df

def main():
    print("Pre-Processing Pipeline Starts...")
    df = load_dataset()
    df = clean_dataset(df)
    df = stratified_sampling(df)
    df = encode(df)
    df = normalize(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()