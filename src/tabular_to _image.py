import numpy as np
import pandas as pd
import cv2
import os

# -----------------------------
# Configuration
# -----------------------------
INPUT_PATH = "data/processed/nf_ton_iotv2_subset.csv"
OUTPUT_DIR = "data/images"

IMAGE_SIZE = 224


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(INPUT_PATH)

X = df.drop("Attack", axis=1).values
y = df["Attack"].values

# encode labels (needed for folder saving)
labels = np.unique(y)
label_to_idx = {label: idx for idx, label in enumerate(labels)}

print("Total samples:", len(X))
print("Classes:", labels)


# -----------------------------
# Create Output Folders
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in labels:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)


# -----------------------------
# Convert Tabular → Image
# -----------------------------
def row_to_image(row):
    """
    Convert a single row into square image
    """

    # normalize to 0–255
    row = (row - row.min()) / (row.max() - row.min() + 1e-8)
    row = (row * 255).astype(np.uint8)

    # find square size
    size = int(np.sqrt(len(row)))

    # trim to square
    row = row[:size * size]

    # reshape to square
    img = row.reshape(size, size)

    # resize to 224x224
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # convert to 3 channels
    img = np.stack([img, img, img], axis=-1)

    return img


# -----------------------------
# Generate Images
# -----------------------------
print("\nGenerating images...")

for i in range(len(X)):

    img = row_to_image(X[i])
    label = y[i]

    save_path = os.path.join(
        OUTPUT_DIR,
        label,
        f"img_{i}.png"
    )

    cv2.imwrite(save_path, img)

    if i % 5000 == 0:
        print(f"Processed {i}/{len(X)}")

print("\nImage generation completed.")