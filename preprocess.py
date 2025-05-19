import os
import cv2
import numpy as np
from tqdm import tqdm  # Progress bar

data_dir = "chest_xray"
sets = ["train", "val", "test"]
categories = ["NORMAL", "PNEUMONIA"]

IMG_SIZE = 150  # Resize all images to 150x150

def load_data(dataset):
    X = []
    y = []
    for category in categories:
        folder_path = os.path.join(data_dir, dataset, category)
        label = 0 if category == "NORMAL" else 1  # NORMAL = 0, PNEUMONIA = 1
        for img_name in tqdm(os.listdir(folder_path), desc=f"Loading {dataset}/{category}"):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip unreadable files
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0  # Normalize pixel values
    y = np.array(y)
    return X, y

# Load train, val, and test data
X_train, y_train = load_data("train")
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

# Reshape to match ML model format (e.g., for HOG or traditional models)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE)
X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE)

print(f"Training Data: {X_train.shape}, Labels: {y_train.shape}")
print(f"Validation Data: {X_val.shape}, Labels: {y_val.shape}")
print(f"Testing Data: {X_test.shape}, Labels: {y_test.shape}")

# Save preprocessed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
