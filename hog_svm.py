import os
import cv2
import numpy as np
import joblib
import random
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# Paths and labels
data_dir = "chest_xray"
categories = ["NORMAL", "PNEUMONIA"]
X = {"train": [], "val": []}
y = {"train": [], "val": []}

# Faster augmentation (light)
def augment_image(img):
    if random.random() > 0.7:
        img = cv2.flip(img, 1)
    return img

# Resize for speed
resize_dim = 64  # smaller image, faster processing

# Load and process
for mode in ["train", "val"]:
    for category in categories:
        folder = os.path.join(data_dir, mode, category)
        label = 0 if category == "NORMAL" else 1
        for i, fname in enumerate(os.listdir(folder)):
            if i >= 800 and mode == "train":  # Optional: limit training images for speed
                break
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (resize_dim, resize_dim))
            if mode == "train":
                img = augment_image(img)
            # Simplified HOG for speed
            features = hog(img, orientations=6, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm="L2-Hys",
                           visualize=False)
            X[mode].append(features)
            y[mode].append(label)

# Convert and shuffle
X_train, y_train = shuffle(np.array(X["train"]), np.array(y["train"]), random_state=42)
X_val, y_val = np.array(X["val"]), np.array(y["val"])

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train SVM (faster kernel)
svm_model = SVC(kernel='linear', C=0.3, class_weight='balanced', probability=False, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = svm_model.predict(X_train_scaled)
y_val_pred = svm_model.predict(X_val_scaled)
import os
import cv2
import numpy as np
import joblib
import random
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Paths and labels
data_dir = "chest_xray"
categories = ["NORMAL", "PNEUMONIA"]
X = {"train": [], "val": []}
y = {"train": [], "val": []}

# Faster augmentation (light)
def augment_image(img):
    if random.random() > 0.7:
        img = cv2.flip(img, 1)
    return img

# Resize for speed
resize_dim = 64  # smaller image, faster processing

# Load and process
for mode in ["train", "val"]:
    for category in categories:
        folder = os.path.join(data_dir, mode, category)
        label = 0 if category == "NORMAL" else 1
        for i, fname in enumerate(os.listdir(folder)):
            if i >= 800 and mode == "train":  # Optional: limit training images for speed
                break
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (resize_dim, resize_dim))
            if mode == "train":
                img = augment_image(img)
            # Simplified HOG for speed
            features = hog(img, orientations=6, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm="L2-Hys",
                           visualize=False)
            X[mode].append(features)
            y[mode].append(label)

# Convert and shuffle
X_train, y_train = shuffle(np.array(X["train"]), np.array(y["train"]), random_state=42)
X_val, y_val = np.array(X["val"]), np.array(y["val"])

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced', random_state=42))
])

# Define hyperparameters
param_grid = {
    'svm__kernel': ['linear', 'rbf'],
    'svm__C': [0.1, 0.3, 1],
    'svm__gamma': ['scale', 'auto']
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)

# Results
# print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred) * 100:.2f}%")
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))

# Save
joblib.dump(best_model, "best_svm_model.pkl")
print("\nBest model saved.")
# Results
# print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred) * 100:.2f}%")
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))

# Fast cross-validation using parallel jobs
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"\nCross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")

# Save
joblib.dump(svm_model, "fast_svm_model.pkl")
joblib.dump(scaler, "fast_scaler.pkl")
print("\nFast model and scaler saved.")
