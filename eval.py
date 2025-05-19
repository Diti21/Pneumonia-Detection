import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load model
svm_model = joblib.load("svm_hog_model_augmented.pkl")

# Map folder names to labels
labels_map = {"NORMAL": 0, "PNEUMONIA": 1}

# Paths
test_folder = "chest_xray/chest_xray/test"  # Change to the correct test folder path
correct = 0
incorrect = 0

# Iterate through the labels (NORMAL, PNEUMONIA)
for label_name in os.listdir(test_folder):
    label_folder = os.path.join(test_folder, label_name)
    
    # Skip non-directory files (e.g., .DS_Store)
    if not os.path.isdir(label_folder):
        continue
    
    # Process each image file in the label folder
    for file in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file)
        
        # Read image in grayscale
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipped unreadable file: {file_path}")
            continue

        # Resize the image to 128x128
        img = cv2.resize(img, (128, 128))
        
        # Extract HOG features
        features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, visualize=True)
        features = np.array(features).reshape(1, -1)

        # Predict using the SVM model
        prediction = svm_model.predict(features)
        predicted_label = prediction[0]
        true_label = labels_map[label_name]

        # Update correct/incorrect count
        if predicted_label == true_label:
            correct += 1
        else:
            incorrect += 1

# Print the result
print(f"\nPredictions Summary:")
print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {incorrect}")
print(f"Total Predictions: {correct + incorrect}")
