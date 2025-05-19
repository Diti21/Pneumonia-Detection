import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog
import os
import sys

# Load the trained SVM model
svm_model = joblib.load("svm_hog_model_augmented.pkl")

def predict_xray(image_path, output_dir="outputs"):
    """
    Predicts whether an X-ray image shows pneumonia or is normal.
    Generates a bar chart of confidence and a side-by-side image display.
    Returns: prediction_label, output_image_path
    """

    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image.")
        return

    img_resized = cv2.resize(img, (128, 128))

    # Extract HOG features
    features, _ = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      orientations=9, visualize=True)
    features = features.reshape(1, -1)

    # Predict
    prediction = svm_model.predict(features)[0]

    try:
        scores = svm_model.decision_function(features)
        pos_score = scores[0]
        confidence = 1 / (1 + np.exp(-pos_score))
        probs = [1 - confidence, confidence]  # [Normal, Pneumonia]
    except:
        probs = [0.5, 0.5]

    label = "Pneumonia" if prediction == 1 else "Normal"
    labels = ["Normal", "Pneumonia"]

    # Prepare output folder
    os.makedirs(output_dir, exist_ok=True)
    output_img_path = os.path.join(output_dir, "result_visual.png")

    # Display original image and bar chart side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img_resized, cmap='gray')
    ax1.set_title("Input X-ray")
    ax1.axis('off')

    bars = ax2.bar(labels, probs, color=['#03A9F4', '#E91E63'])
    ax2.set_ylim(0, 1)
    ax2.set_title("Prediction Confidence")
    ax2.set_ylabel("Confidence Score")

    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width()/2.0, prob + 0.02, f"{prob:.2f}", ha='center', color='white')

    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()

    return label, output_img_path

# CLI support (for Express server)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No image path provided.")
    else:
        image_path = sys.argv[1]
        label, vis_path = predict_xray(image_path)
        print(f"{label}|{vis_path}")
