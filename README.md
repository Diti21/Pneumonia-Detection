# Pneumonia Detection from X-Ray Images

This project explores an efficient and interpretable approach for pneumonia detection from chest X-ray images, achieving an accuracy of **94.71%**.

---

## Overview

Instead of relying on deep learning methods—which often require heavy computational resources and can be difficult to interpret—we developed a traditional machine learning pipeline that offers several advantages:

- **Lightweight and efficient**, suitable for deployment on devices with limited hardware capabilities.
- **Highly interpretable**, enabling clear understanding and trust in the model’s predictions.
- **Designed for low-resource environments**, such as rural or underserved regions lacking access to expert radiologists or expensive GPUs.

---

## Tech Stack & Methodology

- **Feature Extraction:** Histogram of Oriented Gradients (HOG) to extract meaningful shape and texture features from X-ray images.
- **Classifier:** Support Vector Machine (SVM), a robust and effective classification algorithm.
- **Validation:** Stratified K-Fold Cross Validation to ensure balanced and reliable evaluation across all classes.

---

## Performance

- Achieved **94.71% accuracy** on the test dataset.
- Maintains a balanced precision and recall, ensuring reliability in detecting pneumonia while minimizing false positives.

---

## Additional Features

- Performed detailed **Exploratory Data Analysis (EDA)** to gain insights and address data biases.
- Built a **real-time prediction interface** for quick and user-friendly diagnosis.
- Applied **balanced class training** to handle dataset imbalance effectively.
- Ensured **model persistence** for consistent deployment across various platforms.

---

## Impact

This project demonstrates that traditional machine learning techniques, when carefully implemented, can provide practical, interpretable, and accessible solutions—especially valuable in healthcare settings where resources are limited.

---

## How to Use

1. Clone this repository.
2. Install the required dependencies (listed in `requirements.txt`).
3. Run the training and evaluation scripts.
4. Use the real-time prediction interface to test new X-ray images.

---

## Acknowledgments

Special thanks to all contributors and datasets used in this project.
