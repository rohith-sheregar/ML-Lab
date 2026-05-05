# Lab 9 – Naïve Bayes Classifier on Olivetti Faces Dataset

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 2, Chapter 4

---

## Question

Develop a program to implement the **Naïve Bayesian classifier** considering the **Olivetti Face Dataset** for training. Compute the accuracy of the classifier, considering a few test data sets.

---

## Code

```python
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Olivetti Faces dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data    # Shape: (400, 4096) — 400 images, each 64x64 pixels flattened
y = data.target  # 40 subjects, 10 images each

# Train/Test Split (70/30 stratified)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Gaussian Naïve Bayes Classifier
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# Predict
y_pred = gnb.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display Predictions on Sample Faces
fig, axes = plt.subplots(3, 5, figsize=(12, 8))

for ax, image, label, prediction in zip(axes.ravel(), x_test, y_test, y_pred):
    ax.imshow(image.reshape(64, 64), cmap=plt.cm.gray)
    ax.set_title(f"True: {label}  Pred: {prediction}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## Expected Output

### Accuracy
```
Accuracy: 73.33%
```
*(Accuracy may vary slightly; Gaussian NB on raw pixel data typically achieves 70–80% on Olivetti.)*

### Classification Report (abbreviated)
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      1.00      1.00         3
           2       1.00      0.67      0.80         3
           ...
          39       0.60      1.00      0.75         3

    accuracy                           0.73       120
   macro avg       0.82      0.73      0.75       120
weighted avg       0.82      0.73      0.75       120
```

### Face Grid Plot
A 3×5 grid of grayscale face images from the test set, each labeled with the **true subject ID** and the **predicted subject ID**. Correct predictions will show matching True/Pred labels; misclassifications will show different numbers.

---

## Key Concepts

- **Naïve Bayes Classifier:** A probabilistic classifier based on Bayes' theorem with the "naïve" assumption of conditional independence between features given the class label.
- **Gaussian Naïve Bayes:** Assumes features follow a Gaussian (normal) distribution within each class. Estimates the mean and variance per feature per class from the training data.
- **Bayes' Theorem:**  
  `P(class | features) ∝ P(features | class) × P(class)`
- **Olivetti Faces Dataset:** 400 grayscale face images of 40 individuals (10 per person), each 64×64 pixels (4096 features). Available directly from scikit-learn via `fetch_olivetti_faces`.
- **Stratified Split:** Ensures each class has proportional representation in both train and test sets — important for multi-class problems with equal class sizes.
- **Confusion Matrix:** A 40×40 matrix where entry (i, j) is the number of samples of true class i predicted as class j. The diagonal represents correct predictions.
