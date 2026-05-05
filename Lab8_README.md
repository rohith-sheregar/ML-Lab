# Lab 8 – Decision Tree Classification

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 2, Chapter 3

---

## Question

Develop a program to demonstrate the working of the **Decision Tree algorithm**. Use the **Breast Cancer dataset** for building the decision tree and apply this knowledge to classify a new sample.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# Part 1: Decision Tree on Breast Cancer Dataset
# ─────────────────────────────────────────────
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

bc_model = DecisionTreeClassifier(max_depth=4, random_state=42)
bc_model.fit(X_cancer, y_cancer)
print("Model trained successfully.")

# ─────────────────────────────────────────────
# Part 2: Decision Tree on Custom Job Offer Dataset
# ─────────────────────────────────────────────
sample_data = pd.DataFrame({
    'cgpa':               [9.2, 8.5, 9.0, 7.5, 8.2, 9.1, 7.8, 9.3, 8.4, 8.6],
    'interactiveness':    ['yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes'],
    'practical_knowledge':['verygood', 'good', 'average', 'average', 'good', 'good', 'good', 'verygood', 'good', 'average'],
    'communication':      ['good', 'moderate', 'poor', 'good', 'moderate', 'moderate', 'poor', 'good', 'good', 'good'],
    'job_offer':          ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes']
})

# Encode categorical columns
label_encoders = {}
for column in ['interactiveness', 'practical_knowledge', 'communication', 'job_offer']:
    le = LabelEncoder()
    sample_data[column] = le.fit_transform(sample_data[column])
    label_encoders[column] = le

X_sample = sample_data.drop('job_offer', axis=1)
y_sample = sample_data['job_offer']

sample_model = DecisionTreeClassifier(max_depth=4, random_state=42)
sample_model.fit(X_sample, y_sample)

# --- Visualize Tree ---
plt.figure(figsize=(12, 6))
plot_tree(
    sample_model,
    feature_names=X_sample.columns,
    class_names=label_encoders['job_offer'].classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Job Offer Prediction")
plt.show()

# --- Classify a New Sample ---
test_sample = pd.DataFrame([{
    'cgpa': 6.5,
    'interactiveness': 'yes',
    'practical_knowledge': 'good',
    'communication': 'good'
}])

for column in ['interactiveness', 'practical_knowledge', 'communication']:
    test_sample[column] = label_encoders[column].transform(test_sample[column])

prediction = sample_model.predict(test_sample)
predicted_label = label_encoders['job_offer'].inverse_transform(prediction)

print("Predicted Job Offer for test sample:", predicted_label[0])
```

---

## Expected Output

```
Model trained successfully.
Predicted Job Offer for test sample: yes
```

### Decision Tree Visualization
A visual tree diagram showing:
- **Root node** (most discriminating feature, e.g., `cgpa` or `practical_knowledge`).
- **Internal nodes** with split conditions (e.g., `cgpa <= 7.85`).
- **Leaf nodes** color-coded by class (`yes`/`no` job offer), with class counts shown.

---

## Key Concepts

- **Decision Tree:** A supervised learning algorithm that recursively splits the feature space using the feature and threshold that best separates the classes at each node.
- **Splitting Criterion:** The default is Gini impurity. Alternatives include Information Gain (entropy). Both measure how homogeneous a split makes the resulting child nodes.
- **`max_depth`:** Limits the depth of the tree to prevent overfitting. A shallower tree generalizes better on unseen data.
- **LabelEncoder:** Converts categorical string labels to integer codes required by scikit-learn.
- **`plot_tree`:** Visualizes the learned tree structure, showing feature names, thresholds, Gini values, and class distributions at each node.

### Gini Impurity Formula
```
Gini(t) = 1 - Σ pᵢ²
```
Where pᵢ is the proportion of class i at node t. A Gini of 0 means a pure node (all one class).
