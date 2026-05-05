# Lab 5 – k-Nearest Neighbour (kNN) Classification

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 2, Chapter 2

---

## Question

Develop a program to implement the **k-Nearest Neighbour (kNN) algorithm** to classify randomly generated 100 values of x in the range [0, 1]. Perform the following:

a. Label the first 50 points {x₁, ..., x₅₀} as follows: if xᵢ ≤ 0.5, then xᵢ ∈ Class1, else xᵢ ∈ Class2.  
b. Classify the remaining points x₅₁, ..., x₁₀₀ using kNN. Perform this for **k = 1, 2, 3, 4, 5, 20, 30**.

---

## Code

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Generate 100 random values in [0, 1]
data = np.random.rand(100).reshape(-1, 1)

# Label first 50 points based on threshold 0.5
labels = np.array(["Class1" if x <= 0.5 else "Class2" for x in data[:50].flatten()])

train_data = data[:50]
train_labels = labels
test_data = data[50:]

k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    predictions = knn.predict(test_data)

    print(f"Results for k={k}:")
    print(predictions)
    print()
```

---

## Expected Output

The output will vary each run (due to `np.random.rand`), but follows this general format:

```
Results for k=1:
['Class2' 'Class1' 'Class2' 'Class1' 'Class2' ... ]

Results for k=2:
['Class2' 'Class1' 'Class2' 'Class1' 'Class2' ... ]

Results for k=3:
['Class2' 'Class1' 'Class2' 'Class1' 'Class2' ... ]

Results for k=4:
['Class2' 'Class1' 'Class2' 'Class1' 'Class2' ... ]

Results for k=5:
['Class2' 'Class1' 'Class2' 'Class1' 'Class2' ... ]

Results for k=20:
['Class2' 'Class1' 'Class2' 'Class1' 'Class2' ... ]

Results for k=30:
['Class2' 'Class1' 'Class1' 'Class1' 'Class2' ... ]
```

### Observations
- For small k (e.g., k=1), the classifier is sensitive to individual training points — high variance, low bias.
- For large k (e.g., k=20, k=30), predictions become smoother (more Class1 or Class2 domination) — low variance, high bias.
- Since the labeling rule is simply `x ≤ 0.5`, kNN with any k should closely approximate the true boundary at 0.5.

---

## Key Concepts

- **kNN Algorithm:** A non-parametric, instance-based learning method. To classify a new point, it finds the k nearest training points (by Euclidean distance) and assigns the majority class.
- **k (number of neighbors):** Controls the bias-variance trade-off.
  - Small k → complex decision boundary, overfitting risk.
  - Large k → smooth decision boundary, underfitting risk.
- **Decision Boundary:** For this 1D problem, the natural boundary is at x = 0.5. kNN approximates this from the training data.
- **sklearn.neighbors.KNeighborsClassifier:** Scikit-learn's implementation supports multiple distance metrics and weighting schemes.
