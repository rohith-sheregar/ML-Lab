# Lab 3 – Principal Component Analysis (PCA) on Iris Dataset

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 1, Chapter 2

---

## Question

Develop a program to implement **Principal Component Analysis (PCA)** for reducing the dimensionality of the **Iris dataset** from 4 features to 2.

---

## Code

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
data = iris.data
target = iris.target
label_names = iris.target_names

iris_df = pd.DataFrame(data, columns=iris.feature_names)

# Apply PCA: 4 features → 2 principal components
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
reduced_df['target'] = target

# --- Plot PCA result ---
plt.figure(figsize=(10, 15))
colors = ['r', 'g', 'b']

for i, t in enumerate(np.unique(target)):
    plt.scatter(
        reduced_df[reduced_df['target'] == t]['PC1'],
        reduced_df[reduced_df['target'] == t]['PC2'],
        c=colors[i],
        label=label_names[i]
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Iris Dataset')
    plt.legend()
    plt.show()
```

---

## Expected Output

### Scatter Plot (PC1 vs PC2)
A 2D scatter plot with three clusters of points (one per Iris class):
- **Setosa (red):** Clearly separated from the other two classes.
- **Versicolor (green):** Partially overlapping with Virginica.
- **Virginica (blue):** Partially overlapping with Versicolor.

The plot demonstrates that PCA successfully captures most of the variance in just 2 components, enabling visual separation of the three species.

### Explained Variance (if printed)
```
Explained variance ratio: [0.7296, 0.2285]
Total variance explained: ~95.8%
```

---

## Key Concepts

- **PCA (Principal Component Analysis):** An unsupervised linear dimensionality reduction technique. It finds the directions (principal components) of maximum variance in the data and projects the data onto a lower-dimensional subspace.
- **PC1 (First Principal Component):** The direction of the greatest variance in the dataset.
- **PC2 (Second Principal Component):** The direction of the second greatest variance, orthogonal to PC1.
- **Explained Variance Ratio:** The proportion of the total variance captured by each principal component. Together, PC1 and PC2 of the Iris dataset explain ~95.8% of the variance.
- **Use Case:** Dimensionality reduction for visualization, noise reduction, and speeding up downstream ML algorithms.
