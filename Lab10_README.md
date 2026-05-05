# Lab 10 – k-Means Clustering on Wisconsin Breast Cancer Dataset

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 2, Chapter 4

---

## Question

Develop a program to implement **k-Means Clustering** using the **Wisconsin Breast Cancer dataset** and visualize the clustering result.

---

## Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Load Wisconsin Breast Cancer Dataset
data = load_breast_cancer()
X = data.data
y = data.target   # 0 = malignant, 1 = benign

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply k-Means with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Reduce to 2D with PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Build DataFrame for plotting
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = y

# --- Plot 1: K-Means Clusters ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster',
                palette='Set1', s=100, edgecolor='black', alpha=0.7)
plt.title('K-Means Clustering of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# --- Plot 2: K-Means Clusters with Centroids ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster',
                palette='Set1', s=100, edgecolor='black', alpha=0.7)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red',
            marker='X', label='Centroids')
plt.title('K-Means Clustering with Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()
```

---

## Expected Output

### Plot 1 – Cluster Visualization
A 2D scatter plot (PCA-reduced) using seaborn showing:
- **Two colored clusters** (via `palette='Set1'`) representing the two k-Means groups.
- Points are color-coded by their assigned cluster label (0 or 1).
- The two clusters are visually separable in PCA space, reflecting the natural structure of the data.

### Plot 2 – Clusters with Centroids
Same scatter plot as above, with:
- **Red 'X' markers** indicating the two cluster centroids projected into PCA space.

---

## Key Concepts

- **k-Means Clustering:** An unsupervised learning algorithm that partitions n data points into k clusters. Each point belongs to the cluster with the nearest centroid.
- **Algorithm Steps:**
  1. Randomly initialize k cluster centroids.
  2. Assign each point to the nearest centroid (Euclidean distance).
  3. Recompute centroids as the mean of assigned points.
  4. Repeat steps 2–3 until convergence (centroids stop moving).
- **k=2:** Chosen because the dataset has 2 true classes (malignant and benign).
- **StandardScaler:** Essential before k-Means — the algorithm is distance-based, so features with larger scales would dominate otherwise.
- **`fit_predict()`:** Fits the k-Means model and returns the cluster label for each sample in one step.
- **PCA for Visualization:** Reduces the 30-dimensional feature space to 2D so the clusters can be plotted and inspected visually.
- **`pca.transform(kmeans.cluster_centers_)`:** Projects the high-dimensional cluster centroids into the 2D PCA space for overlay on the plot.
- **Wisconsin Breast Cancer Dataset:** 569 samples, 30 numerical features computed from digitized images of breast mass fine needle aspirate (FNA). Available via `sklearn.datasets.load_breast_cancer`.
- **Seaborn `scatterplot`:** Used with `hue='Cluster'` to automatically color-code points by cluster assignment.

### k-Means vs True Labels
k-Means cluster indices (0 and 1) do not necessarily align with the true class labels (0 = malignant, 1 = benign). The cluster with more points typically corresponds to the benign class (357 samples), but the mapping must be verified manually.
