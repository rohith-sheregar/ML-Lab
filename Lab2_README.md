# Lab 2 – Correlation Matrix, Heatmap, and Pair Plot on California Housing Dataset

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 1, Chapter 2

---

## Question

Develop a program to:
- Compute the correlation matrix to understand the relationships between pairs of features.
- Visualize the correlation matrix using a heatmap to identify strong positive/negative correlations.
- Create a pair plot to visualize pairwise relationships between features.

Use the **California Housing dataset**.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame
data = california_housing.frame

# --- Correlation Matrix ---
correlation_matrix = data.corr()

# --- Heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='BuPu',
    fmt='.2f',
    linewidths=1.5
)
plt.title('Correlation Matrix of California Housing Features')
plt.show()

# --- Pair Plot ---
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot of California Housing Features', y=1.02)
plt.show()
```

---

## Expected Output

### Heatmap
A 9×9 annotated heatmap where each cell shows the Pearson correlation coefficient between two features. Key observations:
- `MedInc` (Median Income) has the **strongest positive correlation** with `MedHouseVal` (~0.69).
- `Latitude` and `Longitude` show a moderate negative correlation (~-0.92), reflecting geographic alignment.
- `AveBedrms` and `AveRooms` are positively correlated.

### Pair Plot
A grid of scatter plots for every pair of features, with KDE (Kernel Density Estimate) on the diagonal to show each feature's distribution. The scatter plots reveal:
- A positive linear trend between `MedInc` and `MedHouseVal`.
- Geographic clustering visible in `Latitude` vs `Longitude`.

---

## Key Concepts

- **Correlation Matrix:** A square matrix where entry (i, j) is the Pearson correlation coefficient between feature i and feature j. Values range from −1 (perfect negative correlation) to +1 (perfect positive correlation); 0 indicates no linear relationship.
- **Heatmap:** A color-coded grid visualization of the correlation matrix. Darker or more saturated colors indicate stronger correlation.
- **Pair Plot:** A matrix of scatter plots (one per feature pair) with distribution plots along the diagonal. Useful for spotting linear relationships, clusters, and outliers across all feature combinations at once.
