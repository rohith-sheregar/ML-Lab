# Lab 1 – Histograms and Box Plots on California Housing Dataset

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 1, Chapter 2

---

## Question

Develop a program to create histograms for all numerical features and analyze the distribution of each feature. Generate box plots for all numerical features and identify any outliers. Use the **California Housing dataset**.

---

## Code

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing(as_frame=True)
housing_df = data.frame

numerical_features = housing_df.select_dtypes(include=[np.number]).columns

# --- Histograms ---
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(housing_df[feature], kde=False, bins=30, color='blue')
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# --- Box Plots ---
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=housing_df[feature], color='orange')
    plt.title(f'Box Plot of {feature}')

plt.tight_layout()
plt.show()

# --- Outlier Detection using IQR ---
print("Outliers Detection:")
outliers_summary = {}

for feature in numerical_features:
    Q1 = housing_df[feature].quantile(0.25)
    Q3 = housing_df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = housing_df[
        (housing_df[feature] < lower_bound) |
        (housing_df[feature] > upper_bound)
    ]

    outliers_summary[feature] = len(outliers)
    print(f"{feature}: {len(outliers)} outliers")

# Dataset summary
print("\nDataset Summary:")
print(housing_df.describe())
```

---

## Expected Output

### Histograms
A 3×3 grid of histogram plots, one per numerical feature (e.g., `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`, `MedHouseVal`). Each histogram shows the frequency distribution with 30 bins.

### Box Plots
A 3×3 grid of box plots showing the spread, median, IQR, and whiskers for each feature. Features like `AveOccup` and `Population` are likely to show many outlier points beyond the whiskers.

### Outlier Count (sample output)
```
Outliers Detection:
MedInc: 681 outliers
HouseAge: 0 outliers
AveRooms: 549 outliers
AveBedrms: 703 outliers
Population: 577 outliers
AveOccup: 677 outliers
Latitude: 0 outliers
Longitude: 0 outliers
MedHouseVal: 965 outliers
```

### Dataset Summary
A statistical summary table (count, mean, std, min, 25%, 50%, 75%, max) for all numerical features.

---

## Key Concepts

- **Histogram:** Shows the frequency distribution of a feature. A right-skewed histogram (long tail to the right) indicates most values are low with some very large values.
- **Box Plot:** Summarizes data using the five-number summary. Points beyond 1.5×IQR from Q1 or Q3 are considered outliers.
- **IQR (Interquartile Range):** Q3 − Q1. Used to detect outliers via the rule: value < Q1 − 1.5×IQR or value > Q3 + 1.5×IQR.
