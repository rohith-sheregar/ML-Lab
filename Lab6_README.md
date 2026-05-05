# Lab 6 – Locally Weighted Regression (LWR)

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 1, Chapter 4

---

## Question

Implement the **non-parametric Locally Weighted Regression (LWR)** algorithm in order to fit data points. Select an appropriate dataset for your experiment and draw graphs.

---

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg

# Generate synthetic dataset: noisy sine wave
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

# Fit Locally Weighted Regression using Kernel Regression
model = KernelReg(y, X, 'c')   # 'c' = continuous variable

# Predict on a finer grid
x_test = np.linspace(0, 2 * np.pi, 200)
y_pred, _ = model.fit(x_test)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', edgecolor='black', s=60, alpha=0.7, label="Training Data")
plt.plot(x_test, y_pred, color='blue', linewidth=3, label="LWR Fit (tau=0.5)")
plt.title("Locally Weighted Regression", fontsize=16, fontweight='bold')
plt.xlabel("X")
plt.ylabel("y")
plt.legend(frameon=True)
plt.tight_layout()
plt.show()
```

---

## Expected Output

### Graph
A plot with:
- **Red scatter points:** 100 noisy training data points sampled from `y = sin(x) + noise`.
- **Blue curve:** The LWR fitted curve that closely follows the underlying sine wave, smoothly passing through the scattered data.

The LWR curve is noticeably smoother than a point-to-point interpolation, demonstrating the effect of the local kernel weighting.

---

## Key Concepts

- **Locally Weighted Regression (LWR):** A non-parametric regression method where, for each query point, a separate weighted linear regression is fitted using nearby training points. Points closer to the query point receive higher weights.
- **Kernel Function:** Determines how weight decreases with distance from the query point. Common choices are Gaussian and Epanechnikov kernels.
- **Bandwidth (τ / tau):** Controls the width of the kernel (the neighborhood size). Smaller bandwidth → more local fit (wiggly), larger bandwidth → smoother fit (global-like).
- **Non-parametric:** LWR does not assume a fixed global functional form for the data. Instead, the model is constructed locally at each prediction point.
- **`statsmodels.nonparametric.kernel_regression.KernelReg`:** Scikit-learn-compatible implementation supporting continuous (`'c'`) and unordered (`'u'`) variables.

### LWR vs. Standard Linear Regression
| Feature | Linear Regression | LWR |
|---|---|---|
| Model type | Global parametric | Local non-parametric |
| Fit | Single line/curve for all data | Separate fit per query point |
| Flexibility | Low | High |
| Computation | Fast | Slow at prediction time |
