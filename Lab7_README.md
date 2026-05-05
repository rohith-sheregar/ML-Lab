# Lab 7 – Linear Regression and Polynomial Regression

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 1, Chapter 5  
**Note:** This lab is stored as `Lab10.ipynb` in the repository.

---

## Question

Develop a program to demonstrate the working of:
- **Linear Regression** using the **Boston Housing Dataset** (predict house prices from number of rooms).
- **Polynomial Regression** using the **Auto MPG Dataset** (predict vehicle fuel efficiency from displacement).

---

## Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ─────────────────────────────────────────────
# Part 1: Linear Regression – Boston Housing
# ─────────────────────────────────────────────
def linear_regression_boston():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)

    X = df[["rm"]].values       # Average number of rooms
    y = df["medv"].values        # Median home value ($1000s)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Average number of rooms (RM)")
    plt.ylabel("Median value of homes ($1000)")
    plt.title("Linear Regression - Boston Housing Dataset")
    plt.legend()
    plt.show()

    print("Linear Regression - Boston Housing Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))


# ─────────────────────────────────────────────
# Part 2: Polynomial Regression – Auto MPG
# ─────────────────────────────────────────────
def polynomial_regression_auto_mpg():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = [
        "mpg", "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin"
    ]

    data = pd.read_csv(url, sep=r'\s+', names=column_names, na_values="?")
    data = data.dropna()

    X = data["displacement"].values.reshape(-1, 1)
    y = data["mpg"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    poly_model = make_pipeline(
        PolynomialFeatures(degree=2),
        StandardScaler(),
        LinearRegression()
    )
    poly_model.fit(X_train, y_train)
    y_pred = poly_model.predict(X_test)

    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Displacement")
    plt.ylabel("Miles per gallon (mpg)")
    plt.title("Polynomial Regression - Auto MPG Dataset")
    plt.legend()
    plt.show()

    print("Polynomial Regression - Auto MPG Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))


if __name__ == "__main__":
    print("Demonstrating Linear Regression and Polynomial Regression\n")
    linear_regression_boston()
    polynomial_regression_auto_mpg()
```

---

## Expected Output

### Part 1 – Linear Regression (Boston Housing)

**Plot:** A scatter of actual test values (blue) with the fitted regression line (red) showing the positive relationship between number of rooms (`RM`) and home value (`MEDV`).

```
Linear Regression - Boston Housing Dataset
Mean Squared Error: ~43.6
R² Score: ~0.48
```

### Part 2 – Polynomial Regression (Auto MPG)

**Plot:** Scatter of actual vs. predicted MPG values across displacement. The polynomial curve captures the non-linear (decreasing) relationship between engine displacement and fuel efficiency better than a straight line would.

```
Polynomial Regression - Auto MPG Dataset
Mean Squared Error: ~17.2
R² Score: ~0.68
```

---

## Key Concepts

- **Linear Regression:** Models the target as a linear function of the input feature: `y = w₀ + w₁x`. Optimal weights are found by minimizing Mean Squared Error (MSE).
- **Polynomial Regression:** Extends linear regression by adding polynomial features (e.g., x², x³). The model is still linear in the parameters, but captures non-linear relationships in the data.
- **Pipeline:** `make_pipeline(PolynomialFeatures, StandardScaler, LinearRegression)` chains preprocessing and model steps for cleaner code.
- **MSE (Mean Squared Error):** Average of squared differences between predicted and actual values. Lower is better.
- **R² Score:** Proportion of variance in the target explained by the model. Ranges from 0 to 1; higher is better.
