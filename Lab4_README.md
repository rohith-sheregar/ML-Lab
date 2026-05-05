# Lab 4 – Find-S Algorithm

**Course:** Machine Learning Lab (BCSL606) | Semester 6  
**Reference:** Book 1, Chapter 3

---

## Question

For a given set of training data examples stored in a `.CSV` file, implement and demonstrate the **Find-S algorithm** to output a description of the set of all hypotheses consistent with the training examples.

---

## Code

```python
import pandas as pd
import numpy as np

# Load training data from CSV
data = pd.read_csv('/content/enjoysport.csv')

attribute = np.array(data)[:, :-1]   # All columns except the last (features)
target = np.array(data)[:, -1]       # Last column (class label: yes/no)

print("Training data:\n")
print(data)

def train(att, tar):
    # Initialize hypothesis with the first positive example
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = att[i].copy()
            break

    # Generalize hypothesis for each positive example
    for i, val in enumerate(att):
        if tar[i] == 'yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'   # Generalize differing attribute

    return specific_h

print("\n")
print("Most Specific Hypothesis:", train(attribute, target))
```

### Sample CSV (`enjoysport.csv`)
```
Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,yes
Sunny,Warm,High,Strong,Warm,Same,yes
Rainy,Cold,High,Strong,Warm,Change,no
Sunny,Warm,High,Strong,Cool,Change,yes
```

---

## Expected Output

```
Training data:

      Sky AirTemp Humidity   Wind Water Forecast EnjoySport
0   Sunny    Warm   Normal  Strong  Warm     Same        yes
1   Sunny    Warm     High  Strong  Warm     Same        yes
2   Rainy    Cold     High  Strong  Warm   Change         no
3   Sunny    Warm     High  Strong  Cool   Change        yes

Most Specific Hypothesis: ['Sunny' 'Warm' '?' 'Strong' '?' '?']
```

---

## Key Concepts

- **Find-S Algorithm:** A supervised concept-learning algorithm that starts with the most specific hypothesis and generalizes it step-by-step, only considering **positive training examples**.
- **Hypothesis Representation:** An attribute vector where each position is either a specific value or `'?'` (meaning "any value is acceptable").
- **Generalization Rule:** If a positive example has an attribute that differs from the current hypothesis, that attribute is set to `'?'` to accept both values.
- **Limitation:** Find-S ignores negative examples entirely, so it cannot detect inconsistencies in training data. It assumes the data is noise-free and the target concept is in the hypothesis space.

### Algorithm Steps
1. Initialize hypothesis `h` to the most specific hypothesis (first positive example).
2. For each positive training example `x`:
   - For each attribute `a`:
     - If `h[a] == x[a]` → keep as is.
     - Else → set `h[a] = '?'` (generalize).
3. Output the final hypothesis `h`.
