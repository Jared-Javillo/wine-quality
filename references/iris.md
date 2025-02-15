# Exploratory Data Analysis (EDA) on the Iris Dataset

This document provides a step-by-step explanation of the code used for **Exploratory Data Analysis (EDA)** on the famous **Iris dataset** using `pandas`, `seaborn`, and `matplotlib`.

---

## 1. Importing Libraries and Loading the Dataset

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)  # Load dataset as a Pandas DataFrame
iris = pd.concat([iris.data, iris.target], axis=1)  # Combine features and target
iris.rename(columns={"target": "Species"}, inplace=True)

m = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
iris['Species'] = iris['Species'].map(m)  # Convert numeric labels to species names

display(iris)
```

### Explanation:
- The **Iris dataset** is loaded using `sklearn.datasets.load_iris()`.
- It contains **4 features**: `sepal length`, `sepal width`, `petal length`, `petal width`.
- The **target column (`Species`)** originally contains numbers (`0, 1, 2`) representing three flower species.
- The dictionary `{0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}` maps numeric labels to species names.
- The dataset is displayed.

---

## 2. Pairplot Visualization

```python
plt.figure(figsize=(10,10))
sns.pairplot(iris, hue='Species', height=1.5)
plt.show()
```

### Explanation:
- `sns.pairplot()` creates scatter plots for each pair of numerical features.
- The `hue='Species'` argument colors the points by species.
- This helps **visualize how different species are separated based on features**.

---

## 3. Correlation Matrix

```python
corr_matrix = iris.iloc[:, :4].corr()  # Compute correlation matrix for numeric columns
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.xticks(rotation=45)
plt.show()
```

### Explanation:
- **Computes correlations** between numerical features.
- Uses **`sns.heatmap()`** to visualize the correlation matrix.
- Darker/lighter blue shades indicate **strong/weak correlations**.

---

## 4. Boxplots & Violin Plots

```python
plt.figure(figsize=(8,8))
angle=30

# Matplotlib Boxplot
plt.subplot(221)
plt.boxplot(iris.iloc[:, :4].to_numpy(),
            boxprops=dict(linewidth=2, color='b'),
            medianprops=dict(linewidth=2, color='r'),
            whiskerprops=dict(linewidth=2, color='k'),
            capprops=dict(linewidth=2, color='k'),
            flierprops=dict(marker='+', markeredgecolor='r'))
plt.ylabel('Measurement (cm)')
plt.xticks(np.arange(4)+1, labels=iris.columns.values[:4], rotation=angle)
plt.grid()

# Matplotlib Violin Plot
plt.subplot(222)
plt.violinplot(iris.iloc[:, :4].to_numpy(), showmeans=False, showmedians=True)
plt.xticks(np.arange(4)+1, labels=iris.columns.values[:4], rotation=angle)
plt.grid()

# Seaborn Boxplot
plt.subplot(223)
sns.boxplot(data=iris, width=0.3)
plt.xticks(rotation=angle)
plt.ylabel('Measurement (cm)')
plt.grid()

# Seaborn Violin Plot
plt.subplot(224)
sns.violinplot(data=iris)
plt.xticks(rotation=angle)
plt.grid()

plt.tight_layout()
plt.show()
```

### Explanation:
- **Boxplot** shows **median, quartiles, and outliers**.
- **Violin plot** shows the **distribution of data**.
- Uses **Matplotlib and Seaborn** to compare different visualization methods.

---

## 5. Boxplot for Different Species

```python
melted_iris = pd.melt(iris, id_vars=["Species"], value_vars=iris.columns.values[:4])
melted_iris.columns = ['Species', 'Features', 'Measurements']
display(melted_iris)

plt.figure(figsize=(8,4))
sns.boxplot(data=melted_iris, width=0.8, x="Features", y="Measurements", hue='Species')
plt.rc('axes', labelsize=15)
plt.grid()
plt.show()
```

### Explanation:
- **Melts the DataFrame** to make it suitable for Seaborn plotting.
- Creates a **boxplot comparing measurements** for different species.
- **Hue** differentiates species.

---

## Summary

This notebook explores the **Iris dataset** using:
1. **Basic data preprocessing** (renaming, mapping labels).
2. **Pairplots** to visualize feature relationships.
3. **Correlation heatmap** to check feature dependencies.
4. **Boxplots & violin plots** to analyze feature distributions.
5. **Species-based boxplots** to understand class separations.

Would you like a deeper dive into any specific part? 😊

