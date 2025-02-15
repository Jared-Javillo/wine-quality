```markdown
# **Exploring the Taylor Swift Spotify Dataset: Code Breakdown**

This notebook performs an exploratory data analysis (EDA) on a dataset containing Taylor Swift's songs available on Spotify. The steps include loading the data, visualizing correlations, analyzing album distributions, and examining track features.

## **1. Importing Libraries and Loading Data**
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
```
### **Explanation**
- `numpy` (`np`): Used for numerical operations.
- `pandas` (`pd`): Used to handle data in tabular form.
- `seaborn` (`sns`): Used for advanced statistical plotting.
- `matplotlib.pyplot` (`plt`): Used for creating plots.
- `OrdinalEncoder`: Prepares categorical features for machine learning.

```python
TS = pd.read_csv("taylor_swift_spotify.csv")
display(TS)
TS.info()
```
### **Explanation**
- Loads the dataset into a DataFrame (`TS`).
- `display(TS)`: Displays the DataFrame in Jupyter Notebook.
- `TS.info()`: Prints metadata, including column names, data types, and missing values.

---

## **2. Correlation Heatmap**
```python
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
corr = TS.drop(columns=["Unnamed: 0","track_number"]).corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.xticks(rotation=30)
plt.show()
```
### **Explanation**
- `plt.figure(figsize=(12,6))`: Sets figure size.
- `sns.set(style="whitegrid")`: Applies a clean background.
- `TS.drop(columns=["Unnamed: 0","track_number"])`: Removes unnecessary columns.
- `.corr()`: Computes correlation between numeric variables.
- `sns.heatmap()`: Visualizes correlations.
- `plt.xticks(rotation=30)`: Rotates x-axis labels.
- `plt.show()`: Displays the plot.

---

## **3. Album Distribution**
```python
TS['album'].value_counts()
```
### **Explanation**
- `TS['album'].value_counts()`: Counts the number of songs in each album.

---

## **4. Feature Analysis with Boxplots**
```python
# Collect all features having [0, 1] as the range of values
melted_TS = pd.melt(TS, id_vars="album", value_vars=TS.columns.values[np.r_[7:12, 13, 15]])
melted_TS.columns = ['Album', 'Features', 'Values']
display(melted_TS)
```
### **Explanation**
- `pd.melt()`: Converts wide-format data into long format.
- `id_vars="album"`: Keeps album names as identifiers.
- `value_vars=TS.columns.values[np.r_[7:12, 13, 15]]`: Selects specific columns (likely features ranging from 0 to 1).
- Renames columns for readability.
- `display(melted_TS)`: Displays transformed data.

```python
# Plot boxplots per album
plt.figure(figsize=(10,20))
sns.boxplot(data=melted_TS, width=0.8, orient="h",
            x="Values", y="Features", hue='Album')
plt.rc('axes', labelsize=15)
plt.legend(bbox_to_anchor=(1, -0.05))
plt.grid()
plt.show()
```
### **Explanation**
- Creates boxplots to analyze feature distributions per album.
- `sns.boxplot()`: Plots boxplots horizontally.
- `hue='Album'`: Groups data by album.
- `plt.legend(bbox_to_anchor=(1, -0.05))`: Moves the legend outside the plot.

---

## **5. Song Duration Analysis**
```python
plt.figure(figsize=(10,15))
TS["duration_min"] = TS["duration_ms"]/60000
sns.boxplot(data=TS, x="duration_min", y="album")
plt.grid()
plt.show()
TS.drop(columns="duration_min", inplace=True)
```
### **Explanation**
- Converts duration from milliseconds to minutes.
- Creates a boxplot showing song durations across albums.
- Removes the temporary `duration_min` column.
```