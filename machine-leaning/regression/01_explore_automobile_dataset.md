# Exploring the Automobile MPG Dataset

This guide walks through preparing, exploring, and visualizing the automobile dataset for regression analysis.

## 1. Installing Python Libraries

To install required libraries in Jupyter Notebook, use pip commands with an exclamation mark:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```
## 2. Setting Up Imports

Import the necessary libraries for data manipulation and visualization:

- `pandas` and `numpy` for data handling
- `matplotlib` and `seaborn` for visualization
- `sklearn` for machine learning
- `datetime` for date calculations

## 3. Working with the Automobile Dataset

The dataset is available on Kaggle and contains features for predicting miles per gallon (mpg).

- Load the dataset using `pd.read_csv`.
- View a random sample of records with `df.sample(5)`.

## 4. Dataset Features

Features include:
- cylinders
- displacement
- horsepower
- weight
- acceleration
- model year
- origin
- car name

The target variable is `mpg`.

## 5. Data Preprocessing

- Replace question marks with NaN using `automobile_df.replace`.
- Drop records with missing fields using `dropna`.
- Remove non-predictive features like `origin` and `car name`.

## 6. Feature Engineering

- Convert `model year` to a full year format (e.g., 1973, 1980).
- Calculate the age of the car by subtracting the model year from the current year.
- Drop the original `model year` field after creating the age column.

## 7. Data Types and Conversion

- Ensure all inputs to the machine learning model are numeric.
- Convert the `horsepower` column to numeric using `pd.to_numeric`.

## 8. Descriptive Statistics

Use `describe` to get statistical information about numerical features, including mean, standard deviation, and percentiles.

---

## Visualizing Relationships and Correlations in Features

### Scatter Plot Analysis

1. **Age vs. Miles Per Gallon (MPG):**
   - Older cars tend to have lower mileage, indicating a downward trend.
   - This trend suggests a potential relationship but requires further statistical analysis.
2. **Acceleration vs. MPG:**
   - Upward slope; higher acceleration might be associated with higher mileage.
3. **Weight vs. MPG:**
   - Heavier cars tend to have lower mileage.
4. **Displacement vs. MPG:**
   - Greater displacement correlates with lower mileage.
5. **Horsepower vs. MPG:**
   - Horsepower affects mileage, with a noticeable trend.
6. **Cylinders vs. MPG:**
   - Cars with four cylinders generally have better mileage.

### Correlation Analysis

- Use the `corr` function to list pairwise correlations.
- Correlation values range between -1 and 1.
  - Positive: Variables move in the same direction.
  - Negative: Variables move in opposite directions.
- Key findings:
  - Acceleration is positively correlated with MPG.
  - Weight is highly negatively correlated with MPG (-0.83).

### Heatmap Visualization

- Use Seaborn's heatmap with `annot=True` to display correlation values.
- Lighter colors indicate positive correlation; darker colors indicate negative.
- Example: MPG is very negatively correlated with the age of the car (-0.58).

### Data Preprocessing and Shuffling

- Shuffle the dataset using `sample(frac=1)` and reset indices.
- Drop original index values using `drop=True`.
- Shuffling prevents the model from picking up non-existent patterns due to data order.

### Saving the Processed Dataset

- Save the shuffled and cleaned dataset to a new CSV file (`auto-mpg-processed.csv`) using `to_csv`.
- Confirm the file's existence using the `ls` command.

---

## Example: Loading, Exploring, and Visualizing the Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/auto-mpg.csv')

# Replace missing values
df = df.replace('?', np.nan)
df = df.dropna()

# Drop non-predictive columns
df.drop(columns=['car name', 'origin'], inplace=True)

# Convert model year and create age feature
import datetime
df['model year'] = '19' + df['model year'].astype(str)
df['age'] = datetime.datetime.now().year - df['model year'].astype(int)
df.drop(columns=['model year'], inplace=True)

# Convert horsepower to numeric
df['horsepower'] = df['horsepower'].astype(float)

# View sample and describe
print(df.sample(5))
print(df.describe())

# Visualizations
plt.scatter(df['age'], df['mpg'])
plt.xlabel('Age of the car (years)')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Age vs. MPG')
plt.show()

plt.scatter(df['acceleration'], df['mpg'])
plt.xlabel('Acceleration')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Acceleration vs. MPG')
plt.show()

plt.scatter(df['weight'], df['mpg'])
plt.xlabel('Weight')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Weight vs. MPG')
plt.show()

plt.scatter(df['horsepower'], df['mpg'])
plt.xlabel('Horsepower')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Horsepower vs. MPG')
plt.show()

# Correlation analysis
corr_matrix = df.corr()
print(corr_matrix)

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Shuffle and save processed dataset
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/auto-mpg-processed.csv', index=False)
```

---

## Next Steps

- Use the processed CSV file to build regression models.
