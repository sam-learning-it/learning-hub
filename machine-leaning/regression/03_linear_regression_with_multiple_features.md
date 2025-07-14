# Linear Regression with Multiple Features

## Understanding Multiple Regression: Concepts, Risks, and Mitigation

### What is Multiple Regression?

**Multiple regression** is an extension of simple linear regression that models the relationship between a dependent variable and two or more independent variables. It operates in multiple dimensions, allowing for a more nuanced understanding of complex data sets.

---

### Risks of Multiple Regression

__Multicollinearity__

One of the key risks in multiple regression is **multicollinearity**, which occurs when two or more independent (X) variables contain overlapping or redundant information. This can distort the statistical significance of the model and lead to unreliable estimates.

---

__Diagnosing Multicollinearity__

Multicollinearity can be identified when **explanatory variables are highly correlated**. For example, if variables X₁ and Xₖ tend to move together or show a strong linear relationship, it may indicate multicollinearity.

---

__Impact of Multicollinearity__

- **Reduces the model's explanatory power**: Multicollinearity makes it difficult to determine the individual effect of each variable on the target.
- **Leads to unstable coefficients**: The estimated regression coefficients may fluctuate significantly with small changes in the data.
- **Hinders generalization**: The model may perform well on training data but poorly on new, unseen data (out-of-sample performance).

---

### Mitigation Strategies

__1. Prevention Through Data Understanding__

- Apply domain knowledge to evaluate the relevance and uniqueness of variables.
- Remove or combine **closely related independent variables** that convey similar information.
- Ensure that selected variables provide **distinct and meaningful insights**.

__2. Standardization__

- Standardizing the data involves subtracting the mean and dividing by the standard deviation.
- This transformation expresses values in terms of **standard deviations from the mean**, making it easier to compare variable impacts.

__3. Use Adjusted R-squared__

- Instead of relying on the traditional R², use **adjusted R²** which accounts for the number of predictors in the model.
- Adjusted R² provides a more accurate assessment of model fit by penalizing the inclusion of unnecessary variables.

__4. Dimensionality Reduction Techniques__

- Use statistical techniques like **Factor Analysis** or **Principal Component Analysis (PCA)** to reduce the number of variables while retaining most of the important information.
- These methods help in simplifying the model and reducing noise in the data.

---

## 1. Feature Selection
- Selected `displacement`, `horsepower`, and `weight` as predictor variables (X) to estimate miles per gallon (MPG) as the target variable (Y).

```python
import pandas as pd            # for working with data
import numpy as np             # for numerical calculations
import matplotlib.pyplot as plt  # for making graphs
from sklearn.linear_model import LinearRegression  # the regression model
from sklearn.model_selection import train_test_split  # for splitting the data
from sklearn.metrics import r2_score  # to measure model performance

am_df = pd.read_csv('data/auto-mpg-cleaned.csv')  # load the dataset
am_df.sample(5)  # show a few random rows from the dataset
am_df.dtypes
```

## 2. Data Splitting
- Used `train_test_split` to divide the dataset into training and test sets.

```python
x = am_df[['displacement', 'horsepower', 'weight']]  # features
y = am_df['mpg']  # target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # split the data into training and testing sets
```

## 3. Model Training
- Utilized `LinearRegression` from scikit-learn.
- Normalized all numeric features before training.
- Trained the model using the training data.

```python
model = LinearRegression()  # create a linear regression model
model = model.fit(x_train, y_train)  # train the model on the training data
print(f'Training Score: {model.score(x_train, y_train)}')  # print the score of the model
```

## 4. Model Evaluation
- Achieved an R² score of **70%** on the training data, an improvement from **61%** when using only horsepower.
- All coefficients for the predictors (`horsepower`, `displacement`, `weight`) were negative, indicating that higher values of these features reduce car mileage.
- Predicted on the test set and obtained an R² score of **71%**, showing strong model performance.

```python
predictors = x_train.columns  # get the names of the features 
coef = pd.Series(model.coef_, index=predictors).sort_values() # plot the coefficients of the model
print(coef)  # print the coefficients

y_pred = model.predict(x_test)  # make predictions on the test data
print(f'Test Score: {r2_score(y_test, y_pred)}')  #
```

## 5. Visualization
- Plotted predicted vs. actual MPG values using a line chart.
- The predicted values (blue) closely tracked the actual values (orange).

```python
plt.plot(y_test.values, label='Actual', color='orange')
plt.plot(y_pred, label='Predicted', color='blue')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('MPG')
plt.title('Actual vs Predicted MPG')
plt.show()
```

## 6. Adding More Features
- Included additional features: `acceleration`, `cylinders`, and optionally the car's age.
- Trained a new model with these five features.
- The new model achieved an R² score of **60%** on the training data.
- The R², which dropped to 60%, indicating that adding more features did not necessarily improve the model.

```python
x = am_df[['displacement', 'horsepower', 'weight', 'acceleration', 'cylinders']]  # features
y = am_df['mpg']  # target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # split the data into training and testing sets
model = LinearRegression().fit(x_train, y_train)  # create a linear regression model
print(f'Training Score: {model.score(x_train, y_train)}')  # print the score of the model

y_pred = model.predict(x_test)  # make predictions on the test data
print(f'Test Score: {r2_score(y_test, y_pred)}')  #
```

## Key Points

- **Multiple Features**: Using multiple features can enhance a model's predictive power, as demonstrated by the initial improvement in R-squared scores.

- **Model Coefficients**: Negative coefficients suggest that an increase in the corresponding features results in a decrease in the target variable (MPG).

- **Model Robustness**: A reliable model maintains strong performance on both training and test datasets.

- **Kitchen Sink Regression**: Including too many features without evaluating their predictive relevance can reduce model performance, as seen by the drop in R-squared score on the test data.

- **Machine Learning Process**: Training and evaluating different models is essential to identifying solutions that generalize well in real-world scenarios.

---

By applying these principles, it is possible to build, train, and evaluate a linear regression model using multiple features, while emphasizing the importance of feature selection and model evaluation.

## Summary

Multiple regression is a powerful tool for modeling complex relationships, but it comes with risks like multicollinearity that can reduce model reliability. By carefully selecting variables, applying standardization, and considering dimensionality reduction, it is possible to improve model performance and ensure meaningful, generalizable insights.
