import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("polynomial_regression_500.csv")
# print(dataset.head())

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Corrected parameter grid with valid parameter names
param_grid = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "splitter": ["best", "random"],
    "max_depth": range(1, 20),
    "min_samples_split": range(2, 10),
    "min_samples_leaf": range(1, 10)
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1  # Use all available processors
)

# Fit the grid search
grid_search.fit(x_train, y_train)

# Display results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Make predictions with the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print("Predicted values\n:", y_pred)
print("Actual values:\n", y_test)
print("Train Score:", best_model.score(x_train, y_train) * 100)
print("Test Score:", best_model.score(x_test, y_test) * 100)

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Decision Tree Regression - Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
