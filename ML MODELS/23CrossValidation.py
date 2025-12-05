import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("student_placement_400.csv")

X = dataset[['cgpa']]  # Features (CGPA)
y = dataset['package']

# Get cross-validated predictions
cv_predictions = cross_val_predict(LinearRegression(), X, y, cv=KFold(n_splits=10))

# Calculate performance metrics
mse = mean_squared_error(y, cv_predictions)
r2 = r2_score(y, cv_predictions)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Visualize actual vs predicted
plt.scatter(y, cv_predictions, alpha=0.5)
plt.xlabel("Actual Package")
plt.ylabel("Predicted Package")
plt.title("Actual vs Predicted Packages")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# Show prediction statistics
print(f"Min prediction: {cv_predictions.min()}")
print(f"Max prediction: {cv_predictions.max()}")
print(f"Mean prediction: {cv_predictions.mean()}")