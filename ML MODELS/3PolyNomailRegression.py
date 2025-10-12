import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
try:
    dataset = pd.read_csv('polynomial_regression_500.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'polynomial_regression_500.csv' not found.")
    import sys
    sys.exit(1)

print(f"Dataset shape: {dataset.shape}")
print(dataset.head(1))

# Prepare features and target
X = dataset[["age"]]  # Keep as DataFrame to maintain feature names
y = dataset["salary"].astype(float)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find best polynomial degree
degrees = range(1, 6)
best_degree, best_r2 = 1,-np.inf

print("Testing polynomial degrees 1-5:")
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Convert to DataFrame to maintain feature names
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out())
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out())
    
    model = LinearRegression()
    model.fit(X_train_poly_df, y_train)
    
    y_pred = model.predict(X_test_poly_df)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Degree {degree}: R² = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree
        best_model = model
        best_poly = poly
        best_feature_names = poly.get_feature_names_out()

print(f"\nBest Degree: {best_degree} with R² = {best_r2:.4f}")

# Train final model
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Convert to DataFrame to maintain feature names
X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out())
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out())

model = LinearRegression()
model.fit(X_train_poly_df, y_train)

# Evaluate final model
y_train_pred = model.predict(X_train_poly_df)
y_test_pred = model.predict(X_test_poly_df)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Final Model Performance:")
print(f"  Training R²: {train_r2:.4f}")
print(f"  Testing R²: {test_r2:.4f}")

# Visualization
plt.figure(figsize=(14, 8))
plt.scatter(X_train['age'], y_train, color='green', alpha=0.6, label='Training Data')
plt.scatter(X_test['age'], y_test, color='red', alpha=0.6, label='Testing Data')

# Plot polynomial curve
age_range = pd.DataFrame({"age": np.linspace(X['age'].min(), X['age'].max(), 300)})
age_poly = poly.transform(age_range)
age_poly_df = pd.DataFrame(age_poly, columns=poly.get_feature_names_out())
y_plot = model.predict(age_poly_df)

plt.plot(age_range['age'], y_plot, color='blue', linewidth=3, 
         label=f'Polynomial Fit (Degree {best_degree})')

plt.xlabel("Age")
plt.ylabel("Salary")
plt.title(f"Polynomial Regression: Age vs Salary (Best Degree: {best_degree})")
plt.legend()
plt.grid(True, alpha=0.3)
print("All charts have been generated. Please view and close the window to end the program.")
plt.show()