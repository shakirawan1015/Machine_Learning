import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('Salary_Age.csv')
print("Data shape:", data.shape)

# Prepare features and target
X = data[["age"]].values  
y = data["salary"].values  

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = SVR(kernel="poly", C=100, degree=3)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
print("\nModel Performance:")
print(f"Training Score: {model.score(X_train_scaled, y_train):.2%}")
print(f"Test Score: {model.score(X_test_scaled, y_test):.2%}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Create single visualization
plt.figure(figsize=(10, 6))

# Plot data and regression line
plt.scatter(data["age"], data["salary"], color='blue', alpha=0.6, label='Data points')

# Generate smooth prediction line
age_range = np.linspace(data["age"].min(), data["age"].max(), 100).reshape(-1, 1)
age_range_scaled = scaler.transform(age_range)
salary_pred = model.predict(age_range_scaled)

plt.plot(age_range, salary_pred, color='red', linewidth=2, label='SVR Model')
plt.title('Age vs Salary - SVR Regression')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()