import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and prepare data
dataset = pd.read_csv('salary_data_experience_KNN.csv')
print(dataset.head())

x = dataset.iloc[:, :-1].values  # Features: age, experience
y = dataset.iloc[:, -1].values   # Target: salary

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Find the best K value (1 to 30)
best_score = -1
best_k = 1

for k in range(1, 31):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k

# Train final model with best K
best_model = KNeighborsRegressor(n_neighbors=best_k)
best_model.fit(x_train, y_train)

# Make predictions
y_pred = best_model.predict(x_test)

# Calculate metrics
test_r2 = best_model.score(x_test, y_test)
train_r2 = best_model.score(x_train, y_train)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Print results
print(f"Best K Value: {best_k}")
print(f"Test Accuracy: {test_r2 * 100:.2f}%")
print(f"Training Accuracy: {train_r2 * 100:.2f}%")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Plot K values vs performance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 31), 
         [KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train).score(x_test, y_test) 
         for k in range(1, 31)], marker='o')
plt.xlabel('K Values')
plt.ylabel('R² Score')
plt.title('KNN Performance')
plt.grid(True, alpha=0.3)

# Plot actual vs predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Example prediction
example = scaler.transform([[35, 5]])  # Age 35, 5 years experience
salary = best_model.predict(example)
print(f"\nPredicted salary (Age 35, 5 yrs exp): ${salary[0]:,.2f}")