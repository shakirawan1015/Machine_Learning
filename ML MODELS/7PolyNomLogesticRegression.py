import pandas as pd
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset with error handling
try:
    dataset = pd.read_csv('polynomial_logistic_dataset_500.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file 'polynomial_logistic_dataset_500.csv' not found.")
    print("Please make sure the dataset file exists in the project directory.")
    sys.exit(1)

# Extract features and target variable
x = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].values   


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Apply polynomial features transformation
poly_features = PolynomialFeatures(degree=3)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)

# Train logistic regression model on polynomial features
model = LogisticRegression()
model.fit(x_train_poly, y_train)  # Correct method for training

# Make predictions
y_pred = model.predict(x_test_poly)

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy with Polynomial Features: {accuracy:.2f}%")

# For comparison, let's also train a model without polynomial features
model_linear = LogisticRegression()
model_linear.fit(x_train, y_train)
y_pred_linear = model_linear.predict(x_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear) * 100
print(f"Accuracy without Polynomial Features: {accuracy_linear:.2f}%")