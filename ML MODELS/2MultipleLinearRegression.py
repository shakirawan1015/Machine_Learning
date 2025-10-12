# Import required libraries
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Starting Multiple Linear Regression Analysis...")

# Step 1: Load the dataset
try:
    print("Loading 'Salary_Age.csv' dataset...")
    dataset = pd.read_csv('Salary_Age.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'Salary_Age.csv' file not found.")
    print("Please make sure the file exists in the current directory.")
    sys.exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit()

# Step 2: Display basic dataset information
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"\nDataset shape: {dataset.shape}")
print("\nMissing values in each column:")
print(dataset.isnull().sum())

# Check if dataset is empty
if dataset.empty:
    print("Warning: Dataset is empty!")
else:
    print(f"Dataset contains {dataset.shape[0]} rows and {dataset.shape[1]} columns.")

# Step 3: Visualize relationships
print("\nGenerating pairplot...")
try:
    pairplot = sns.pairplot(dataset, height=4)
    print("Pairplot generated Successfully!")
except Exception as e:
    print(f"Warning: Could not generate pairplot: {e}")

print("Generating correlation heatmap...")
try:
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    print("Correlation heatmap generated Successfully!")
except Exception as e:
    print(f"Warning: Could not generate correlation heatmap: {e}")

# Step 4: Define features and target
X = dataset[['age', 'experience']]   
y = dataset['salary']                

print("\n" + "="*50)
print("DATA PREPARATION")
print("="*50)
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Step 6: Train the model
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)
print("Training the Multiple Linear Regression model...")
try:
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed!")
except Exception as e:
    print(f"Error during model training: {e}")
    sys.exit()

# Step 7: Evaluate the model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
try:
    score = model.score(X_test, y_test) * 100
    print(f"Model Accuracy (R-squared): {score:.2f}%")
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)
except Exception as e:
    print(f"Error during model evaluation: {e}")

# Step 8: Make predictions
print("\n" + "="*50)
print("MAKING PREDICTIONS")
print("="*50)
try:
    predictions = model.predict(X_test)
    print("Predictions generated!")
    
    # Show first 5 actual vs predicted values
    print("\nFirst 5 Actual vs Predicted Salaries:")
    results = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})
    print(results.head())
except Exception as e:
    print(f"Error during predictions: {e}")

# Step 9: Plot actual vs predicted values
print("\nGenerating visualization of actual vs predicted salaries...")
try:
    fig = plt.figure(figsize=(8,6))
    plt.scatter(y_test, predictions, alpha=0.7, color='blue', label='Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Ideal Prediction')
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.grid(True)
    plt.legend()
    print("Visualization generated!")
except Exception as e:
    print(f"Warning: Could not generate visualization: {e}")

# After the Actual vs Predicted plot
print("\nGenerating Residuals Plot...")
try:
    residuals = y_test - predictions
    fig = plt.figure(figsize=(8,6))
    plt.scatter(predictions, residuals, alpha=0.7, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs Predicted Values")
    plt.grid(True)
    print("Residuals plot generated!")
except Exception as e:
    print(f"Warning: Could not generate residuals plot: {e}")

# Show all plots and keep them open until manually closed
print("\nDisplaying all visualizations...")
print("All plots are now displayed. Close all plot windows to finish execution.")
plt.show()