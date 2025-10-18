import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load dataset
dataset = pd.read_csv('student_academic_performance.csv')
print("Dataset shape:", dataset.shape)
print(dataset.head())

# Prepare features - using multiple columns if available
# Assuming the dataset might have more columns like 'attendance', 'hours_studied', etc.
feature_columns = [col for col in dataset.columns if col not in ['score', 'student_id']]
print("Features considered:", feature_columns)

X = dataset[feature_columns].values
y = dataset['score'].values

# Standardize features for better SVR performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Define different kernels to compare
kernels = ['linear', 'poly', 'rbf']
models, scores = {}, {}


print("\n--- Model Comparison ---")
for kernel in kernels:
    try:
        # Train SVR with different kernels
        model = SVR(kernel=kernel, C=1.0, gamma='scale')
        model.fit(X_train, y_train)
        
        # Store model
        models[kernel] = model
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)  # 5-fold cross-validation
        cv_score = np.mean(cv_scores)
        
        scores[kernel] = {'train': train_score, 'test': test_score, 'cv': cv_score}
        
        print(f"\n{kernel.upper()} Kernel:")
        print(f"  Train Score: {train_score*100:.2f}%")
        print(f"  Test Score: {test_score*100:.2f}%")
        print(f"  Cross-Validation Score: {cv_score*100:.2f}%")
        
        # Predictions for MSE calculation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"  Mean Squared Error: {mse:.2f}")
    except Exception as e:
        print(f"Error with {kernel} kernel: {e}")

# Visualization of model comparisons
plt.figure(figsize=(12, 5))
# Plot 1: Score Comparison
plt.subplot(1, 2, 1)
kernels_list = list(kernels)
train_scores = [scores[k]['train']*100 for k in kernels_list]
test_scores = [scores[k]['test']*100 for k in kernels_list]

x = np.arange(len(kernels_list))
width = 0.35

plt.bar(x - width/2, train_scores, width, label='Train Score')
plt.bar(x + width/2, test_scores, width, label='Test Score')

plt.xlabel('Kernel Type')
plt.ylabel('Accuracy (%)')
plt.title('SVR Performance Comparison')
plt.xticks(x, [k.upper() for k in kernels_list])
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Predictions vs Actual for best model (highest test score)
best_kernel = max(scores, key=lambda k: scores[k]['test'])
best_model = models[best_kernel]

plt.subplot(1, 2, 2)
y_pred_best = best_model.predict(X_test)

plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title(f'Actual vs Predicted ({best_kernel.upper()} Kernel)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nBest performing kernel: {best_kernel.upper()} with R2 Score: {scores[best_kernel]['test']:.4f}")
print(f"Explanation: The {best_kernel.upper()} kernel achieved the highest test accuracy of {scores[best_kernel]['test']*100:.2f}%")
