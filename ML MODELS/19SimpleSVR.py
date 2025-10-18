import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


# Load dataset
dataset = pd.read_csv('student_academic_performance.csv')
print(dataset.head())

# Prepare features and target
X = dataset[['cgpa']].values
y = dataset['score'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train SVR
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Evaluate
print("SVR Model Test Score:", model.score(X_test, y_test) * 100)
print("SVR Model Train Score:", model.score(X_train, y_train) * 100)

# Create smooth prediction curve (sorted by CGPA for clean plot)
X_smooth = np.linspace(dataset['cgpa'].min(), dataset['cgpa'].max(), 300).reshape(-1, 1)
y_smooth = model.predict(X_smooth)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='cgpa', y='score', data=dataset, alpha=0.6, label='Actual Data')
plt.show()

# Plot smooth SVR fit
plt.plot(X_smooth, y_smooth, color='red', linewidth=2, label='SVR Fit')

plt.xlabel('CGPA')
plt.ylabel('Score')
plt.title('Support Vector Regression (SVR) - RBF Kernel')
plt.legend()
plt.grid(alpha=0.3)
plt.show()