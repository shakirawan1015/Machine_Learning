import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

# Load data
dataset = pd.read_csv('student_cgpa.csv')
print(dataset.head())

# Create binary target variable (high_package)
threshold = dataset['package'].median()
dataset['high_package'] = (dataset['package'] >= threshold).astype(int)

print(f"\nUsing median package ({threshold:.2f}) as threshold")
print(f"Class distribution: Low={sum(dataset['high_package']==0)}, High={sum(dataset['high_package']==1)}")

# Prepare features and target (ONLY cgpa as feature, high_package as target)
X = dataset[['cgpa']]  # Only CGPA as feature
y = dataset['high_package']  # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Correct accuracy calculation
accuracy = model.score(X_test, y_test) * 100
print(f"\nGaussianNB Model Accuracy: {accuracy:.2f}%")
print("\nSimple Predictions",model.predict([6.5,9.0]))

model_2 = BernoulliNB()
model_2.fit(X_train, y_train)
accuracy_2 = model_2.score(X_test, y_test) * 100
print(f"\nBernoulliNB Model Accuracy: {accuracy_2:.2f}%")

model_3 = MultinomialNB()
model_3.fit(X_train, y_train)
accuracy_3 = model_3.score(X_test, y_test) * 100
print(f"\nMultinomialNB Model Accuracy: {accuracy_3:.2f}%")


# Correct visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataset, x='cgpa', y='package', hue='high_package', palette=['red', 'blue'])
plt.title('CGPA vs Package (Colored by High Package)')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.legend(title='High Package', labels=['Low Package', 'High Package'])
plt.grid(True, alpha=0.3)
plt.show()

# Optional: If you want KDE plot (density plot)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=dataset, x='cgpa', hue='high_package', fill=True, alpha=0.5)
plt.title('CGPA Distribution by Package Category')
plt.xlabel('CGPA')
plt.show()