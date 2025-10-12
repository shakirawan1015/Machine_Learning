import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('student_cgpa.csv')

print("Dataset Info:")
print(dataset.head())
print(f"\nDataset shape: {dataset.shape}")
print(f"Package statistics:\n{dataset['package'].describe()}")

threshold = dataset['package'].median()
print(f"\nUsing median package value ({threshold:.2f}) as threshold for classification")

dataset['high_package'] = (dataset['package'] >= threshold).astype(int)

print(f"\nClass distribution:")
print(f"Low package (0): {sum(dataset['high_package'] == 0)} students")
print(f"High package (1): {sum(dataset['high_package'] == 1)} students")

x = dataset[['cgpa']]
y = dataset['high_package']

unique_classes = y.unique()
print(f"\nUnique classes in target: {sorted(unique_classes)}")

if len(unique_classes) < 2:
    print("ERROR: Not enough classes for classification. Please check your data.")
    exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

print(f"\nTraining set class distribution:")
print(f"Low package (0): {sum(y_train == 0)} samples")
print(f"High package (1): {sum(y_train == 1)} samples")

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100 if len(np.unique(y_test)) > 1 else 0
recall = recall_score(y_test, y_pred) * 100 if len(np.unique(y_test)) > 1 else 0
f1 = f1_score(y_test, y_pred) * 100 if len(np.unique(y_test)) > 1 else 0

print(f"\nModel Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Package', 'High Package'], yticklabels=['Low Package', 'High Package'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()