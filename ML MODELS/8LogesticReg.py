import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
try:
    dataset = pd.read_csv("iris.csv")
except FileNotFoundError:
    print("File 'iris.csv' not found. Please check the path.")
    sys.exit()

# print("Unique Species:", dataset["species"].unique())
# sns.pairplot(dataset, hue="species")
# plt.show()

# Prepare full dataset for multiclass models
X_full = dataset.iloc[:, :-1]
y_full = dataset["species"]

le_full = LabelEncoder()
y_full = le_full.fit_transform(y_full)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# One-vs-Rest Logistic Regression
print("\nOne-vs-Rest Logistic Regression (3 classes)")
ovr_model = OneVsRestClassifier(LogisticRegression(max_iter=200))
ovr_model.fit(X_train, y_train)
ovr_acc = ovr_model.score(X_test, y_test) * 100
print(f"Accuracy: {ovr_acc:.2f}%")

# Multinomial Logistic Regression (default behavior)
print("\nMultinomial Logistic Regression (3 classes)")
multi_model = LogisticRegression(max_iter=200)
multi_model.fit(X_train, y_train)
multi_acc = multi_model.score(X_test, y_test) * 100
print(f"Accuracy: {multi_acc:.2f}%")

# True Binary Logistic Regression (2 classes only)
print("\nBinary Logistic Regression (2 classes: setosa vs versicolor)")

binary_data = dataset[dataset["species"].isin(["setosa", "versicolor"])]
X_binary = binary_data.iloc[:, :-1]
y_binary = binary_data["species"]

le_binary = LabelEncoder()
y_binary = le_binary.fit_transform(y_binary)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

binary_model = LogisticRegression(max_iter=200)
binary_model.fit(X_train_b, y_train_b)
binary_acc = binary_model.score(X_test_b, y_test_b) * 100
print(f"Accuracy (setosa vs versicolor): {binary_acc:.2f}%")

# Final summary
print("\nFinal Comparison:")
print(f"One-vs-Rest (3 classes)     : {ovr_acc:.2f}%")
print(f"Multinomial (3 classes)     : {multi_acc:.2f}%")
print(f"Binary LR (2 classes only)  : {binary_acc:.2f}%")