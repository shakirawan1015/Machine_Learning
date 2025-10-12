import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
import numpy as np

dataset = pd.read_csv('social_network_ads_1000.csv')
print(dataset.head())
print("Dataset Shape",dataset.shape)
print(dataset.isnull().sum())

x = dataset[['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

DT_Model = DecisionTreeClassifier(max_depth = 4, random_state=42)
DT_Model.fit(x_train,y_train)

train_acc = DT_Model.score(x_train, y_train)
test_acc = DT_Model.score(x_test, y_test)
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy:  {test_acc:.2%}")

# First plot - Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    DT_Model,
    feature_names=['Age', 'EstimatedSalary'],
    class_names=['Not Purchased', 'Purchased'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()

# Second plot - Decision Regions
plt.figure(figsize=(10, 8))
plot_decision_regions(x, y, clf=DT_Model, legend=2)
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Decision Regions for Decision Tree Classifier")
plt.show()

# Third plot - Scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=dataset,
                x="Age",
                y="EstimatedSalary",
                hue="Purchased")
plt.title("Data Distribution")

plt.show()