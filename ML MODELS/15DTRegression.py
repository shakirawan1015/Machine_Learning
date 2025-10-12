import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree


dataset = pd.read_csv("salary_regression_dataset_1000.csv")
print(dataset.head()) 


sns.pairplot(dataset)
plt.show()

x = dataset[['Age', 'Experience']].values
y = dataset['Salary'].values
print("Features (X):",x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=5,          # Prevent overfitting
    min_samples_split=15, # Require enough samples to split
    min_samples_leaf=8,   # Ensure stable predictions in leaves
    random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Predicted values\n:",y_pred)
print("Actual values:\n",y_test)
print("Train Score:", model.score(x_train, y_train)*100)
print("Test Score: ", model.score(x_test, y_test)*100)


plt.figure(figsize=(12,8))
plot_tree(model,
          feature_names=['Age', 'Experience'], 
    filled=True,
    rounded=True,
    fontsize=10,
    precision=0)
plt.title("Decision Tree Regressor (Pruned)", fontsize=16)          
plt.show()
