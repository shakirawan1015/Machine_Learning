import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_decision_regions



dataset = pd.read_csv("social_network_ads_1000.csv")
print(dataset.head())


sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=dataset
                )
plt.show()

x = dataset.iloc[:, :-1].values  # Select all rows and all columns except the last
y = dataset.iloc[:, -1].values   # Select all rows and only the last column
print("Feature matrix x:")
print(x)


Sc = StandardScaler()
x = Sc.fit_transform(x)
print("Scaled feature matrix x:")
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# to check the best k value
# for i in range(1,50):
#     KNN_Model = KNeighborsClassifier(n_neighbors=i)
#     KNN_Model.fit(x_train,y_train)
#     y_pred = KNN_Model.predict(x_test)

#     accuracy = accuracy_score(y_test, y_pred)*100
#     print(f"Model Accuracy: {accuracy:.2f}")

#     score= KNN_Model.score(x_test,y_test)*100
#     print(f"test Score: {score:.2f}")

#     score= KNN_Model.score(x_train,y_train)*100
#     print(f"Train Score: {score:.2f}")


best_k = 23  # Best value based on results 
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
final_accuracy = accuracy_score(y_test, y_pred) * 100
final_score = model.score(x_test, y_test) * 100
train_score = model.score(x_train, y_train) * 100

print(f"KNN Model with best k value ={best_k}")
print(f"Model Accuracy: {final_accuracy:.2f}%")
print(f"Test Score: {final_score:.2f}%")
print(f"Train Score: {train_score:.2f}%")

# Visualizing decision boundaries
plt.figure(figsize=(10, 6))
plot_decision_regions(x, y, clf=model, legend=2)
plt.title(f'KNN Decision Boundaries (k={best_k})')
plt.xlabel('Age (scaled)')
plt.ylabel('Estimated Salary (scaled)')
plt.show()



prediction = model.predict([[27, 49373]])
print(f"Prediction for customer (Age=27, Salary=49373): {'Purchased' if prediction[0] == 1 else 'Not Purchased'}")
