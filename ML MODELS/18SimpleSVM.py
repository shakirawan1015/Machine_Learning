import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = pd.read_csv('Placement_Svc.csv')

print(dataset.head())



x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# train the SVC model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = SVC(kernel = "linear")
model.fit(x_train, y_train)
print("SVC Model Test Score:",model.score(x_test,y_test)*100)
print("SVC Model Train Score:",model.score(x_train,y_train)*100)

sns.scatterplot(data=dataset, x="cgpa", y="score", hue="placement", palette="coolwarm")
plt.axvline(x=6.75, color='green', linestyle='--', alpha=0.7)
plt.xlabel("CGPA")
plt.ylabel("Score")
plt.title("Perfectly Linearly Separable Data for SVC")
plt.legend(title="Placement")

plt.show()