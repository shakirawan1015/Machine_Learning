import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

try:
    dataset = pd.read_csv("student_cgpa.csv")
except FileNotFoundError:
    dataset = pd.read_csv("../student_cgpa.csv")

print(dataset.head(3))

x = dataset[["cgpa"]]
print(x.ndim)

y = dataset["package"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Model Score:",model.score(x_test, y_test) * 100)

# Prepare regression line for plotting - Fixed to ensure feature names consistency
x_line = pd.DataFrame({"cgpa": [x['cgpa'].min(), x['cgpa'].max()]}, columns=["cgpa"])
y_line = model.predict(x_line)

# Plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=dataset, x="cgpa", y="package", label="Data Points")
plt.plot(x_line["cgpa"], y_line, color='red', linewidth=2, label="Regression Line")
plt.grid()
plt.title("Package Based on CGPA")
plt.xlabel("CGPA")
plt.ylabel("Package (LPA)")
plt.legend()  # legend for "Data Points" and "Regression Line"
plt.savefig("SimpleLinearRegression.png")
plt.show()

