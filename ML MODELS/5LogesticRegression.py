import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


try:
    dataset = pd.read_csv('social_network_ads_1000.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    try:
        dataset = pd.read_csv(r"C:\Users\Quantom Coder\Desktop\social_network_ads_1000.csv")
        print("Dataset loaded successfully from alternative path.")
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        sys.exit()

print(dataset.head())

# Features: Age + Salary
x = dataset[["Age", "EstimatedSalary"]]
# Target: Purchased (0/1)
y = dataset["Purchased"]

# Split 80/20, fixed random state
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train LR model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict & score
y_pred = model.predict(x_test)
accuracy = model.score(x_test, y_test) * 100
print("Predicted Values:", y_pred)
print(f"Model Score: {accuracy:.1f}%")

# Plot actual data
sns.scatterplot(data=dataset, x="Age", y="Purchased", label="Actual Data", alpha=0.7)

# Prep smooth prediction line (fix salary at mean)
dataset_sorted = dataset.sort_values("Age")
mean_salary = dataset["EstimatedSalary"].mean()
X_plot = dataset_sorted[["Age"]].copy()
X_plot["EstimatedSalary"] = mean_salary

# Plot model's predictions
sns.lineplot(
    x=dataset_sorted["Age"],
    y=model.predict(X_plot),
    color='red',
    linewidth=2,
    label=f"LR Prediction (Salary = {int(mean_salary):,})"
)

# Polish plot
plt.title("Logistic Regression: Purchase Prediction by Age", weight='bold', size=12)
plt.xlabel("Age")
plt.ylabel("Purchased (0 = No, 1 = Yes)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()