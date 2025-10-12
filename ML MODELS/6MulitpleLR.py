import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

# Load data
dataset = pd.read_csv('social_network_ads_1000.csv')
print("Data loaded.")

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataset, x='Age', y='EstimatedSalary', hue='Purchased')
plt.title('Age vs Estimated Salary')
plt.show()

# Prepare features and target
X = dataset[['Age', 'EstimatedSalary']]
y = dataset['Purchased'].astype(int)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
print(f"Model Accuracy: {accuracy:.1f}%")

# Predict for new samples
new_data = pd.DataFrame([[35, 80000], [25, 40000]], columns=['Age', 'EstimatedSalary'])
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)

# Plot decision regions
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train.to_numpy(), clf=model, legend=2)
plt.title('Logistic Regression Decision Regions')
plt.show()

print("\nPredictions:")
print(f"Person 1 (Age=35, Salary=80K)  Will buy: {predictions[0]}")
print(f"Person 2 (Age=25, Salary=40K)  Will buy: {predictions[1]}")
