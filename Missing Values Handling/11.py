import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load dataset and visualize unemployment rate distribution
dataset = pd.read_csv("empOut.csv")
sns.histplot(dataset["unemployment_rate"])
plt.title("Distribution of Unemployment Rate")
plt.show()

# Standardize the unemployment rate
scaler = StandardScaler()
scaler.fit(dataset[["unemployment_rate"]])
dataset["unemployment_rate"] = scaler.transform(dataset[["unemployment_rate"]])


# Display standardized data
print(dataset.head(3))
print(dataset.describe())