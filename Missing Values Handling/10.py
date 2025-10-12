import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data
dataset = pd.read_csv("empOut.csv")
dataset["unemployment_rate"] = pd.to_numeric(dataset["unemployment_rate"], errors="coerce").fillna(dataset["unemployment_rate"].median())

# Calculate outlier bounds
q1, q3 = dataset["unemployment_rate"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

# Find outliers
outliers = dataset[(dataset["unemployment_rate"] < lower) | (dataset["unemployment_rate"] > upper)]

# Display results
print(f"Dataset head:\n{dataset.head()}")
print(f"Outlier bounds: [{lower:.2f}, {upper:.2f}]")
print(f"Number of outliers: {len(outliers)}")

# Plot
sns.boxplot(x="unemployment_rate", data=dataset)
plt.title("Distribution of Unemployment Rate")
plt.show()