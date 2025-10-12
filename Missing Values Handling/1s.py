import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("car_data_messy.csv")
dataset_filled = dataset.fillna(0)  
print("First 4 rows of the dataset:")
print(dataset.head())

# Create the heatmap
plt.figure(figsize=(10, 6))
var = sns.heatmap(dataset.isnull(), cmap='viridis', cbar=True)
plt.title('Missing Values Heatmap')
plt.show()

