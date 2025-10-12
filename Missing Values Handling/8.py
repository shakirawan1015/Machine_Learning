import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("C:\\Users\\Quantom Coder\\Desktop\\CarsData.csv")
print(dataset.head())
print("\nInfo:")
dataset.info()
print("\nDescribe:")
print(dataset.describe())

sns.boxplot(x="Price", data=dataset)
plt.show()