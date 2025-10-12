import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("empOut.csv")

print(dataset.head())


sns.boxplot(x = "unemployment_rate",data=dataset)
plt.show()


sns.boxplot(x = "unemployment_rate",data=dataset)
plt.show()