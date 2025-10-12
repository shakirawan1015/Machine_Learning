import pandas as pd
import numpy as np

dataset = pd.read_csv(r"C:\Users\Quantom Coder\Desktop\employee_data_missing.csv")
data = dataset.isnull().sum()
print(dataset)
print("="*50)
print("Missing Values:", data)