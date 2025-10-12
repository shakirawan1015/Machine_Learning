import pandas as pd
import numpy as np

dataset = pd.read_csv(r"C:\Users\Quantom Coder\Desktop\employee_data_missing.csv")
print(dataset)

pd.set_option('display.max_rows', None)   # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', 1000)      # Adjust width to fit content


print("="*50)
data = dataset.isnull().sum()
print("Missing Values:", data)

print("="*50)
dataset['Salary']=dataset['Salary'].fillna(dataset["Salary"].mean())
dataset["Education"] = dataset["Education"].fillna(dataset["Education"].mode()[0])
print("Dataset after filling missing values:")
print(dataset)

#checking the missing values again
data1 = dataset.isnull().sum()
print("Missing Values after filling:", data1)