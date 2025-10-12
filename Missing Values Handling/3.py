import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\Quantom Coder\Desktop\employee_data_missing.csv")
print("Original Data:")
print("="*40)
print(data)
print("\n")

# Set display options for better visibility
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Columns to encode
en_data = data[["Salary", "Education"]]

# Create the encoded data (don't call .info() here)
var = pd.get_dummies(en_data, dtype=int)

print("="*40)
print("Encoded Data:")
print(var)  # This will show the actual data
print("="*40)

# If you want to see info, call it separately
print("\nData Info:")
var.info()