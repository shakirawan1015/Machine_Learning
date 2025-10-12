import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Load data
data = pd.read_csv("C:\\Users\\Quantom Coder\\Desktop\\CarsData.csv")

# Setting display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print(data.head(3))

# Get unique values in the "Color" column
var = data["Color"].unique()
print("Unique colors in the dataset:")
print(var)

# Convert unique values into list of list (sorted to keep order stable)
var_list = [sorted(var.tolist())]

# Apply OrdinalEncoder
encoder = OrdinalEncoder(categories=var_list)
encoded_ordinal_data = encoder.fit_transform(data[["Color"]])

# Add encoded column back to dataframe
data["Color_Encoded"] = encoded_ordinal_data

print(data.head(10))
