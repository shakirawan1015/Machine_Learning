import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create data
data = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    
    "Age": [25, 20, 30, 22, 23],
})

# setting display options
pd.set_option("display.max_rows", None) 
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)  

# Apply Label Encoding
le = LabelEncoder()
data[["encoded_Name"]] = data[["Name"]].apply(le.fit_transform)

# Show only the data with encoded column
print("Data with Encoded Column:")
print(data)