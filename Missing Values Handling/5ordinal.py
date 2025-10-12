import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Create DataFrame
data = pd.DataFrame({
    "letters": ["a", "u", "c", "k", "g", "w", "b", "d", "e", "f", "h", "i"],
})

# Display settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


ord_data = [["a", "u", "c", "k", "g", "w", "b", "d", "e", "f", "h", "i"]]
# Ordinal Encoding
ordinal_encoder = OrdinalEncoder(categories=ord_data)
data["Encoded_letters"] = ordinal_encoder.fit_transform(data[["letters"]]).flatten()

# Print result
print("=" * 40, "\n\t\t\tEncoded Data:\n", data)
