import pandas as pd
import seaborn as sns

dataset = pd.read_csv("Persons.csv")

#  Set the display option to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 1)  # Control decimal places

# Display the first 10 rows of the original dataset
print("Original DataFrame:\n", dataset)

# fill the NAN vlues
dataset[["Age","Salary"]] = dataset[["Age","Salary"]].fillna(dataset[["Age","Salary"]].mean())
dataset["Department"] = dataset["Department"].fillna(dataset["Department"].mode()[0])

print("\nDataFrame after filling NAN values:\n", dataset)

# DataFrame after the removal of duplicates
dataset = dataset.drop_duplicates(subset = ["Department"], inplace = True)

print("\nDataFrame after removing duplicates:\n", dataset)