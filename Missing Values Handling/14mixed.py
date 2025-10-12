import pandas as pd
from word2number import w2n

# Load dataset
dataset = pd.read_csv('Employees.csv')

# Display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('future.no_silent_downcasting', True)

# Fill missing Name and FullTime with mode
dataset[["Name", "FullTime"]] = dataset[["Name", "FullTime"]].fillna(
    dataset[["Name", "FullTime"]].mode().iloc[0]
)

# Clean Salary column
dataset["Salary"] = (
    dataset["Salary"]
    .astype(str)
    .str.replace(r'[$,]', '', regex=True)
    .replace('', None)
    .astype(float)
)

# Convert Age from words/numbers to integers
def convert_age(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except:
        try:
            return w2n.word_to_num(str(x))
        except:
            return None

dataset["Age"] = dataset["Age"].apply(convert_age)

# Fill missing numeric columns with median
numeric_cols = ["Age", "Salary"]
dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].median(),inplace=True)

# Handle FeedbackTag (categorical)
dataset["FeedbackTag"] = dataset["FeedbackTag"].replace("Unknown", "Feedback is pending")
dataset["FeedbackTag"] = dataset["FeedbackTag"].fillna("Feedback is pending")

# Fill missing HoursWorked with mode
dataset["HoursWorked"] = dataset["HoursWorked"].fillna(dataset["HoursWorked"].mode()[0])

# Convert JoiningDate to datetime
dataset["JoiningDate"] = pd.to_datetime(dataset["JoiningDate"], unit="ns").ffill()
# dataset["JoiningDate"] = dataset["JoiningDate"].astype("int64")

# Display cleaned dataset
print("\n\n\t\tDataset After Cleaning...\n", dataset.head(20))

# Most common Name
most_common_name = dataset["Name"].mode()[0]
print("\n\nMost Occurring Name in the Column (Name):", most_common_name)
print(dataset["JoiningDate"].info())