import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Boston.csv")  # adjust path

# Separate features and target
X = df.drop('House_Price', axis=1)
y = df['House_Price']

# Sanity check
print("Full dataset shape:", df.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split correctly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\n--- TRAIN/TEST SPLIT ---")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

