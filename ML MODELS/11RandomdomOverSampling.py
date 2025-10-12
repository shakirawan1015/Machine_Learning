import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler  

# Load data
dataset = pd.read_csv('social_network_ads_imbalanced.csv')
print(dataset.head())
print("\nOriginal class distribution:")
print(dataset["Purchased"].value_counts())

# Features and target
x = dataset[['Age', 'EstimatedSalary']]
y = dataset['Purchased']

#  Split FIRST (keep test set untouched)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=0,
    stratify=y 
)

#  Oversample ONLY training data
ros = RandomOverSampler(random_state=42)  # ← CHANGED SAMPLER
x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

print("\nAfter oversampling (training set only):")
print(y_train_resampled.value_counts())

# Train model
model = LogisticRegression()
model.fit(x_train_resampled, y_train_resampled)

# Evaluate on ORIGINAL imbalanced test set
print(f"\nModel Accuracy: {model.score(x_test, y_test) * 100:.2f}%")

# Sample prediction
new_data = pd.DataFrame([[30, 50000]], columns=['Age', 'EstimatedSalary'])
print("\nSample prediction:", model.predict(new_data))