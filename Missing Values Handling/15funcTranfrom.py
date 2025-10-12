import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
dataset = pd.read_csv('LoanCsv.csv')

# Show basic statistics
print("Original LoanAmount statistics:")
print(dataset['LoanAmount'].describe())

# Use 3-sigma rule for outlier detection
mean = dataset['LoanAmount'].mean()
std = dataset['LoanAmount'].std()
lower_bound = mean - 3 * std
upper_bound = mean + 3 * std

print(f"\nMean: {mean:.2f}, Std Dev: {std:.2f}")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

# Detect and remove outliers
is_outlier = (dataset['LoanAmount'] < lower_bound) | (dataset['LoanAmount'] > upper_bound)
outliers = dataset[is_outlier]
clean_dataset = dataset[~is_outlier].copy()

print(f"\nDetected Outliers: {len(outliers)} row(s)")
print(f"Original: {len(dataset)}, Cleaned: {len(clean_dataset)}")

# Show statistics after cleaning
print("\nLoanAmount statistics after outlier removal:")
print(clean_dataset['LoanAmount'].describe())

# Save cleaned data
clean_dataset.to_csv('LoanCsv_CLEANED.csv', index=False)
print("\nSaved cleaned dataset to 'LoanCsv_CLEANED.csv'")

# Simple before/after visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.xlabel('Loan Amount', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
sns.histplot(dataset['LoanAmount'], color="blue", kde=True)
plt.title('Before: LoanAmount Distribution', fontsize = 16, fontweight = 'bold'  )

plt.subplot(1, 2, 2)
plt.xlabel('Loan Amount', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
sns.histplot(clean_dataset['LoanAmount'], color="green", kde=True)
plt.title('After: Outliers Removed',fontsize = 16, fontweight = 'bold'  )

plt.tight_layout()
plt.show()