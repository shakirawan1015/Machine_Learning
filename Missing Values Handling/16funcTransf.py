import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Load dataset
dataset = pd.read_csv('LoanCsv.csv')

# Show basic statistics
print("Original LoanAmount statistics:")
print(dataset['LoanAmount'].describe())

#Use 3-sigma rule for outlier detection
mean = dataset['LoanAmount'].mean()
std = dataset['LoanAmount'].std()
lower_bound = mean - 3 * std
upper_bound = mean + 3 * std

print(f"\nMean: {mean:.2f}, Std Dev: {std:.2f}")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

# Detect and remove outliers
outliers = dataset[(dataset['LoanAmount'] < lower_bound) | (dataset['LoanAmount'] > upper_bound)]
clean_dataset = dataset[~((dataset['LoanAmount'] < lower_bound) | (dataset['LoanAmount'] > upper_bound))].copy()

print(f"\nDetected Outliers: {len(outliers)} row(s)")
print(f"Original: {len(dataset)}, Cleaned: {len(clean_dataset)}")

# Show statistics after cleaning
print("\nLoanAmount statistics after outlier removal:")
print(clean_dataset['LoanAmount'].describe())

# Save cleaned data
clean_dataset.to_csv('LoanCsv_CLEANED.csv', index=False)
print("\nSaved cleaned dataset to 'LoanCsv_CLEANED.csv'")


ft = FunctionTransformer(np.log1p, validate = True)
dataset["LoanAmount_TF"] = ft.fit_transform(dataset[["LoanAmount"]])
# ft = FunctionTransformer(func = lambda x : x**2, validate = True)
# dataset["LoanAmount_TF"] = ft.fit_transform(dataset[["LoanAmount"]])


# Plot before and after transformation
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(dataset['LoanAmount'], 
             bins=30, color="#1733D4", 
             alpha=0.7,kde=True,
             line_kws={"color": "#77A115F8"})  
plt.xlabel('Loan Amount', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Before: LoanAmount Distribution', fontsize = 16, fontweight = 'bold'  )

plt.subplot(1, 2, 2)
sns.histplot(dataset['LoanAmount_TF'],
              bins=30, color='green',
                alpha=0.7, kde=True,
                line_kws={"color": "#123BF0"})  
plt.xlabel('Loan Amount (Transformed)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('After: Log Transformation',fontsize = 16, fontweight = 'bold'  )

plt.tight_layout()
plt.show()