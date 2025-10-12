# === Imports ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# === Load & Prep Data ===
df = pd.read_csv("house_data_500.csv")

# Feature Engineering
df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
df.loc[df['yr_renovated'] == 0, 'yr_renovated'] = df['yr_built']
df.loc[df['yr_renovated'] > 2015, 'yr_renovated'] = df['yr_built']

# Separate features & target
X = df.drop('price', axis=1)
y = df['price']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features (no leakage!)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# === Train Models ===
models = {
    "Linear": LinearRegression(),
    "Lasso": Lasso(alpha=0.001, max_iter=5000),
    "Ridge": Ridge(alpha=0.001)
}

results = []
coef_data = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    # Metrics
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    results.append({'Model': name, 'R²': r2, 'MAE': mae})
    
    # Coefficients
    coef_data.append({'Feature': X.columns, name: model.coef_})

# === Display Results ===
print("\n MODEL PERFORMANCE")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n COEFFICIENTS (Top 5 by Linear Regression)")
coef_df = pd.DataFrame(coef_data[0])  # Start with Linear
coef_df = coef_df.reindex(coef_df['Linear'].abs().sort_values(ascending=False).index)
print(coef_df.head().to_string(index=False))

# === Plots ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    axes[i].bar(X.columns, model.coef_, color=['lightblue','lightgreen','lightcoral'][i], edgecolor='black')
    axes[i].set_title(f"{name} Coefficients")
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\n Analysis complete. Clean. Simple. Powerful.")