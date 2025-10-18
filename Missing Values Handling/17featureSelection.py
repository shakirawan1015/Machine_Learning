# # import pandas as pd
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.model_selection import StratifiedKFold
# # from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# # data = pd.read_csv("Foods.csv").dropna(subset=["Outcome"])

# # # Replace zeros with NaN for specific columns
# # for col in ['Glucose','BloodPres','SkinThick','BMI']:
# #     data[col] = data[col].replace(0, pd.NA)

# # data = data.dropna()

# # X = data.drop("Outcome", axis=1)
# # y = data["Outcome"]

# # lr = LogisticRegression(max_iter=1000, random_state=42)

# # # Create a StratifiedKFold object
# # cv = StratifiedKFold(n_splits=5, 
# #                      shuffle=True, 
# #                      random_state=42)

# # fs = SFS(lr, k_features=2, 
# #          forward=True, 
# #          scoring='accuracy', 
# #          cv=cv)
# # fs.fit(X, y)

# # print("Selected features:", fs.k_feature_names_)
# # print("CV score:", fs.k_score_)

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from mlxtend.plotting import plot_sequential_feature_selection

# data = pd.read_csv("Foods.csv").dropna(subset=["Outcome"])

# # Replace zeros with NaN for specific columns
# for col in ['Glucose','BloodPres','SkinThick','BMI']:
#     data[col] = data[col].replace(0, pd.NA)

# data = data.dropna()

# X = data.drop("Outcome", axis=1)
# y = data["Outcome"]

# lr = LogisticRegression(max_iter=1000, random_state=42)

# # Create a StratifiedKFold object
# cv = StratifiedKFold(n_splits=5, 
#                      shuffle=True, 
#                      random_state=42)

# fs = SFS(lr, k_features=2, 
#          forward=True, 
#          scoring='accuracy', 
#          cv=cv)
# fs.fit(X, y)

# print("Selected features:", fs.k_feature_names_)
# print("CV score:", fs.k_score_)

# # PLOTTING CODE - Add this part
# fig = plot_sequential_feature_selection(fs.get_metric_dict(), 
#                                        kind='std_dev',
#                                        figsize=(10, 6))

# plt.title('Sequential Feature Selection')
# plt.grid(True)
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection

# Use Agg backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

data = pd.read_csv("Foods.csv").dropna(subset=["Outcome"])

# Replace zeros with NaN for specific columns
for col in ['Glucose','BloodPres','SkinThick','BMI']:
    data[col] = data[col].replace(0, pd.NA)

data = data.dropna()

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

lr = LogisticRegression(max_iter=1000, random_state=42)

# Create a StratifiedKFold object
cv = StratifiedKFold(n_splits=5, 
                     shuffle=True, 
                     random_state=42)

fs = SFS(lr, k_features=2, 
         forward=True, 
         scoring='accuracy', 
         cv=cv)
fs.fit(X, y)

print("Selected features:", fs.k_feature_names_)
print("CV score:", fs.k_score_)

# Plot and save as image instead of showing
fig = plot_sequential_feature_selection(fs.get_metric_dict(), 
                                       kind='std_dev',
                                       figsize=(10, 6))

plt.title('Sequential Feature Selection')
plt.grid(True)
plt.savefig('feature_selection_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'feature_selection_plot.png'")