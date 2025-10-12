import pandas as pd
import seaborn as sns

data = {"Letters": ["a", "b", "b", "c", "d", "d", "e", "e"], 
        "Eng": [8, 7, 7, 5, 8, 8, 8, 4], 
        "urdu": [2, 3, 3, 4, 5, 5, 2, 6]}

DF = pd.DataFrame(data)
DF["Duplicated"] = DF.duplicated()
print("DataFrame with Duplicates:\n", DF)

# DataFrame after the removal of duplicates
DF2=DF.drop_duplicates(subset=["Letters", "Eng", "urdu"],inplace=True)
print("DataFrame without Duplicates:\n", DF2)