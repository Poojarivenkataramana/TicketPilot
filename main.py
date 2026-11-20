import pandas as pd
df1 = pd.read_csv("aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df2 = pd.read_csv("dataset-tickets-german_normalized.csv")
df3 = pd.read_csv("dataset-tickets-german_normalized_50_5_2.csv")
df4 = pd.read_csv("dataset-tickets-multi-lang3-4k.csv")
df5 = pd.read_csv("dataset-tickets-multi-lang-4-20k.csv")
# print(df.loc["language"].head())
print(df1.info())
print(df2.info())
print(df3.info())
print(df4.info())
print(df5.info())

