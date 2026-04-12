import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''Reading the dataset'''
data = pd.read_csv('full_data.csv')
df = pd.DataFrame(data)
print(df.shape)
print(df.describe)
print(df.isnull().sum().sum()) #Number of Null Values


'''Preprocessing Phase-1 (Removing Unnecessary Columns)'''
drop_columns = ['NomId', 'FilmId', 'NomineeIds',
    'Category', 'Nominees',
    'Detail', 'Note', 'Citation', 'MultifilmNomination']
df.drop(columns = drop_columns, axis = 1)

'''Preprocessing Phase-2 (Filtering rows with 100% win-rate)'''
df = df[~df["Class"].isin(["SciTech", "Special"])]
print(df.shape)             # should show 10,730 rows
print(df["Class"].unique()) # SciTech and Special should be gone

'''Preprocessing Phase-3 (Fixing the target variable winner)'''
df['Winner'].fillna("False") # replacing null values with false for no wins
df['Winner'].replace("True", 1) # 1 = win 
df['Winner'].replace("False", 0) # 0 = no win

