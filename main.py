import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''Reading the dataset'''
data = pd.read_csv('full_data.csv',sep = '\t')
df = pd.DataFrame(data)
print(df.shape)
print(df.describe())
print(df.isnull().sum().sum()) #Number of Null Values


'''Preprocessing Phase-1 (Removing Unnecessary Columns)'''
drop_columns = ['NomId', 'FilmId', 'NomineeIds',
    'Category', 'Nominees',
    'Detail', 'Note', 'Citation', 'MultifilmNomination']
df = df.drop(columns = drop_columns)

'''Preprocessing Phase-2 (Filtering rows with 100% win-rate)'''
df = df[~df["Class"].isin(["SciTech", "Special"])]
print(df.shape)             # should show 10,730 rows
print(df["Class"].unique()) # SciTech and Special should be gone

'''Preprocessing Phase-3 (Fixing the target variable winner)'''
df['Winner'] = df['Winner'].fillna(False) # replacing null values with false for no wins
df['Winner'] = df['Winner'].astype(int) 
print(df['Winner'].value_counts())

'''Preprocessing Phase-4 (Engineering the Year Column)'''
df['Year_clean'] = df['Year']
df['Year_clean'] = df['Year_clean'].replace("1927/28","1928")
df['Year_clean'] = df['Year_clean'].replace("1928/29","1929")
df['Year_clean'] = df['Year_clean'].replace("1929/30","1930")
df['Year_clean'] = df['Year_clean'].replace("1930/31","1931")
df['Year_clean'] = df['Year_clean'].replace("1931/32","1932")
df['Year_clean'] = df['Year_clean'].replace("1932/33","1933")
df['Year'].astype(int)
df['Decade'] = (df['Year_clean'] // 10) * 10

'''Preprocessing Phase-5 (Handle remaining missing values)'''
df['Film'] = df["Film"].fillna('Unknown')
df['Name'] = df["Name"].fillna('Unknown')

'''Preprocessing Phase-6 (Feature Engineering)'''
#Part-1(Number of times a film has been nominated)
df["film_nom_count"] = df.groupby(["Film","Year_clean"])['Year_clean'].transform('count')
print(df['Film','Ceremony','film_nom_count'].head(20))
print(f"\nMax nominations in a single film: {df['film_nom_count'].max()}")
#Part-2(Winning rate of a category)
df['category_win_rate'] = df.groupby(['CanonicalCategory'])['Winner'].transform('mean')
print(df['category_win_rate'].head(20))