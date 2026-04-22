import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split

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
df['Year_clean'] = df['Year_clean'].astype(int)
df['Decade'] = (df['Year_clean'] // 10) * 10

'''Preprocessing Phase-5 (Handle remaining missing values)'''
df['Film'] = df["Film"].fillna('Unknown')
df['Name'] = df["Name"].fillna('Unknown')

'''Preprocessing Phase-6 (Feature Engineering)'''
#Part-1(Number of times a film has been nominated)
df["film_nom_count"] = df.groupby(["Film","Year_clean"])['Year_clean'].transform('count')
print(df[['Film','Ceremony','film_nom_count']].head(20))
print(f"\nMax nominations in a single film: {df['film_nom_count'].max()}")
#Part-2(Winning rate of a category)
df['category_win_rate'] = df.groupby(['CanonicalCategory'])['Winner'].transform('mean')
print(df['category_win_rate'].head(20))

'''Preprocessing Phase-7 (Encoding Categorical Values)'''
class_encoder = OneHotEncoder(sparse_output=False)
class_encoded_df = pd.DataFrame(
    class_encoder.fit_transform(df[['Class']]),
    columns=class_encoder.get_feature_names_out(['Class']),
    index = df.index
)
df = pd.concat([df,class_encoded_df],axis=1)
cat_encoder = LabelEncoder()
df['CanonicalCategory_encoded'] = pd.Series(
    cat_encoder.fit_transform(df['CanonicalCategory']),
    index=df.index  
)

'''Define Final Feature Set'''
feature_columns = [
    'Ceremony', 'Year_clean', 'Decade',
    'film_nom_count', 'category_win_rate',
    'CanonicalCategory_encoded',
    'Class_Acting', 'Class_Directing', 'Class_Music',
    'Class_Production', 'Class_Title', 'Class_Writing'
]

x = df[feature_columns]
y = df['Winner']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

print(f"Shape of x train: {x_train.shape}")
print(f"Shape of x test: {x_test.shape}")
print(f"Value counts of y train: {y_train.value_counts()}")
train_df = x_train.copy()
train_df['Winner'] = y_train.values
train_df['Class'] = df.loc[x_train.index, 'Class']           # original text column
train_df['CanonicalCategory'] = df.loc[x_train.index, 'CanonicalCategory']  # useful later
train_df['Film'] = df.loc[x_train.index, 'Film']             # useful later
nominations_per_decade = train_df.groupby('Decade')['Winner'].count()
win_rate_decade = train_df.groupby('Decade')['Winner'].mean()

'''Exploratory Data Analysis'''

#Target Variable Analysis
sns.countplot(x=y_train,palette=["#AF1577","#151AAD"])
plt.title("Winner count")
plt.xlabel("Winner (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

sns.barplot(x=train_df["Class"],y=y_train)
plt.title("Win rate by class")
plt.xlabel("Class")
plt.ylabel("Winner (0 = No, 1 = Yes)")
plt.show()

#Time Trends
sns.lineplot(x=train_df['Decade'],y=nominations_per_decade)
plt.title("Number of Nominations per decade")
plt.xlabel("Decade")
plt.ylabel("Number of nominations")
plt.show()

sns.lineplot(x=win_rate_decade.index, y=win_rate_decade.values)
plt.title("Win-Rate per decade")
plt.xlabel("Decade")
plt.ylabel("Win-Rate")
plt.show()
