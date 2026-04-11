import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the dataset
data = pd.read_csv('full_data.csv')
df = pd.DataFrame(data)

#Preprocessing Phase-1
drop_columns = ['NomId', 'FilmId', 'NomineeIds',
    'Category', 'Nominees',
    'Detail', 'Note', 'Citation', 'MultifilmNomination']
df.drop(columns = drop_columns, axis = 1)