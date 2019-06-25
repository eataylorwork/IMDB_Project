#%% [markdown]
#**IMDB dataset project**
#
# TODO: Get ready for GitHub by making the file download the data or check if the data is in the folder.
#
# TODO: Consider creating a version that doesn't use the notebook style.
#
# TODO: Continue project.
#
#%% Change working directory from the workspace root to the ipynb file location.
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'C:/Users/Elliot/Desktop/PythonProjects/Portfolio/IMDB'))
	print(os.getcwd())
except:
	pass

#%%
#Import libraries needed.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import numpy as np
import pandas as pd
import time
import seaborn as sns
import math as mt
from scipy import stats
from functools import reduce
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#%%
try:
    df_imdb = pd.read_csv('df_imdb.csv.gz', sep='|')
    print('File found loaded to dataframe.')
except:
    print('File not found loading files to create dataframe.')
    df0 = pd.read_csv('Data/akas.tsv', sep='\t')
    df1 = pd.read_csv('Data/basics.tsv', sep='\t')
    df2 = pd.read_csv('Data/ratings.tsv', sep='\t')

    df0 = df0.rename(columns={'titleId':'tconst'})

    dfs = [df0, df1, df2]

    df_imdb = reduce(lambda left,right: pd.merge(left,right,on='tconst'), dfs)

    df_imdb.to_csv('df_imdb.csv.gz'
            , sep='|'
            , header=True
            , index=False
            , chunksize=100000
            , compression='gzip'
            , encoding='utf-8')

#%% [markdown]
# This gave me problems as originally the isOriginalTitle was of dtype object because of a missing value whereas it should be and int value.
#
# This stopped me from correctly using isOriginalTitle to index the dataframe for only original titles. I solved this by removing the missing value
# and converting the column to dtype int which allowed me to index correctly.
#
# As there was only one missing value in the column I decided to simply drop the row.
df_imdb = pd.DataFrame(df_imdb[df_imdb.isOriginalTitle != '\\N'])

df_imdb['isOriginalTitle'] = pd.to_numeric(df_imdb['isOriginalTitle'])

df_imdb.info()

df_originals = df_imdb.loc[df_imdb.isOriginalTitle == 1]


#%%
#Replace the numerous different TV title types to one single feature to reduce complexity, if model shows underfitting can be put back in.
#Possible idea for feature reduction. 

df_originals.head()

df_originals['titleType'].value_counts()

df_orig = df_originals.copy()

tv_list = ['tvSeries','tvMovie','tvEpisode','tvMiniSeries', 'tvSpecial', 'tvShort']

df_orig = df_orig.replace(tv_list, 'tv')

df_orig['titleType'].value_counts()

#%% [markdown]
#To make this notebook's output identical at every run
np.random.seed(42)

train_set, test_set = train_test_split(df_originals, test_size=0.15, random_state=42)

#%% [markdown]
#Split the dataset to start data exploration

print(len(train_set), "train +", len(test_set), "test")

#%%
imdb = train_set.copy()


#%%
for c in imdb.columns:
    print ("---- %s ---" % c)
    print (imdb[c].value_counts())

#%%
imdb = imdb.replace('\\N', np.NaN)

#%% [markdown]

#**Data Exploration results**
#
#For end year all none TV values are /N meaning we can impute them with the same year or add 1 year.
#
#Tconst, ordering, isOriginaltitle and all title attributes give no information
#
#Region, language, attributes, have lots of missing values possibly too many to fix.
#

imdb = imdb.drop(['tconst', 'ordering', 'isOriginalTitle', 'region', 'language', 'attributes'], axis=1)


#%% [markdown]
# From info we can see that some of the columns are of the wrong dtype this makes data exploration difficult and must be fixed first.
# I will try to write functions for this so it can be automated into the pipeline at the end. This will also help to make edits later on.
#
# From .info() we can see that the following are miss typed: startYear, endYear, and runtimeMinutes.
# This is most likely due to missing values being entered as '/N'
imdb.info()







#%% 
# These lines fix the endYear and startYear columns
#TODO Check if this works.
# startYear, endYear

imdb = imdb.dropna(axis=0,subset=['startYear'])

imdb['endYear'] = imdb['endYear'].fillna(imdb['startYear'], axis=0)

imdb['startYear'] = pd.to_numeric(imdb['startYear'], downcast='integer')

imdb['endYear'] = pd.to_numeric(imdb['endYear'], downcast='integer')

np.isnan(imdb['startYear']).sum()

#%% [markdown]
# startYear only has ten missing values these could be dropped from the dataframe with little impact however as the majority of endYear and 
# runetimeMinutes are missing values I will fix the missing values first before working on the dtypes and then move onto full data exploration.
imdb.info()










#%%
# runtime
items = imdb.titleType.unique()
for i in items:
    print(i)
    print(imdb['runtimeMinutes'].loc[imdb['titleType'] == i])

#%%
# This imputes the values of a column based on the mean values of that column for each type.
#
# In this case it takes the mean for the runtimeMinutes based on each different title type for example TV or movie, it then sets all missing
# values of that type in the column to the mean. This means the values better represent the values for that type without having to remove those rows.
def imputeByType(df, typeCol, valCol):
    types = df[typeCol].unique()
    print(types)
    means=[]
    for i in types:
        mean = imdb[valCol].loc[imdb[typeCol] == i].mean()
        mean = mean
        print(i + ':')
        print(mean)
        imdb[valCol].loc[imdb[typeCol] == i] = imdb[valCol].loc[imdb[typeCol] == i].fillna(mean, axis=0)
        means.append(mean)
    return means

imdb['runtimeMinutes'] = pd.to_numeric(imdb['runtimeMinutes'])
imputeByType(imdb, 'titleType', 'runtimeMinutes')

#%%
imdb.info()











#%%
# genres
# Currently the genres column does not work as some of the rows have no value, some have only one, and others have multiple comma
# separated values these can be turned into dummy variables. 

imdb['genres'].value_counts()

dummy_genres = pd.Series(imdb['genres']).str.get_dummies(sep=',')

dummy_genres

# This works I think. Yay!

#%%
imdb.info()














#%%
# averageRating
corr_matrix = imdb.corr()
corr_matrix["averageRating"].sort_values(ascending=False)

#%%
attributes = ["numVotes", "startYear" ,"endYear" ,"runtimeMinutes"]

for attribute in attributes:
    imdb[attribute] = imdb[attribute].astype(int)
    imdb[attribute+'Log'] = np.log1p(imdb[attribute])

#%%
from pandas.plotting import scatter_matrix
#scatter_matrix(imdb[attributes], figsize=(12, 8))

# Looking at the scatter graphs actually shows me an outlier in runtimeMinutes which I will remove before continuing
for attribute in attributes:
    #imdb.plot(kind='scatter', x='averageRating', y=attribute, alpha=0.2)
    sns.jointplot(y=imdb[attribute], x=imdb['averageRating'], data=imdb, alpha=0.1)