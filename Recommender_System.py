
# coding: utf-8



# In[136]:

import numpy as np
import pandas as pd


# ## Get the Data

# In[137]:

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)


# In[138]:

df.head()


# Now let's get the movie titles:

# In[139]:

movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()


# We can merge them together:

# In[140]:

df = pd.merge(df,movie_titles,on='item_id')
df.head()


# # EDA
# 
# 
# ## Visualization Imports

# In[160]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().magic('matplotlib inline')


# In[142]:

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[143]:

df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[144]:

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()



# In[159]:

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()



# In[146]:

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# In[147]:

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[148]:

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)



# In[149]:

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# Most rated movie:

# In[150]:

ratings.sort_values('num of ratings',ascending=False).head(10)


# Let's choose two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.

# In[161]:

ratings.head()



# In[162]:

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()



# In[163]:

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)



# In[164]:

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()



# In[155]:

corr_starwars.sort_values('Correlation',ascending=False).head(10)



# In[165]:

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()



# In[157]:

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()



# In[158]:

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


