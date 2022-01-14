#!/usr/bin/env python
# coding: utf-8

# # Hitters data preparation

# We illustrate the following regression methods on a data set called "Hitters", which includes 20 variables and 322 observations of major league baseball players. The goal is to predict a baseball player’s salary on the basis of various features associated with performance in the previous year. We don't cover the topic of exploratory data analysis in this notebook. 
# 
# - Visit [this documentation](https://cran.r-project.org/web/packages/ISLR/ISLR.pdf) if you want to learn more about the data
# 
# Note that scikit-learn provides a [**pipeline**](https://kirenz.github.io/ds-python/docs/data.html#pipelines-in-scikit-learn
# ) library for data preprocessing and feature engineering, which is considered best practice for data preparation. However, since we use scikit-learn as well as statsmodels in some of our examples, we won't create a data prerocessing pipeline in this example.
# 
# ## Import

# In[1]:


import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/Hitters.csv")


# In[2]:


df


# In[3]:


df.info()


# ### Missing values
# 
# Note that the salary is missing for some of the players:

# In[4]:


print(df.isnull().sum())


# We simply drop the missing cases: 

# In[5]:


# drop missing cases
df = df.dropna()


# ## Create label and features
# 
# Since we will use algorithms from scikit learn, we need to encode our categorical features as one-hot numeric features (dummy variables):

# In[6]:


dummies = pd.get_dummies(df[['League', 'Division','NewLeague']])


# In[7]:


dummies.info()


# In[8]:


print(dummies.head())


# Next, we create our label y:

# In[9]:


y = df['Salary']


# We drop the column with the outcome variable (Salary), and categorical columns for which we already created dummy variables:

# In[10]:


X_numerical = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')


# Make a list of all numerical features (we need them later):

# In[11]:


list_numerical = X_numerical.columns
list_numerical


# In[12]:


# Create all features
X = pd.concat([X_numerical, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X.info()


# ### Split data

# Split the data set into train and test set with the first 70% of the data for training and the remaining 30% for testing.

# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[14]:


X_train.head()

