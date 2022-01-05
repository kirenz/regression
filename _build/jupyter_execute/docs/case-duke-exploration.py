#!/usr/bin/env python
# coding: utf-8

# # Data exploration

# ## Setup

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

sns.set_theme()


# ## Import data

# In[2]:


ROOT = "https://raw.githubusercontent.com/kirenz/modern-statistics/main/data/"
DATA = "duke-forest.csv"

df = pd.read_csv(ROOT + DATA)


# ## Data inspection

# In[3]:


df


# In[4]:


df.info()


# In[5]:


# show missing values (missing values - if present - will be displayed in yellow)
sns.heatmap(df.isnull(), 
            yticklabels=False,
            cbar=False, 
            cmap='viridis');


# In[6]:


print(df.isnull().sum())


# ## Data transformation

# In[7]:


# drop column with too many missing values
df = df.drop(['hoa'], axis=1)

# drop remaining row with one missing value
df = df.dropna()


# In[8]:


# Drop irrelevant features
df = df.drop(['url', 'address'], axis=1)


# In[9]:


print(df.isnull().sum())


# In[10]:


# Convert data types
df['type'] = df['type'].astype("category")
df['heating'] = df['heating'].astype("category")
df['cooling'] = df['cooling'].astype("category")
df['parking'] = df['parking'].astype("category")


# In[11]:


# summary statistics for all categorical columns
df.describe(include=['category']).transpose()


# - Variable `type` has zero veriation (only single family) and therefore can be exluded from the analysis and the model. 
# 
# - We will also exclude `heating`and `parking` to keep this example as simple as possible.

# In[12]:


df = df.drop(['type', 'heating', 'parking'], axis=1)
df


# ## Data splitting

# In[13]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_dataset


# ## Exploratory data analysis

# In[14]:


# summary statistics for all numerical columns
round(train_dataset.describe(),2).transpose()


# In[15]:


sns.pairplot(train_dataset);


# ## Correlation analysis

# In[16]:


# Create correlation matrix for numerical variables
corr_matrix = train_dataset.corr()
corr_matrix


# In[17]:


# Simple heatmap
heatmap = sns.heatmap(corr_matrix)


# In[18]:


# Make a pretty heatmap

# Use a mask to plot only part of a matrix
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)]= True

# Change size
plt.subplots(figsize=(11, 15))

# Build heatmap with additional options
heatmap = sns.heatmap(corr_matrix, 
                      mask = mask, 
                      square = True, 
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .6,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 10})


# ## Modeling
# 
# See separate notebooks.
