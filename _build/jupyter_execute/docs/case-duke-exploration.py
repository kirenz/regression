#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Setup" data-toc-modified-id="Setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Setup</a></span></li><li><span><a href="#Import-data" data-toc-modified-id="Import-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import data</a></span></li><li><span><a href="#Data-inspection" data-toc-modified-id="Data-inspection-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data inspection</a></span></li><li><span><a href="#Data-transformation" data-toc-modified-id="Data-transformation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data transformation</a></span></li><li><span><a href="#Data-splitting" data-toc-modified-id="Data-splitting-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Data splitting</a></span></li><li><span><a href="#Exploratory-data-analysis" data-toc-modified-id="Exploratory-data-analysis-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Exploratory data analysis</a></span></li><li><span><a href="#Correlation-analysis" data-toc-modified-id="Correlation-analysis-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Correlation analysis</a></span></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Modeling</a></span></li></ul></div>

# # Data exploration

# ## Setup

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant  

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
categorical_list = ['type', 'heating', 'cooling', 'parking']

for i in categorical_list:
    df[i] = df[i].astype("category")


# In[11]:


df.info()


# In[12]:


# summary statistics for all categorical columns
df.describe(include=['category']).transpose()


# - Variable `type` has zero veriation (only single family) and therefore can be exluded from the analysis and the model. 
# 
# - We will also exclude `heating`and `parking` to keep this example as simple as possible.

# In[13]:


df = df.drop(['type', 'heating', 'parking'], axis=1)
df


# ## Data splitting

# In[14]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_dataset


# ## Exploratory data analysis

# In[15]:


# summary statistics for all numerical columns
round(train_dataset.describe(),2).transpose()


# In[16]:


sns.pairplot(train_dataset);


# ## Correlation analysis

# In[17]:


# Create correlation matrix for numerical variables
corr_matrix = train_dataset.corr()
corr_matrix


# In[18]:


# Simple heatmap
heatmap = sns.heatmap(corr_matrix)


# In[19]:


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


# Instead of inspecting the correlation matrix, a better way to assess **multicollinearity** is to compute the variance inflation factor (VIF). Note that we ignore the intercept in this test.
# 
# - The smallest possible value for VIF is 1, which indicates the complete absence of collinearity. 
# - Typically in practice there is a small amount of collinearity among the predictors. 
# - As a rule of thumb, a VIF value that exceeds 5 indicates a problematic amount of collinearity and the parameter estimates will have large standard errors because of this. 
# 
# Note that the function `variance_inflation_factor` expects the presence of a constant in the matrix of explanatory variables. Therefore, we use `add_constant` from statsmodels to add the required constant to the dataframe before passing its values to the function.

# In[20]:


# choose features and add constant
features = add_constant(df[['bed', 'bath', 'area', 'lot']])
# create empty DataFrame
vif = pd.DataFrame()
# calculate vif
vif["VIF Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
# add feature names
vif["Feature"] = features.columns

vif.round(2)


# We don't have a problematic amount of collinearity in our data.

# ## Modeling
# 
# See separate notebooks.
