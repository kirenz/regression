#!/usr/bin/env python
# coding: utf-8

# # Statsmodels

# ## Setup

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import mse, rmse

sns.set_theme(style="ticks", color_codes=True)


# ## Data preparation

# In[2]:


# See notebook "Data Exploration" for details about data preprocessing
from case_duke_data_prep import *


# ## Data splitting

# In[3]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)


# ## Modeling

# ### Train the model

# In[4]:


# Fit Model
lm = smf.ols(formula='price ~ area', data=train_dataset).fit()


# In[5]:


# Short summary
lm.summary().tables[1]


# In[6]:


# Full summary
lm.summary()


# To obtain single statistics:

# In[7]:


# Adjusted R squared 
lm.rsquared_adj


# In[8]:


# R squared
lm.rsquared


# In[9]:


# AIC
lm.aic


# In[10]:


train_dataset.info()


# In[11]:


# Add the regression predictions (as "pred") to our DataFrame
train_dataset['y_pred'] = lm.predict()


# In[12]:


# MSE
mse(train_dataset['price'], train_dataset['y_pred'])


# In[13]:


# RMSE
rmse(train_dataset['price'], train_dataset['y_pred'])


# In[14]:


# Plot regression line 
plt.scatter(train_dataset['area'], train_dataset['price'],  color='black')
plt.plot(train_dataset['area'], train_dataset['y_pred'], color='darkred', linewidth=3);


# In[15]:


# Plot with Seaborn

import seaborn as sns 
sns.set_theme(style="ticks")

sns.lmplot(x='area', y='price', data=train_dataset, line_kws={'color': 'darkred'}, ci=False);


# In[16]:


sns.residplot(x="y_pred", y="price", data=train_dataset, scatter_kws={"s": 80});


# ### Validation with test data

# In[17]:


# Add regression predictions for the test set (as "pred_test") to our DataFrame
test_dataset['y_pred'] = lm.predict(test_dataset['area'])


# In[18]:


test_dataset.head()


# In[19]:


# Plot regression line 
plt.scatter(test_dataset['area'], test_dataset['price'],  color='black')
plt.plot(test_dataset['area'], test_dataset['y_pred'], color='darkred', linewidth=3);


# In[20]:


sns.residplot(x="y_pred", y="price", data=test_dataset, scatter_kws={"s": 80});


# In[21]:


# RMSE
rmse(test_dataset['price'], test_dataset['y_pred'])


# ## Multiple regression

# In[22]:


lm_m = smf.ols(formula='price ~ area + bed + bath + year_built + cooling + lot', data=train_dataset).fit()


# In[23]:


lm_m.summary()

