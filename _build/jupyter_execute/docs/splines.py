#!/usr/bin/env python
# coding: utf-8

# # Splines in Python

# The following code tutorial is mainly based on code provided by[Jordi Warmenhoven](https://nbviewer.org/github/JWarmenhoven/ISL-python/blob/master/Notebooks/Chapter%207.ipynb). To learn more about the regression methods, review ["An Introduction to Statistical Learning"](https://www.statlearning.com/) from James et al. (2021).

# ## Data

# ### Import

# In[1]:


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/kirenz/datasets/master/wage.csv')
df


# ### Create label and feature
# 
# We only use the feature age to predict wage:

# In[2]:


X = df[['age']]
y = df[['wage']]


# ### Data split

# Dividing data into train and test datasets

# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# ### Data exploration
# 
# Visualize the relationship between age and wage:

# In[4]:


import seaborn as sns  

# seaborn settings
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# plot
sns.scatterplot(x=X_train['age'], y=y_train['wage'], alpha=0.4);


# ## Simple regression

# In[5]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)


# In[6]:


print(lm.coef_)
print(lm.intercept_)


# In[7]:


from sklearn.metrics import mean_squared_error

# Training data
pred_train = lm.predict(X_train)
rmse_train = mean_squared_error(y_train, pred_train, squared=False)

# Test data
pred_test = lm.predict(X_test)
rmse_test =mean_squared_error(y_test, pred_test, squared=False)

# Save model results
model_results_lm = pd.DataFrame(
    {
    "model": "Linear Model (lm)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test],
    })
model_results_lm


# In[8]:


sns.regplot(x=X_train['age'], 
            y=y_train['wage'], 
            ci=None, 
            line_kws={"color": "orange"});


# ## Polynomial regression

# In[9]:


from sklearn.preprocessing import PolynomialFeatures

# polynomial degree 2
poly = PolynomialFeatures(2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


# In[10]:


pm = LinearRegression()
pm.fit(X_train_poly,y_train)


# In[11]:


# Training data
pred_train = pm.predict(X_train_poly)
rmse_train = mean_squared_error(y_train, 
                                pred_train, 
                                squared=False)

# Test data
pred_test = pm.predict(X_test_poly)
rmse_test =mean_squared_error(y_test, 
                              pred_test, 
                              squared=False)

# Save model results
model_results_pm = pd.DataFrame(
    {
    "model": "Polynomial Model (pm)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test],
    })

results = pd.concat([model_results_lm, model_results_pm], axis=0)
results


# In[12]:


# plot
sns.regplot(x=X_train['age'], 
            y=y_train['wage'], 
            ci=None, 
            order=2, 
            line_kws={"color": "orange"});


# ## Cubic spline
# 
# We use the module [patsy](https://patsy.readthedocs.io/en/latest/overview.html) to create non-linear transformations of the input data. We will fit 2 models with different number of knots.

# In[13]:


from patsy import dmatrix


# In[14]:


# Generating cubic spline with 3 knots at 25, 40 and 60
transformed_x = dmatrix(
            "bs(train, knots=(25,40,60), degree=3, include_intercept=False)", 
                {"train": X_train},return_type='dataframe')


# In[15]:


transformed_x.head()


# We use statsmodels to estimate a generalized linear model:

# In[16]:


import statsmodels.api as sm


# In[17]:


# Fitting generalised linear model on transformed dataset
cs = sm.GLM(y_train, transformed_x).fit()


# In[18]:


# Training data
pred_train = cs.predict(dmatrix("bs(train, knots=(25,40,60), include_intercept=False)", {"train": X_train}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train, pred_train, squared=False)

# Test data
pred_test = cs.predict(dmatrix("bs(test, knots=(25,40,60), include_intercept=False)", {"test": X_test}, return_type='dataframe'))
rmse_test =mean_squared_error(y_test, pred_test, squared=False)

# Save model results
model_results_cs = pd.DataFrame(
    {
    "model": "Cubic spline (cs)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test]
    })
results = pd.concat([results, model_results_cs], axis=0)
results


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# Create observations
xp = np.linspace(X_test.min(),X_test.max(), 100)
# Make some predictions
pred = cs.predict(dmatrix("bs(xp, knots=(25,40,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# plot
sns.scatterplot(x=X_train['age'], y=y_train['wage'])

plt.plot(xp, pred, label='Cubic spline with degree=3 (3 knots)', color='orange')
plt.legend();


# ## Natural cubic spline

# In[20]:


transformed_x3 = dmatrix("cr(train,df = 3)", {"train": X_train}, return_type='dataframe')

ncs = sm.GLM(y_train, transformed_x3).fit()


# In[21]:


# Training data
pred_train = ncs.predict(dmatrix("cr(train, df=3)", {"train": X_train}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train, pred_train, squared=False)

# Test data
pred_test = ncs.predict(dmatrix("cr(test, df=3)", {"test": X_test}, return_type='dataframe'))
rmse_test =mean_squared_error(y_test, pred_test, squared=False)

# Save model results
model_results_ncs = pd.DataFrame(
    {
    "model": "Natural cubic spline (ncs)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test]
    })

results = pd.concat([results, model_results_ncs], axis=0)
results


# In[22]:


# Make predictions
pred = ncs.predict(dmatrix("cr(xp, df=3)", {"xp": xp}, return_type='dataframe'))

# plot
sns.scatterplot(x=X_train['age'], y=y_train['wage'])
plt.plot(xp, pred, color='orange', label='Natural spline with df=3')
plt.legend();

