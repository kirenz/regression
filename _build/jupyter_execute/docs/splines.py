#!/usr/bin/env python
# coding: utf-8

# # Splines in Python

# The following tutorial is mainly based on examples from {cite:p}`James2021` and Python code from [Jordi Warmenhoven](https://nbviewer.org/github/JWarmenhoven/ISL-python/blob/master/Notebooks/Chapter%207.ipynb).

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


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(X_train, y_train, facecolor='None', edgecolor='k', alpha=0.3)
plt.show()


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
    }
)
model_results_lm


# ## Polynomial regression

# In[8]:


from sklearn.preprocessing import PolynomialFeatures

# polynomial degree 2
poly = PolynomialFeatures(2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


# In[9]:


pm = LinearRegression()
pm.fit(X_train_poly,y_train)


# In[10]:


# Training data
pred_train = pm.predict(X_train_poly)
rmse_train = mean_squared_error(y_train, pred_train, squared=False)

# Test data
pred_test = pm.predict(X_test_poly)
rmse_test =mean_squared_error(y_test, pred_test, squared=False)

# Save model results
model_results_pm = pd.DataFrame(
    {
    "model": "Polynomial Model (pm)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test],
    }
)

results = pd.concat([model_results_lm, model_results_pm], axis=0)

results


# ## Cubic spline
# 
# We use the module [patsy](https://patsy.readthedocs.io/en/latest/overview.html) to create non-linear transformations of the input data. We will fit 2 models with different number of knots.

# In[11]:


from patsy import dmatrix


# In[12]:


# Generating cubic spline with 3 knots at 25, 40 and 60
transformed_x = dmatrix(
            "bs(train, knots=(25,40,60), degree=3, include_intercept=False)", 
                {"train": X_train},return_type='dataframe')


# In[13]:


transformed_x.head()


# We use statsmodels to estimate a generalized linear model:

# In[14]:


import statsmodels.api as sm


# In[15]:


# Fitting Generalised linear model on transformed dataset
cs = sm.GLM(y_train, transformed_x).fit()


# In[16]:


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
    }
)
results = pd.concat([results, model_results_cs], axis=0)
results


# In[17]:


import numpy as np

# We will plot the graph for 100 observations 
xp = np.linspace(X_test.min(),X_test.max(), 100)

# Make some predictions
pred = cs.predict(dmatrix("bs(xp, knots=(25,40,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# Plot the splines and error bands
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)

plt.plot(xp, pred, label='Cubic spline with degree=3 (3 knots)')
plt.legend()
plt.xlabel('age')
plt.ylabel('wage')
plt.show()


# ## Natural cubic spline

# In[18]:


# Generating natural cubic spline with df=3
transformed_x3 = dmatrix("cr(train,df = 3)", {"train": X_train}, return_type='dataframe')

ncs = sm.GLM(y_train, transformed_x3).fit()


# In[19]:


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
    }
)

results = pd.concat([results, model_results_ncs], axis=0)
results


# In[20]:


# We will plot the graph for 100 observations only
xp = np.linspace(X_test.min(),X_test.max(),100)
pred = ncs.predict(dmatrix("cr(xp, df=3)", {"xp": xp}, return_type='dataframe'))

# Plot the spline
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(xp, pred,color='g', label='Natural spline')

plt.legend()
plt.xlabel('age')
plt.ylabel('wage')
plt.show()


# ## Natural cubic spline with scipy

# <!--BOOK_INFORMATION-->
# 
# *This notebook contains an excerpt from the [Python Programming and Numerical Methods - A Guide for Engineers and Scientists](https://www.elsevier.com/books/python-programming-and-numerical-methods/kong/978-0-12-819549-9), the content is also available at [Berkeley Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html).*

# In[21]:


from scipy.interpolate import CubicSpline
plt.style.use('seaborn-poster')


# In[22]:


# make simple dataset
x = [0, 1, 2]
y = [1, 3, 2]


# In[23]:


# use bc_type = 'natural'
natural_spline = CubicSpline(x, y, bc_type='natural')


# In[24]:


x_new = np.linspace(0, 2, 100)
y_new = natural_spline(x_new)


# In[25]:


plt.figure(figsize = (10,8))
plt.plot(x_new, y_new, 'b')
plt.plot(x, y, 'ro')
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

