#!/usr/bin/env python
# coding: utf-8

# # Generalized Additive Models (GAM)
# 
# *The following code tutorial is mainly based on the [statsmodels documentation](https://www.statsmodels.org/v0.10.0/gam.html) about generalized additive models (GAM). To learn more about this method, review ["An Introduction to Statistical Learning"](https://www.statlearning.com/) from {cite:t}`James2021`. GAMs were originally developed by Trevor Hastie and Robert Tibshirani (who are two coauthors of {cite:t}`James2021`) to blend properties of generalized linear models with additive models.*
# 
# Generalized additive models allow us to use regression splines, smoothing splines and local regression to deal with multiple predictors. In [](splines.ipynb), we discussed regression splines, which we created by specifying a set of knots, producing a sequence of basis functions, and then using least squares to estimate the spline coefficients. 
# 
# In this tutorial, we use a Generalized Additive Model with a reguralized estimation of [smooth components using B-Splines](https://www.statsmodels.org/stable/generated/statsmodels.gam.smooth_basis.BSplines.html#statsmodels.gam.smooth_basis.BSplines). We could also use additive smooth components using [cyclic cubic regression splines](https://www.statsmodels.org/stable/generated/statsmodels.gam.smooth_basis.CyclicCubicSplines.html#statsmodels.gam.smooth_basis.CyclicCubicSplines).

# ## Data preparation
# 
# See [](hitters_data.ipynb) for details about data preprocessing.

# In[1]:


from hitters_data import *


# In[2]:


df_train


# ## Spline basis
# 
# First we have to create a basis spline ("B-Spline"). Here, we select only two features to demonstrate the procedere: `CRuns` and `Hits`.

# In[3]:


# choose features
x_spline = df_train[['CRuns', 'Hits']]


# In[4]:


import seaborn as sns

sns.scatterplot(x='CRuns', y='Salary', data=df_train);


# In[5]:


sns.scatterplot(x='Hits', y='Salary', data=df_train);


# Now we need to divide the range of X into K distinct regions (for every feature). Within each region, a polynomial function is fit to the data. 
# 
# Instead of providing the number of knots, in statsmodels, we have to specify the degrees of freedom (df). `df` defines how many parameters we have to estimate. They have a specific relationship with the number of knots and the degree, which depends on the type of spline (see [Stackoverflow](https://stats.stackexchange.com/a/517479)):
# 
# In the case of **B-splines**: 
# 
# - $df=𝑘+degree$ if you specify the knots or 
# - $𝑘=df−degree$ if you specify the degrees of freedom and the degree. 
# 
# As an example: 
# 
# - A cubic spline (degree=3) with 4 knots (K=4) will have $df=4+3=7$ degrees of freedom. If we use an intercept, we need to add an additional degree of freedom.
# - A cubic spline (degree=3) with 5 degrees of freedom (df=5) will have $𝑘=5−3=2$ knots (assuming the spline has no intercept).
# 
# In our case, we want to fit a cubic spline (degree=3) with an intercept and three knots (K=3). We use this values for both of our features (we also could use different values for one of the features). This equals $df=3+3+1=7$ for both of the features. This means that these degrees of freedom are used up by an intercept, plus six basis functions.
# 
# :::{Note}
# The higher the degrees of freedom, the "wigglier" the spline gets because the number of knots is increased {cite:p}`James2021`.
# :::
# 
# In the case of natural splines: $df=𝑘−1$ if you specify the knots or $𝑘=df+1$ if you specify the degrees of freedom.
# 
# In the statsmodels function `BSplines`, we need to provide `df` and `degree`:
# 
# `df`: number of basis functions or degrees of freedom; should be equal in length to the number of columns of x; may be an integer if x has one column or is 1-D.  
# 
# `degree`: degree(s) of the spline; the same length and type rules apply as to df.

# In[6]:


import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

# create basis spline
bs = BSplines(x_spline, df=[7, 7], degree=[3, 3])


# ## Model

# In[7]:


import numpy as np

# we use a penalization weight of 1 for both features
alpha = np.array([1, 1])


# In[8]:


# build model
gam_bs = GLMGam.from_formula('Salary ~ CRuns + Hits', 
                                data=df_train, 
                                smoother=bs, 
                                alpha=alpha)


# Note that optimal penalization weights alpha could be obtained through generalized k-fold cross-validation by using the function [select_penweight_kfold](https://www.statsmodels.org/dev/generated/statsmodels.gam.generalized_additive_model.GLMGam.select_penweight_kfold.html).

# In[9]:


# fit model
res_bs = gam_bs.fit()

# print results
print(res_bs.summary())


# ## Plot

# The results classes provide a `plot_partial` method that plots the partial linear prediction of a smooth component. The partial residual or component plus residual can be added as scatter point with cpr=True.
# 
# Spline for feature `CRuns`:

# In[10]:


res_bs.plot_partial(0, cpr=True)


# Spline for feature `Hits`:

# In[11]:


res_bs.plot_partial(1, cpr=True)


# ## First evaluation

# In[12]:


df_train['y_pred'] = res_bs.predict()


# In[13]:


from statsmodels.tools.eval_measures import mse, rmse

# MSE
print('MSE:', mse(df_train['Salary'], df_train['y_pred']))
print('RMSE:', rmse(df_train['Salary'], df_train['y_pred']))

