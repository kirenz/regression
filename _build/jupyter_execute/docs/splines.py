#!/usr/bin/env python
# coding: utf-8

# # Splines

# The following code tutorial is mainly based on:
# 
# - the [scikit learn documentation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py) about splines provided by Mathieu Blondel, Jake Vanderplas, Christian Lorentzen and Malte Londschien. 
# - code from [Jordi Warmenhoven](https://nbviewer.org/github/JWarmenhoven/ISL-python/blob/master/Notebooks/Chapter%207.ipynb)
# 
# To learn more about the spline regression methods, review ["An Introduction to Statistical Learning"](https://www.statlearning.com/) from James et al. (2021).

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


# ## Ridge regression

# In[5]:


from sklearn.linear_model import Ridge

reg = Ridge()
reg.fit(X_train,y_train)


# In[6]:


print(reg.coef_)
print(reg.intercept_)


# In[7]:


from sklearn.metrics import mean_squared_error

# create function to obtain model mse
def model_results(model_name):

    # Training data
    pred_train = reg.predict(X_train)
    rmse_train = round(mean_squared_error(y_train, pred_train, squared=False),4)

    # Test data
    pred_test = reg.predict(X_test)
    rmse_test =round(mean_squared_error(y_test, pred_test, squared=False),4)

    # Print model results
    result = pd.DataFrame(
        {"model": model_name, 
        "rmse_train": [rmse_train], 
        "rmse_test": [rmse_test]}
        )
    
    return result;


# In[8]:


model_results(model_name="ridge")


# In[9]:


sns.regplot(x=X_train['age'], 
            y=y_train['wage'], 
            ci=None, 
            line_kws={"color": "orange"});


# ## Polynomial regression
# 
# Next, we use a pipeline to add non-linear features to a ridge regression model:

# In[10]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# use polynomial features with degree 3
reg = make_pipeline(PolynomialFeatures(degree=2), 
                      Ridge())

reg.fit(X_train, y_train)


# In[11]:


model_results(model_name="poly")


# In[12]:


# plot
sns.regplot(x=X_train['age'], 
            y=y_train['wage'], 
            ci=None, 
            order=2, 
            line_kws={"color": "orange"});


# ## Splines 
# 
# ### Splines in Scikit learn
# 
# Spline transformers are a new feature in [scikit learn 1.0](https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_0_0.html). Therefore, make sure to use the latest version of scikit learn. If you use Aanconda, you can update all packages using `conda update --all`  

# In[13]:


from sklearn.preprocessing import SplineTransformer

# use a spline wit 4 knots and 3 degrees with a ridge regressions
reg = make_pipeline(SplineTransformer(n_knots=4, degree=3), 
                       Ridge(alpha=1))
                     
reg.fit(X_train, y_train)

y_pred = reg.predict(X_train)


# In[14]:


model_results(model_name = "spline")


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

# Create observations
x_new = np.linspace(X_test.min(),X_test.max(), 100)
# Make some predictions
pred = reg.predict(x_new)

# plot
sns.scatterplot(x=X_train['age'], y=y_train['wage'])

plt.plot(x_new, pred, label='Cubic spline with degree=3', color='orange')
plt.legend();


# In some settings, e.g. in time series data with seasonal effects, we expect a periodic continuation of the underlying signal. Such effects can be modelled using periodic splines, which have equal function value and equal derivatives at the first and last knot. Review this notebook to learn more about periodic splines in scikit learn:
# 
# - [periodic splines](https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#periodic-splines)
# 
# 

# ### Splines in patsy
# 
# Next, we use the module [patsy](https://patsy.readthedocs.io/en/latest/overview.html) to create non-linear transformations of the input data. Additionaly, we use statsmodels to fit 2 models with different number of knots.

# In[16]:


from patsy import dmatrix


# In[17]:


# Generating cubic spline with 3 knots at 25, 40 and 60
transformed_x = dmatrix(
            "bs(train, knots=(25,40,60), degree=3, include_intercept=False)", 
                {"train": X_train},return_type='dataframe')


# We use statsmodels to estimate a generalized linear model:

# In[18]:


import statsmodels.api as sm


# In[19]:


# Fitting generalised linear model on transformed dataset
reg = sm.GLM(y_train, transformed_x).fit()


# In[20]:


# Training data
pred_train = reg.predict(dmatrix("bs(train, knots=(25,40,60), include_intercept=False)", {"train": X_train}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train, pred_train, squared=False)

# Test data
pred_test = reg.predict(dmatrix("bs(test, knots=(25,40,60), include_intercept=False)", {"test": X_test}, return_type='dataframe'))
rmse_test =mean_squared_error(y_test, pred_test, squared=False)

# Save model results
model_results = pd.DataFrame(
    {
    "model": "Cubic spline (cs)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test]
    })

model_results


# In[21]:


import numpy as np
import matplotlib.pyplot as plt

# Create observations
xp = np.linspace(X_test.min(),X_test.max(), 100)
# Make some predictions
pred = reg.predict(dmatrix("bs(xp, knots=(25,40,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# plot
sns.scatterplot(x=X_train['age'], y=y_train['wage'])

plt.plot(xp, pred, label='Cubic spline with degree=3 (3 knots)', color='orange')
plt.legend();


# ## Natural spline
# 
# Finally, we fit a natural spline with patsy and statsmodels.

# In[22]:


transformed_x3 = dmatrix("cr(train,df = 3)", {"train": X_train}, return_type='dataframe')

reg = sm.GLM(y_train, transformed_x3).fit()


# In[23]:


# Training data
pred_train = reg.predict(dmatrix("cr(train, df=3)", {"train": X_train}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train, pred_train, squared=False)

# Test data
pred_test = reg.predict(dmatrix("cr(test, df=3)", {"test": X_test}, return_type='dataframe'))
rmse_test = mean_squared_error(y_test, pred_test, squared=False)

# Save model results
model_results_ns = pd.DataFrame(
    {
    "model": "Natural spline (ns)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test]
    })

model_results_ns


# In[24]:


# Make predictions
pred = reg.predict(dmatrix("cr(xp, df=3)", {"xp": xp}, return_type='dataframe'))

# plot
sns.scatterplot(x=X_train['age'], y=y_train['wage'])
plt.plot(xp, pred, color='orange', label='Natural spline with df=3')
plt.legend();

