#!/usr/bin/env python
# coding: utf-8

# # Model with simulated data

# In[1]:


# Python set up (load modules) 
import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

import plotly.express as px
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
# seaborn settings
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, palette='winter')


# In this section you will create some simulated data and fit simple linear regression models to it {cite:p}`James2021`. 
# 
# Make sure to use `np.random.seed(123)` prior to starting part (a) to ensure consistent results.
# 
# Task (a) 
# 
# - Create a vector `X`, containing 100 observations drawn from a normal distribution with mean 0 and standard deviation 1. This represents our feature, X. 
# 
# - Furthermore, create a dependent variable `Y_perfect` as follows: `Y_perfect = −1 + 0.5 X`. This is also called a **population model** with known parameters since we "know" the true relationship between X and Y (which usually is not the case).  
# 
# 
# Task (b) 
# 
# - Create an error vector, `err` (error), containing 100 observations drawn from a N(0, 0.25) distribution i.e. a normal distribution with mean zero and standard deviation 0.25.
# 
# Task (c)
# 
# - Using `X` and `err`, generate a new vector `Y` according to the relationship: `Y = −1 + 0.5 X + err` (we call this model 1). 
# - Questions: what is the length of the vector Y? What are the values of $β_0$ and $β_1$ in this linear relationship between Y and X?
# 
# Task (d)
# 
# - Use Pandas to create a DataFrame from `X`, `Y_perfect`, `err` and `Y` (call it `df`) 
# - Make a **scatterplot** displaying the relationship between X and Y (with Seaborn and Plotly Express). 
# - Comment on what you observe.
# 
# Task (e)
# 
# 
# - Fit a **ordinary least squares linear model** (this is our model 2) to predict `Y` using `X` (which will yield $\hat{Y}$). 
# - Comment on the model obtained (use `summary()` and the mean squared error (MSE) of the residuals). 
# - How do $\hat{β_0}$ and $\hat{β_1}$ of model 2 compare to $β_0$ and $β_1$ of model 1 in (c)?
# 
# Task (f)
# 
# - Display the OLS regression for X and Y in a scatterplot with the color red (use Plotly Express). 
# 
# Task (g)
# 
# - Now fit a polynomial regression model that predicts Y using X and $X^2$. 
# - Is there evidence that the quadratic term improves the model fit? 
# - Explain your answer.
# 
# Task (h)
# 
# - Repeat (a)–(f) after modifying the data generation process in such a way that there is **less noise** in the data (only use Seaborn for your plots):
#     - You can do this by decreasing the standard deviation of the normal distribution used to generate the **err** (error) term in (b) 
#     - Use sd = 0.05. 
# - Describe your results.
# 
# Task (i)
# 
# -  What are the confidence intervals for $β_0$ and $β_1$ based on 
#     - the original data set (used in e) and 
#     - the less noisy data set (created in h)? 
# - Comment on your results.
# 
# Task (j)
# 
# - What are the standard errors for $β_0$ and $β_1$ based on 
#     - the original data set (used in e) and 
#     - the less noisy data set (created in h)? 
# - Comment on your results.

# # Solution
# 
# **Set seed**
# 
# We use `np.random.seed()` to generate a sequence of random numbers. The seed enables us to use the same random numbers multiple times.

# In[2]:


np.random.seed(123)


# ## a)

# In[3]:


# Normal distributed values, with mean = 0 and sd = 1 (using Numpy)
X = np.random.normal(0, 1, 100)


# In[4]:


# Generate Y_perfect
Y_perfect = -1 + 0.5 * X


# ## b)

# In[5]:


# Normal distributed values, with mean = 0 and sd = 0.25
err = np.random.normal(0, 0.25, 100)


# ## c)

# In[6]:


# Model 1
Y = -1 + 0.5 * X + err


# In[7]:


# length of vector Y? 
Y.size


# Values of our parameters $\beta_0$ and $\beta_1$ in the linear model:
# 
# - $\beta_0 = -1$ (intercept)
# - $\beta_1 = 0.5$ (slope)

# ## d)

# In[8]:


df = pd.DataFrame({'X': X, 'Y': Y, 'Y_perfect': Y_perfect, 'err': err})


# In[9]:


# Seaborn
sns.scatterplot(x='X', y='Y', data=df);


# In[10]:


# Plotly Express
px.scatter(df, x='X', y='Y')


# Positive linear relationship between X and Y with some variance in the data (... as expected since we created this data with some noise (error)...)

# ## e)

# In[11]:


# Fit Model
lm = smf.ols(formula='Y ~ X', data=df).fit()


# In[12]:


# Print summary
lm.summary()


# In[13]:


# Mean squared error of residuals
lm.mse_resid


# - The linear regression fits a model which is almost true to the values of the coefficients of the population model
# (as we constructed it). 
# 
# - The predictor x is highly statistically significant and we can observe a large F-statistic
# with a near-zero p-value so the null hypothesis can be rejected.
# 
# - Our model explains around 84% of the variation in the data (see Adjusted R-squared)

# ## f)

# In[14]:


# create a scatterplot with OLS trendline
fig = px.scatter(df, x="X", y="Y", trendline="ols")

# make trendline red
fig.data[1].line.color = 'red'

# show figure
fig.show()


# ## g) 

# In[15]:


# Fit Model
lm_2 = smf.ols(formula='Y ~ X + I(X**2)', data=df).fit()
lm_2.summary()


# In[16]:


# Mean squared error of residuals
lm_2.mse_resid


# Arguments, that the polynomial provides a **better** fit:
# 
# - $\beta_0$ and $\beta_1$ are slightly closer to $\hat{\beta_0}$ and $\hat{\beta_1}$
#    
# Arguments, that the polynomial provides **not a better** fit:
# 
# - While $R^2$ is same for both models, $Adj. R^2$ (which takes model complexity into account) is slightly lower (worse)
# - $\beta_0$ and $\beta_1$ have slightly higher standard error
# - $\beta_2$ with $p = 0.856$ is not signifcant
# - Scores for AIC and BIC are higher (worse)
# - Most importantly, the p-value of the t-statistic suggests that there isn’t a significant relationship between Y
# and $X^2$ why we should use the model in (e) 
#     - this makes sense since we know the population model doesn't contain a quadratic relationship.

# ## h)
# 

# In[17]:


# a) Normal distributed values, with mean = 0 and sd = 1
X_h = np.random.normal(0, 1, 100)
# b) with lower standard deviation 
err_h = np.random.normal(0, 0.05, 100)
# c) Model
Y_h = -1 + 0.5 * X_h + err_h
# d) Scatterplot
sns.scatterplot(x=X_h, y=Y_h);


# Positive linear relationship between x2 and y2 with only little variance in the data (... as expected since we created this model...).

# In[18]:


# e)
df_h = pd.DataFrame({'X': X_h, 'Y': Y_h})
lm_h = smf.gls(formula='Y ~ X', data=df_h).fit()
lm_h.summary()


# In[19]:


# Mean squared error of residuals
lm_h.mse_resid


# We can observe very good results ... which is not surprising since we have an almost perfect linear relationship between $Y_h$ and $X_h$.

# In[20]:


# f)
# Scatter Plot 
sns.lmplot(x="X", y="Y", data=df_h, ci=None, line_kws={'color': 'red'});


# Almost identical... (perfect fit)

# ## i)

# In[21]:


print('.'*35)
print ('95% CI noisier data (SD = 0.25):')
print(lm.conf_int())

print('.'*35)
print ('95% CI less noisy data (SD = 0.05):')
print(lm_h.conf_int())


# For the less noisy data set, the confidence intervals for both coefficients are more narrow.

# ## j)

# In[22]:


# standard error (se) of paramters (b)
print('.'*35)
print ('SE for noisier data (SD = 0.25):')
display(lm.bse)
print('.'*35)
print ('SE for less noisy data (SD = 0.05):')
lm_h.bse


# The very small standard errors in the less noisy data set indicate that most sample means are similar to the population
# mean (i.e., our sample parameters accurately reflect the population mean).
