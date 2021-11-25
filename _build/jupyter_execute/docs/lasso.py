#!/usr/bin/env python
# coding: utf-8

# # Lasso

# ## Basics

# Lasso performs a so called `L1 regularization` (a process of introducing additional information in order to prevent overfitting), i.e. adds penalty equivalent to absolute value of the magnitude of coefficients.
# 
# In particular, the minimization objective does not only include the residual sum of squares (RSS) - like in the OLS regression setting - but also the sum of the absolute value of coefficients.
# 
# The residual sum of squares (RSS) is calculated as follows:
# 
# $$ RSS = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 $$
# 
# This formula can be stated as:
# 
# $$ RSS = \sum_{i=1}^{n} \bigg(y_i - \big( \beta_{0} + \sum_{j=1}^{p} \beta_{j} x_{ij} \big) \bigg)^2  $$
# 
# - $n$ represents the number of distinct data points, or observations, in our sample.
# - $p$ denotes the number of variables that are available in the dataset.
# - $x_{ij}$ represents the value of the jth variable for the ith observation, where i = 1, 2, . . ., n and j = 1, 2, . . . , p.
# 
# In the lasso regression, the minimization objective becomes:
# 
# $$ \sum_{i=1}^{n} \bigg(y_i - \big( \beta_{0} + \sum_{j=1}^{p} \beta_{j} x_{ij} \big) \bigg)^2 + \lambda \sum_{j=1}^{p} |\beta_j|   $$
# 
# which equals:
# 
# $$RSS + \lambda \sum_{j=1}^{p} |\beta_j|  $$
# 
# 
# $\lambda$ (lambda) provides a trade-off between balancing RSS and magnitude of coefficients.
# 
# $\lambda$ can take various values:
# 
#   - $\lambda$ = 0: Same coefficients as simple linear regression
#   - $\lambda$ = ∞: All coefficients zero (same logic as before)
#   - 0 < $\lambda$ < ∞: coefficients between 0 and that of simple linear regression

# ## Python setup

# In[58]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import data
# 
# This notebook involves the use of the Lasso regression on an automobile dataset. 

# In[59]:


df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/Auto.csv")


# In[60]:


df


# In[61]:


df.info()


# ## Tidying data
# 
# We only use observations 1 to 200 for our analysis and drop the `name` variable.

# In[62]:


df = df.iloc[0:200]
df = df.drop(['name'], axis=1)


# In[63]:


df['origin'] = pd.Categorical(df['origin'])
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

print(df.isnull().sum())


# In[64]:


# drop missing cases
df = df.dropna()


# ## Transform data

# In[65]:


# Convert all columns to integer
df = df.astype('int')


# ### Split data
# 
# Split the data set into train and test sets (use `X_train`, `X_test`, `y_train`, `y_test`), with the first 75% of the data for training and the remaining for testing. (module: `from sklearn.model_selection import train_test_split`)

# In[66]:


X = df.drop(['mpg'], axis=1)
y = df['mpg']


# In[67]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


# ### Standardization
# 
# *Standardize the features with the module: `from sklearn.preprocessing import StandardScaler`*
# 
# It is important to standardize the features by removing the mean and scaling to unit variance. The L1 (Lasso) and L2 (Ridge) regularizers of linear models assume that all features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

# In[68]:


dfs.columns


# In[69]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# ## Model

# Apply **Lasso regression** on the training set with the regularization parameter **lambda = 0.5** (module: `from sklearn.linear_model import Lasso`) and print the $R^2$-score for the training and test set. Comment on your findings.

# In[70]:


from sklearn.linear_model import Lasso

reg = Lasso(alpha=0.5)
reg.fit(X_train, y_train)


# In[71]:


print('Lasso Regression: R squared score on training set', round(reg.score(X_train, y_train)*100, 2))
print('Lasso Regression: R squared score on test set', round(reg.score(X_test, y_test)*100, 2))


# ### Tune lambda

# #### Fit models

# Apply the **Lasso regression** on the training set with the following **λ parameters: (0.001, 0.01, 0.1, 0.5, 1, 2, 10)**. 
# 
# Evaluate the $R^2$ score for all the models you obtain on both the train and test sets.

# In[72]:


lambdas = (0.001, 0.01, 0.1, 0.5, 1, 2, 10)
l_num = 7
pred_num = X.shape[1]


# In[73]:


# prepare data for enumerate
coeff_a = np.zeros((l_num, pred_num))
train_r_squared = np.zeros(l_num)
test_r_squared = np.zeros(l_num)


# In[74]:


# enumerate through lambdas with index and i
for ind, i in enumerate(lambdas):    
    reg = Lasso(alpha = i)
    reg.fit(X_train, y_train)

    coeff_a[ind,:] = reg.coef_
    train_r_squared[ind] = reg.score(X_train, y_train)
    test_r_squared[ind] = reg.score(X_test, y_test)


# #### Plot result

# Plot all values for both data sets (train and test $R^2$-values) as a function of λ. Note that we use the index of the values at the x-axis, not the values themselves. 

# In[75]:


plt.figure(figsize=(18, 8))

plt.plot(train_r_squared, 'o-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
plt.plot(test_r_squared, 'o-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)

plt.xlabel('Lamda index'); plt.ylabel(r'$R^2$')
plt.xlim(0, 6)
plt.title(r'Evaluate lasso regression with lamdas: 0 = 0.001, 1= 0.01, 2 = 0.1, 3 = 0.5, 4= 1, 5= 2, 6 = 10')
plt.legend(loc='best')
plt.grid()


# #### Identify best lambda

# Store your test data results in a DataFrame and indentify the lambda where the $R^2$ has it's **maximum value** in the **test data**. 

# In[76]:


df_lam = pd.DataFrame(test_r_squared*100, columns=['R_squared'])
df_lam['lambda'] = (lambdas)
# returns the index of the row where column has maximum value.
df_lam.loc[df_lam['R_squared'].idxmax()]


# Fit a Lasso model with this lambda parameter (use the training data) and obtain the corresponding **regression coefficients**. 
# 

# In[77]:


reg_best = Lasso(alpha = 0.1)
reg_best.fit(X_train, y_train)
reg_best.coef_


# Obtain the **mean squared error** for the test data of this model (module: `from sklearn.metrics import mean_squared_error`)

# In[78]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, reg_best.predict(X_test))


# ## Tune with k-fold cross validation

# Evaluate the performance of a Lasso regression for different regularization parameters λ using **5-fold cross validation** on the training set (module: `from sklearn.model_selection import cross_val_score`) and plot the cross-validation (CV) $R^2$ scores of the training and test data as a function of λ.
# 

#  We use the following lambda parameters:

# In[79]:


l_min = 0.05
l_max = 0.2
l_num = 20
lambdas = np.linspace(l_min,l_max, l_num)


# In[80]:


# data preparation
train_r_squared = np.zeros(l_num)
test_r_squared = np.zeros(l_num)

pred_num = X.shape[1]
coeff_a = np.zeros((l_num, pred_num))


# In[81]:


# models
from sklearn.model_selection import cross_val_score

for ind, i in enumerate(lambdas):    
    reg = Lasso(alpha = i)
    reg.fit(X_train, y_train)
    results = cross_val_score(reg, X, y, cv=5, scoring="r2")

    train_r_squared[ind] = reg.score(X_train, y_train)    
    test_r_squared[ind] = reg.score(X_test, y_test)


# In[82]:


# Plotting
plt.figure(figsize=(18, 8))

plt.plot(train_r_squared, 'o-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
plt.plot(test_r_squared, 'o-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)

plt.xlabel('Lamda index'); plt.ylabel(r'$R^2$')
plt.xlim(0, 19)
plt.title(r'Evaluate 5-fold cv with different lamdas')
plt.legend(loc='best')
plt.grid()


# Finally, store your test data results in a DataFrame and identify the lambda where the $R^2$ has it's **maximum value** in the **test data**. 

# In[83]:


df_lam = pd.DataFrame(test_r_squared*100, columns=['R_squared'])
df_lam['lambda'] = (lambdas)
# returns the index of the row where column has maximum value.
df_lam.loc[df_lam['R_squared'].idxmax()]


# Fit a Lasso model with this lambda parameter (use the training data) and obtain the corresponding **regression coefficients**. Furthermore, obtain the **mean squared error** for the test data of this model (module: `from sklearn.metrics import mean_squared_error`)

# In[84]:


# Best Model
reg_best = Lasso(alpha = 0.05)
reg_best.fit(X_train, y_train)


# In[85]:


mean_squared_error(y_test, reg_best.predict(X_test))


# In[86]:


reg_best.coef_

