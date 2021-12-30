#!/usr/bin/env python
# coding: utf-8

# # Lasso regression

# 
# *This tutorial is mainly based on the excellent book ["An Introduction to Statistical Learning"](https://www.statlearning.com/) from James et al. (2021), the scikit-learn documentation about [regressors with variable selection](https://scikit-learn.org/stable/modules/classes.html#regressors-with-variable-selection) as well as Python code provided by Jordi Warmenhoven in this [GitHub repository](https://nbviewer.org/github/JWarmenhoven/ISL-python/blob/master/Notebooks/Chapter%206.ipynb).*
# 
# Lasso regression relies upon the linear regression model but additionaly performs a so called `L1 regularization`, which is a process of introducing additional information in order to prevent overfitting. As a consequence, we can fit a model containing all possible predictors and use lasso to perform variable selection by using a technique that regularizes the coefficient estimates (it shrinks the coefficient estimates towards zero). In particular, the minimization objective does not only include the residual sum of squares (RSS) - like in the OLS regression setting - but also the sum of the absolute value of coefficients.
# 
# The residual sum of squares (RSS) is calculated as follows:
# 
# $$ RSS = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 $$
# 
# This formula can be stated as:
# 
# $$ RSS = \sum_{i=1}^{n} \bigg(y_i - \big( \beta_{0} + \sum_{j=1}^{p} \beta_{j} x_{ij} \big) \bigg)^2  $$
# 
# - $n$ represents the number of observations.
# - $p$ denotes the number of variables that are available in the dataset.
# - $x_{ij}$ represents the value of the jth variable for the ith observation, where i = 1, 2, . . ., n and j = 1, 2, . . . , p.
# 
# In the lasso regression, the minimization objective becomes:
# 
# $$ \sum_{i=1}^{n} \bigg(y_i - \big( \beta_{0} + \sum_{j=1}^{p} \beta_{j} x_{ij} \big) \bigg)^2 + \alpha \sum_{j=1}^{p} |\beta_j|   $$
# 
# which equals:
# 
# $$RSS + \alpha \sum_{j=1}^{p} |\beta_j|  $$
# 
# $\alpha$ (alpha) can take various values:
# 
#   - $\alpha$ = 0: Same coefficients as least squares linear regression
#   - $\alpha$ = ∞: All coefficients are zero
#   - 0 < $\alpha$ < ∞: coefficients are between 0 and that of least squares linear regression
# 
# Lasso regression’s advantage over least squares linear regression is rooted in the bias-variance trade-off. As $\alpha$ increases, the flexibility of the lasso regression fit decreases, leading to decreased variance but increased bias. This procedure is more restrictive in estimating the coefficients and - depending on your value of $\alpha$ - may set a number of them to exactly zero. This means in the final model the response variable will only be related to a small subset of the predictors—namely, those with nonzero coeffcient estimates. Therefore, selecting a good value of $\alpha$ is critical.

# ## Data
# 
# We illustrate the use of lasso regression on a data frame called "Hitters" with 20 variables and 322 observations of major league players (see [this documentation](https://cran.r-project.org/web/packages/ISLR/ISLR.pdf) for more information about the data). We want to predict a baseball player’s salary on the basis of various statistics associated with performance in the previous year.
# 
# ### Import

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


# ### Create labels and features
# 
# Since we will use the lasso algorithm from scikit learn, we need to encode our categorical features as one-hot numeric features (dummy variables):

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


# ### Standardization
# 
# Lasso performs best when all numerical features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
# 
# This means it is important to standardize our features. We do this by subtracting the mean from our observations and then dividing the difference by the standard deviation. This so called standard score $z$ for an observation $x$ is calculated as:
# 
# $$z = \frac{(x- \bar x)}{s}$$
# 
# where:
# 
# - x is an observation in a feature
# - $\bar x$ is the mean of that feature
# -  s is the standard deviation of that feature.
# 
# To avoid [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)), the standardization of numerical features should always be performed after data splitting and only from training data. Furthermore, we obtain all necessary statistics for our features (mean and standard deviation) from training data and also use them on test data. Note that we don't standardize our dummy variables (which only have values of 0 or 1).

# In[15]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


# In[16]:


X_train


# ## Lasso regression

# First, we apply lasso regression on the training set with an arbitrarily regularization parameter $\alpha$ of 1. 

# In[17]:


from sklearn.linear_model import Lasso

reg = Lasso(alpha=1)
reg.fit(X_train, y_train)


# ### Model evaluation

# We print the $R^2$-score for the training and test set.

# In[18]:


print('R squared training set', round(reg.score(X_train, y_train)*100, 2))
print('R squared test set', round(reg.score(X_test, y_test)*100, 2))


# MSE for the training and test set.

# In[19]:


from sklearn.metrics import mean_squared_error

# Training data
pred_train = reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
print('MSE training set', round(mse_train, 2))

# Test data
pred = reg.predict(X_test)
mse_test =mean_squared_error(y_test, pred)
print('MSE test set', round(mse_test, 2))


# ## Role of alpha
# 

# To better understand the role of alpha, we plot the lasso coefficients as a function of alpha (`max_iter` are the maximum number of iterations):

# In[20]:


import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0.01,500,100)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha');


# Remember that if alpha = 0, then the lasso gives the least squares fit, and when alpha becomes very large, the lasso gives the null model in which all coefficient estimates equal zero. 
# 
# Moving from left to right in our plot, we observe that at first the lasso models contains many predictors with high magnitudes of coefficient estimates. With increasing alpha, the coefficient estimates approximate towards zero.
# 
# Next, we use cross-validation to find the best value for alpha.

# ## Lasso with optimal alpha
# 
# To find the optimal value of alpha, we use scikit learns lasso linear model with iterative fitting along a regularization path ([LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)). The best model is selected by cross-validation.

# ### k-fold cross validation

# In[21]:


from sklearn.linear_model import LassoCV

# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
model.fit(X_train, y_train)


# Show best value of penalization chosen by cross validation:

# In[22]:


model.alpha_


# ### Best model

# Use best value for our final model:

# In[23]:


# Set best alpha
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)


# Show model coefficients and names:

# In[24]:


print(list(zip(lasso_best.coef_, X)))


# ### Model evaluation

# In[25]:


print('R squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
print('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))


# In[26]:


mean_squared_error(y_test, lasso_best.predict(X_test))


# Lasso path: plot results of cross-validation with mean squared erros (for more information about the plot visit the [scikit-learn documentation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py))

# In[27]:


plt.semilogx(model.alphas_, model.mse_path_, ":")
plt.plot(
    model.alphas_ ,
    model.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(
    model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("alphas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

ymin, ymax = 50000, 250000
plt.ylim(ymin, ymax);

