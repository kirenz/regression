#!/usr/bin/env python
# coding: utf-8

# # Lasso Regression and model subset selection

# This content is mainly based on the following scikit learn documentations:
# 
# - [Model-based and sequential feature selection](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py) from Manoj Kumar, Maria Telenczuk and Nicolas Hug.
# - [Common pitfalls in the interpretation of coefficients of linear models](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py)

# ## Import data

# In[1]:


import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/happiness_report.csv")
df.info()


# In[2]:


# reset index
df.set_index('country', inplace=True)


# In[3]:


# show data
df


# ## Data preprocessing & exploration

# In[4]:


# prepare data for scikit learn models
feature_names = ['gdp', 'family', 'lifeexpectancy', 'trust']

X = df[feature_names]
y = df['happiness']


# - We split the sample into a train and a test dataset. 
# - Only the train dataset will be used in the following exploratory analysis. 
# - This is a way to emulate a real situation where predictions are performed on an unknown target, and we don’t want our analysis and decisions to be biased by our knowledge of the test data.

# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=120)


# In[6]:


# explore data
import seaborn as sns

train_dataset = X_train.copy()
sns.pairplot(train_dataset, kind="reg", diag_kind="kde");


# ## Lasso model

# Fit lasso regression with k-fold cross validation:

# In[7]:


from sklearn.linear_model import LassoCV

lasso = LassoCV().fit(X, y)


# Show feature importance plot

# In[8]:


import matplotlib.pyplot as plt
import numpy as np

importance = np.abs(lasso.coef_)
feature_names = np.array(feature_names)

plt.bar(height=importance, x=feature_names)

plt.title("Feature importances via coefficients")
plt.show()


# ## Selecting features based on importance
# 
# We want to select the two features which are the most important according to the coefficients. The SelectFromModel is meant just for that. SelectFromModel accepts a threshold parameter and will select the features whose importance (defined by the coefficients) are above this threshold.
# 
# In our case, we want to select only 2 features. Hence, we will set the threshold slightly above the coefficient of the third most important feature. 
# 
# We also record the time the algorithm takes to obtain the results.

# In[9]:


from sklearn.feature_selection import SelectFromModel
from time import time

# set threshold
threshold = np.sort(importance)[-3] + 0.01

# obtain time
tic = time()

# fit model
sfm = SelectFromModel(lasso, threshold=threshold).fit(X, y)

# obtain time
toc = time()

# print results
print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
print(f"Done in {toc - tic:.3f}s")


# ## Selecting features with Sequential Feature Selection (SFS)
# 
# Another way of selecting features is to use SequentialFeatureSelector (SFS). SFS is a greedy procedure where, at each iteration, we choose the best new feature to add to our selected features based a cross-validation score. 
# 
# - `Forward-Selection`: That is, we start with 0 features and choose the best single feature with the highest score. The procedure is repeated until we reach the desired number of selected features.
# 
# - `Backward selection`: We can also go in the reverse direction (backward SFS), i.e. start with all the features and greedily choose features to remove one by one. We illustrate both approaches here.
# 
# 

# ### Forward selection

# In[10]:


from sklearn.feature_selection import SequentialFeatureSelector

tic_fwd = time()

sfs_forward = SequentialFeatureSelector(
    lasso, n_features_to_select=2, direction="forward"
).fit(X, y)

toc_fwd = time()


# In[11]:


print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")


# ### Backward selection

# In[12]:


tic_bwd = time()

sfs_backward = SequentialFeatureSelector(
    lasso, n_features_to_select=2, direction="backward"
).fit(X, y)

toc_bwd = time()


# In[13]:


print(
    "Features selected by backward sequential selection: "
    f"{feature_names[sfs_backward.get_support()]}"
)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")


# ## Discussion
# 
# To finish with, we should note that 
# 
# - SelectFromModel is significantly faster than SFS (SelectFromModel only needs to fit a model once, while SFS needs to cross-validate many different models for each of the iterations)
# 
# - SFS however works with any model, while SelectFromModel requires the underlying estimator to expose a coef_ attribute or a feature_importances_ attribute. 
# 
# - The forward SFS is faster than the backward SFS because it only needs to perform n_features_to_select = 2 iterations, while the backward SFS needs to perform n_features - n_features_to_select.
