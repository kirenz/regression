#!/usr/bin/env python
# coding: utf-8

# # Scikit-learn
# 
# 
# In this tutorial, we will build a model with the Python [`scikit-learn`](https://scikit-learn.org/stable/) module. Additionally, you will learn how to create a data preprocessing pipline.

# # Data preparation

# In[1]:


# See notebook "Data Exploration" for details about data preprocessing
from case_duke_data_prep import *


# ## Data preprocessing pipeline

# In[2]:


# Modules
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[3]:


# for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])


# In[4]:


# for categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


# In[5]:


# Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


# # Simple regression

# In[6]:


# Select features for simple regression
features = ['area']
X = df[features]

# Create response
y = df["price"]


# In[7]:


# check feature
X.info()


# In[8]:


# check label
y


# In[9]:


# check for missing values
print("Missing values X:",X.isnull().any(axis=1).sum())

print("Missing values Y:",y.isnull().sum())


# ## Data splitting

# In[10]:


from sklearn.model_selection import train_test_split

# Train Test Split
# Use random_state to make this notebook's output identical at every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Modeling

# In[11]:


from sklearn.linear_model import LinearRegression

# Create pipeline with model
lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])


# In[12]:


# Fit model
lm_pipe.fit(X_train, y_train)


# In[13]:


# Obtain model coefficients
lm_pipe.named_steps['lm'].coef_


# ## Evaluation with training data
# 
# There are various options to evaluate a model in scikit-learn. Review this overview about [metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html).

# In[14]:


X_train.head()


# In[15]:


y_pred = lm_pipe.predict(X_train)


# In[16]:


y_pred


# In[17]:


from sklearn.metrics import r2_score

r2_score(y_train, y_pred)  


# In[18]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_pred)


# In[19]:


# RMSE
mean_squared_error(y_train, y_pred, squared=False)


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

# Plot with matplotlib
plt.scatter(X_train, y_train,  color='black')
plt.plot(X_train, y_pred, color='darkred', linewidth=3);


# In[21]:


import seaborn as sns 
sns.set_theme(style="ticks")

# Plot with Seaborn

# We first need to create a DataFrame
df_train = pd.DataFrame({'x': X_train['area'], 'y':y_train})

sns.lmplot(x='x', y='y', data=df_train, line_kws={'color': 'darkred'}, ci=False);


# In[22]:


import plotly.express as px

# Plot with Plotly Express
px.scatter(x=X_train['area'], y=y_train, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred')


# In[23]:


sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80});


# ## Evaluation with test data

# In[24]:


y_pred = lm_pipe.predict(X_test)


# In[25]:


print('MSE:', mean_squared_error(y_test, y_pred))

print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))


# In[26]:


# Plot with Plotly Express
px.scatter(x=X_test['area'], y=y_test, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred')


# Model generalization on unseen data (see [plotly documentation](https://plotly.com/python/ml-regression/))
# 

# In[27]:


import numpy as np
import plotly.graph_objects as go

x_range = pd.DataFrame({ 'area': np.linspace(X_train['area'].min(), X_train['area'].max(), 100)})
y_range =  lm_pipe.predict(x_range)

go.Figure([
    go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
    go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
    go.Scatter(x=x_range.area, y=y_range, name='prediction')
])


# # Multiple regression

# In[28]:


# Select features for multiple regression
features= [
 'bed',
 'bath',
 'area',
 'year_built',
 'cooling',
 'lot'
  ]
X = df[features]

X.info()
print("Missing values:",X.isnull().any(axis = 1).sum())

# Create response
y = df["price"]


# In[29]:


# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


# Create pipeline with model
lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])

# Fit model
lm_pipe.fit(X_train, y_train)


# In[31]:


# Obtain model coefficients
lm_pipe.named_steps['lm'].coef_


# In[32]:


y_pred = lm_pipe.predict(X_train)


# In[33]:


r2_score(y_train, y_pred)

