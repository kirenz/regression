#!/usr/bin/env python
# coding: utf-8

# # TensorFlow
# 
# ## Case study: Houses for sale

# ## Setup

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

sns.set_theme(style="ticks", color_codes=True)


# ## Data preparation
# 
# See notebook `10a-application-model-data-exploration.ipynb` for details about data preprocessing and data exploration.

# In[10]:


ROOT = "https://raw.githubusercontent.com/kirenz/modern-statistics/main/data/"
DATA = "duke-forest.csv"
df = pd.read_csv(ROOT + DATA)

# Drop irrelevant features
df = df.drop(['url', 'address', 'type'], axis=1)

# Convert data types
df['heating'] = df['heating'].astype("category")
df['cooling'] = df['cooling'].astype("category")
df['parking'] = df['parking'].astype("category")

# drop column with too many missing values
df = df.drop(['hoa'], axis=1)

df.columns.tolist()


# # Simple regression

# In[11]:


# Select features for simple regression
features = ['area']
X = df[features]

X.info()
print("Missing values:",X.isnull().any(axis = 1).sum())

# Create response
y = df["price"]


# ## Data splitting

# In[12]:


from sklearn.model_selection import train_test_split

# Train Test Split
# Use random_state to make this notebook's output identical at every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Linear regression

# Start with a single-variable linear regression, to predict price from area.
# 
# Training a model with tf.keras typically starts by defining the model architecture.
# 
# In this case use a keras.Sequential model. This model represents a sequence of steps. In this case there is only one step:
# 
# - Apply a linear transformation to produce 1 output using layers.Dense.
#   
# The number of inputs can either be set by the input_shape argument, or automatically when the model is run for the first time.

# Build the sequential model:

# In[13]:


lm = tf.keras.Sequential([
    layers.Dense(units=1, input_shape=(1,))
])

lm.summary()


# This model will predict price from area.
# 
# Run the untrained model on the first 10 area values. The output won't be good, but you'll see that it has the expected shape, (10,1):

# In[14]:


lm.predict(X_train[:10])


# Once the model is built, configure the training procedure using the Model.compile() method. The most important arguments to compile are the loss and the optimizer since these define what will be optimized (mean_absolute_error) and how (using the optimizers.Adam).

# In[15]:


lm.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# Once the training is configured, use Model.fit() to execute the training:

# In[16]:


get_ipython().run_cell_magic('time', '', 'history = lm.fit(\n    X_train, y_train,\n    epochs=400,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.1)')


# In[17]:


y_train


# In[18]:


# Calculate R squared
from sklearn.metrics import r2_score

y_pred = lm.predict(X_train).astype(np.int64)
y_true = y_train.astype(np.int64)

r2_score(y_train, y_pred)  


# In[19]:


# slope coefficient
lm.layers[0].kernel


# Visualize the model's training progress using the stats stored in the history object.

# In[20]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[21]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [price]')
  plt.legend()
  plt.grid(True)


# In[22]:


plot_loss(history)


# Collect the results (mean squared error) on the test set, for later:

# In[23]:


test_results = {}

test_results['lm'] = lm.evaluate(
    X_test,
    y_test, verbose=0)

test_results


# Since this is a single variable regression it's easy to look at the model's predictions as a function of the input:

# In[26]:


x = tf.linspace(0.0, 6200, 6201)
y = lm.predict(x)

y


# In[27]:


def plot_area(x, y):
  plt.scatter(X_train['area'], y_train, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('area')
  plt.ylabel('price')
  plt.legend()


# In[28]:


plot_area(x,y)


# # Multiple Regression

# In[29]:


# Select all relevant features
features= [
 'bed',
 'bath',
 'area',
 'year_built',
 'cooling',
 'lot'
  ]
X = df[features]

# Convert categorical to numeric
X = pd.get_dummies(X, columns=['cooling'], prefix='cooling', prefix_sep='_')

X.info()
print("Missing values:",X.isnull().any(axis = 1).sum())

# Create response
y = df["price"]


# In[30]:


from sklearn.model_selection import train_test_split

# Train Test Split
# Use random_state to make this notebook's output identical at every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


lm_2 = tf.keras.Sequential([
    layers.Dense(units=1, input_shape=(7,))
])

lm_2.summary()


# In[32]:


lm_2.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[33]:


get_ipython().run_cell_magic('time', '', 'history = lm_2.fit(\n    X_train, y_train,\n    epochs=400,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.1)')


# In[34]:


# Calculate R squared
from sklearn.metrics import r2_score

y_pred = lm_2.predict(X_train).astype(np.int64)
y_true = y_train.astype(np.int64)

r2_score(y_train, y_pred)  


# In[35]:


# slope coefficients
lm_2.layers[0].kernel


# In[36]:


plot_loss(history)


# In[37]:


test_results['lm_2'] = lm_2.evaluate(
    X_test, y_test, verbose=0)


# # DNN regression

# The previous section implemented linear models for single and multiple inputs.
# 
# This section implements a multiple-input DNN models. The code is basically the same except the model is expanded to include some "hidden" non-linear layers. The name "hidden" here just means not directly connected to the inputs or outputs.
# 
# These models will contain a few more layers than the linear model:
# 
# - Two hidden, nonlinear, Dense layers using the relu nonlinearity.
# - A linear single-output layer.

# In[38]:


dnn_model = keras.Sequential([
      layers.Dense(units=1, input_shape=(7,)),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

dnn_model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))


# In[39]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    X_train, y_train,\n    epochs=100,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.1)')


# In[40]:


# Calculate R squared
from sklearn.metrics import r2_score

y_pred = dnn_model.predict(X_train).astype(np.int64)
y_true = y_train.astype(np.int64)

r2_score(y_train, y_pred)  


# In[41]:


plot_loss(history)


# In[42]:


test_results['dnn_model'] = dnn_model.evaluate(
    X_test, y_test, verbose=0)


# # Performance comparison

# In[43]:


pd.DataFrame(test_results, index=['Mean absolute error [price]']).T

