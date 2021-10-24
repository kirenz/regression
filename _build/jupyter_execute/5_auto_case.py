#!/usr/bin/env python
# coding: utf-8

# **Applied Statistics**<br/>
# Prof. Dr. Jan Kirenz <br/>
# Hochschule der Medien Stuttgart

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Import-data" data-toc-modified-id="Import-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import data</a></span></li><li><span><a href="#Tidying-data" data-toc-modified-id="Tidying-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Tidying data</a></span><ul class="toc-item"><li><span><a href="#Data-inspection" data-toc-modified-id="Data-inspection-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Data inspection</a></span></li><li><span><a href="#Handle-missing-values" data-toc-modified-id="Handle-missing-values-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Handle missing values</a></span></li></ul></li><li><span><a href="#Transform-data" data-toc-modified-id="Transform-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Transform data</a></span></li><li><span><a href="#Visualize-data" data-toc-modified-id="Visualize-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Visualize data</a></span><ul class="toc-item"><li><span><a href="#Distibution-of-Variables" data-toc-modified-id="Distibution-of-Variables-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Distibution of Variables</a></span></li></ul></li><li><span><a href="#Regression-Models" data-toc-modified-id="Regression-Models-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Regression Models</a></span><ul class="toc-item"><li><span><a href="#Models" data-toc-modified-id="Models-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Models</a></span><ul class="toc-item"><li><span><a href="#Simple-Linear-Regression" data-toc-modified-id="Simple-Linear-Regression-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Simple Linear Regression</a></span></li><li><span><a href="#GLS-Regression" data-toc-modified-id="GLS-Regression-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>GLS Regression</a></span></li><li><span><a href="#Mixed-Linear-Model" data-toc-modified-id="Mixed-Linear-Model-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>Mixed Linear Model</a></span></li></ul></li><li><span><a href="#Interpretation-of-OLS-Model" data-toc-modified-id="Interpretation-of-OLS-Model-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Interpretation of OLS-Model</a></span></li><li><span><a href="#Regression-Diagnostics" data-toc-modified-id="Regression-Diagnostics-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Regression Diagnostics</a></span><ul class="toc-item"><li><span><a href="#Residuals-vs-fitted-plot" data-toc-modified-id="Residuals-vs-fitted-plot-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>Residuals vs fitted plot</a></span></li><li><span><a href="#Normal-Q-Q" data-toc-modified-id="Normal-Q-Q-5.3.2"><span class="toc-item-num">5.3.2&nbsp;&nbsp;</span>Normal Q-Q</a></span></li><li><span><a href="#Scale-Location-plot" data-toc-modified-id="Scale-Location-plot-5.3.3"><span class="toc-item-num">5.3.3&nbsp;&nbsp;</span>Scale-Location plot</a></span></li><li><span><a href="#Residuals-vs-leverage-plot" data-toc-modified-id="Residuals-vs-leverage-plot-5.3.4"><span class="toc-item-num">5.3.4&nbsp;&nbsp;</span>Residuals vs leverage plot</a></span></li></ul></li></ul></li></ul></div>

# In[1]:


# Python set up (load modules) 
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.graphics.regressionplots import plot_leverage_resid2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot') 
import seaborn as sns  
sns.set() 
from IPython.display import Image


# # Linear regression tutorial 'Auto' 
# 
# This tutorial involves the use of simple linear regression on the **Auto data set** (see data description). 
# 
# We use the lm() function to perform linear regressions with **mpg** as the response and **horsepower** as the predictor. Furthermore, we use the summary() function to print the results and comment on the output. For example:
#    1. Is there a relationship between the predictor and the response?
#    2. How strong is the relationship between the predictor and the response?
#    3. Is the relationship between the predictor and the response positive or negative?
#    4. What is the predicted mpg associated with a horsepower of 98? What are the associted 95% confidence and prediction intervals?
# 
# Finally, we plot the response and the predictor and produce some **diagnostic plots** (1. Residuals vs fitted plot, 2. Normal Q-Q plot, 3. Scale-location plot, 4. Residuals vs leverage plot) to describe the linear regression fit. We comment on any problems we see with the fit. 
# 
# ---

# ## Import data

# In[11]:


# Load the csv data files into pandas dataframes
df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/Auto.csv")


# ## Tidying data

# ### Data inspection

# First of all, let's take a look at the variables (columns) in the data set.

# In[12]:


# show all variables in the data set
df.columns


# In[13]:


# show the first 5 rows (i.e. head of the DataFrame)
df.head(5)


# In[14]:


# show the lenght of the variable name (i.e. the number of observations)
len(df["name"])


# In[15]:


# unique names 
len(df['name'].unique())


# It is not possible to easily check for duplicates since it is plausible that there are multiple car types of the same name...

# In[16]:


# data overview (with meta data)
df.info()


# In[19]:


# change data type
df['origin'] = pd.Categorical(df['origin'])
df['name'] = pd.Categorical(df['name'])


# In[17]:


df['horsepower'] = pd.to_numeric(df['horsepower']) # produces error


# We have an issue with the horsepower variable. It seems that there is an string present, were only integers should be allowed. We transfrom the data with pd.to_numeric and replace the string with a NAN ... [see Pandas documenation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_numeric.html):
# 
#   * pandas.to_numeric(arg, errors='raise', downcast=None)[source]
# 
#   * errors : {‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
#   * If ‘raise’, then invalid parsing will raise an exception
#   * If ‘coerce’, then invalid parsing will be set as NaN
#   * If ‘ignore’, then invalid parsing will return the input

# ### Handle missing values

# In[20]:


# show missing values (missing values - if present - will be displayed in yellow )
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# We can also check the column-wise distribution of null values:

# In[21]:


print(df.isnull().sum())


# In[22]:


df = df.dropna()


# In[23]:


print(df.isnull().sum())


# ## Transform data

# In[24]:


# summary statistics for all numerical columns
round(df.describe(),2)


# In[25]:


# summary statistics for all categorical columns
df.describe(include=['category'])


# ## Visualize data

# ### Distibution of Variables

# In[26]:


df.hist(figsize=(10,10), bins=20);


# In[27]:


# boxplot 
sns.boxplot(y='mpg', data=df, palette='winter');


# In[28]:


# check relationship with a joint plot
sns.jointplot(x="horsepower", y="mpg", data=df, stat_func=None);


# In[29]:


sns.pairplot(df);


# ## Regression Models

# ### Models
# 
# (a) Use the lm() function to perform a simple linear regression with **mpg** as the response and **horsepower** as the predictor. Use the summary() function to print the results. 

# #### Simple Linear Regression

# In[37]:


ols = smf.ols(formula ='mpg ~  horsepower', data=df, groups=df["cylinders"]).fit()
ols.summary()


# We use [Seaborne's lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html) to plot the regression line:

# In[42]:


# Plot regression line with 95% confidence intervall
sns.lmplot(x='horsepower', y='mpg', data=df, line_kws={'color':'red'}, height=5, ci=95, );


# #### GLS Regression

# In[36]:


gls = smf.gls(formula ='mpg ~  horsepower', data=df).fit()
gls.summary()


# #### Mixed Linear Model

# You find more information about mixed linear models in [Statsmodels documentation](https://www.statsmodels.org/stable/mixed_linear.html)

# In[38]:


mlm = smf.mixedlm(formula ='mpg ~  horsepower', data=df, groups=df["cylinders"]).fit()
mlm.summary()


# ### Interpretation of OLS-Model
# 
# **1. Is there a relationship between the predictor and the response?**
# 
# Yes, according to our linear model there is a statistically significant relationship between horsepower and mpg. The model coefficients are all significant on the 0.001 level and the F-statistic is far larger than 1 with a p-value close to zero. Therefore we can reject the null hypothesis and state that there is a statistically significant relationship between horsepower and mpg.

# **2. How strong is the relationship between the predictor and the response?**

# In[39]:


# Test relationship and strength with correlation
stats.pearsonr(df['mpg'], df['horsepower'])


# The $R^2$ of our model indicates a moderate to strong relationship (around 61% of variation in the data can be explained with our model) between the predictor and the response. Furthermore, we used Pearson's product-moment correlation to test the relationship between the predictor and the response (see code above). The results of the correlation indicate a strong, statistically significant negative relationship.

# **3. Is the relationship between the predictor and the response positive or negative?**
# 
# The relationship between mpg and horsepower is negative (see regression coefficient of horsepower (-0.1578). That means the more horsepower an automobile has the less mpg fuel efficiency the automobile will have according to our model. 
# 
# In particular, all other things being equal, an increase of 1 horsepower leads to an 0.1578 decrease in mpg on average. 

# **4. What is the predicted mpg associated with a horsepower of 98? What are the associted 95% confidence and prediction intervals?**

# In[40]:


to_predict = pd.DataFrame({'horsepower':[98]})
results = ols.get_prediction(to_predict)
results.summary_frame(alpha=0.05)


# The predicted mpg for 98 horsepower is about 24.47 mpg with a 95% confidence interval between 23.97 mpg and 24.96 mpg.
# 
# That means if we’d collected 100 samples, and for each sample calculated the regression coefficient for horsepower and a confidence interval for it, then for 95 of these samples, the confidence interval contains the value of the regression coefficient in the population, and in 5 of the samples the confidence interval does not contain the population paramater (i.e. the regrssion coefficient). 

# In[41]:


# CI of the parameter (however, this was not the question...)
lm.conf_int(alpha=0.05)


# ---
# ---
# 
# ### Regression Diagnostics
# 
# (c) Produce diagnostic plots of the least squares regression fit. Comment on any problems you see with the fit.

# #### Residuals vs fitted plot

# In[44]:


# fitted values
model_fitted_y = ols.fittedvalues;
# Basic plot
plot = sns.residplot(model_fitted_y, 'mpg', data=df, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 
                               'lw': 1, 'alpha': 0.8});

plot.set_title('Residuals vs Fitted');
plot.set_xlabel('Fitted values');
plot.set_ylabel('Residuals');


# The residuals are not equally spread around a horizontal line which is an indication for a non-linear relationship. This means there seems to be a non-linear relationship between the predictor and the response variable which the model doesn’t capture.

# #### Normal Q-Q

# This plots the standardized (z-score) residuals against the theoretical normal quantiles. Anything quite off the diagonal lines may be a concern for further investigation.

# In[46]:


# Use standardized residuals
sm.qqplot(ols.get_influence().resid_studentized_internal);


# This plot shows if residuals are normally distributed. If a normal distribution is present, the residuals should (more or less) follow a straight line. 
# We can observe that only some residuals (in the lower left and the upper right corner) deviate from the straight line. 
# 
# We could obtain the 3 observations with the largest deviations from our advanced plot below (observations 330, 327 and 320). 

# #### Scale-Location plot

# In[49]:


# Scale Location plot
plt.scatter(ols.fittedvalues, np.sqrt(np.abs(ols.get_influence().resid_studentized_internal)), alpha=0.5)
sns.regplot(ols.fittedvalues, np.sqrt(np.abs(ols.get_influence().resid_studentized_internal)), 
            scatter=False, ci=False, lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});


# This plot shows if residuals are spread equally along the ranges of predictors. This is how we can check the assumption of equal variance (homoscedasticity). It’s good if we observe a horizontal line with equally (randomly) spread points.
# 
# In our model the residuals begin to spread wider along the y-axis as it passes the x value of around 18. Because the residuals spread wider and wider with an increase of x, the red smooth line is not horizontal and shows a positive angle. This is an indication of heteroskedasticity.

# #### Residuals vs leverage plot

# In[51]:


fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(ols, ax = ax)


# In[53]:


# Additionally, obtain critical Cook's d values
ols_cooksd = ols.get_influence().cooks_distance[0]
n = len(df["name"])

critical_d = 4/n
print('Critical Cooks d:', critical_d)

#identification of potential outliers
out_d = ols_cooksd > critical_d

# Output potential outliers
df.index[out_d],ols_cooksd[out_d]

