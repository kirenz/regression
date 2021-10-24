#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot') 
import seaborn as sns  
sns.set() 
from IPython.display import Image


# # Application 4: Linear regression diagnostics
# 
# When we fit a linear regression model to a particular data set, many problems may occur. Most common among these are the following:
# 
# 1. Non-linearity of the response-predictor relationships 
# 2. Normally distributed errors (outliers).
# 3. Correlation of error terms.
# 4. Non-constant variance of error terms (heteroskedasticity).
# 5. High-leverage points.
# 6. Multicollinearity.
# 
# In many cases of statistical analysis, we are not sure whether our statistical model is correctly specified. For example when using ols, then linearity and homoscedasticity are assumed, some test statistics additionally assume that the errors are normally distributed or that we have a large sample. Since our results depend on these statistical assumptions, the results are only correct of our assumptions hold (at least approximately).
# 
# One solution to the problem of uncertainty about the correct specification is to use robust methods, for example robust regression or robust covariance (sandwich) estimators. 
# 
# The second approach is to test whether our sample is consistent with these assumptions, which we cover in this application.
# 
# Source: [Statsmodel](https://www.statsmodels.org/stable/diagnostic.html)
# 
# For presentation purposes, we use the zip(name,test) construct to pretty-print short descriptions in the codes below.
# 
# ---
# Sources
# 
# See Statsmodels [regression diagnostic page](https://www.statsmodels.org/stable/diagnostic.html) and [Statsmodel examples with code](https://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html)

# ## 1 Import data

# In[2]:


# Load the csv data files into pandas dataframes
PATH = '/Users/jankirenz/Dropbox/Data/' 
df = pd.read_csv(PATH + 'Auto.csv')


# ## Tidying data

# In[3]:


# show all variables in the data set
df.columns


# In[4]:


# show the first 5 rows (i.e. head of the DataFrame)
df.head(5)


# In[5]:


# show the lenght of the variable id (i.e. the number of observations)
len(df["name"])


# In[6]:


# check for duplicates and print results (if the two numbers match, we have no duplicates)
# show the lenght of the variable id (i.e. the number of observations)
print(f'IDs: {len(df["name"])}')
# count the number of individual id's
print(f'Unique IDs: {len(df["name"].value_counts())}')


# It is not possible to easily check for duplicates since it is plausible that there are multiple car types of the same name...

# In[7]:


# data overview (with meta data)
df.info()


# In[8]:


# change data type
df['origin'] = pd.Categorical(df['origin'])
df['year'] = pd.Categorical(df['year'], ordered=True)
#df['horsepower'] = pd.to_numeric(df['horsepower']) # produces error
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce') # solution


# ### Handle missing values

# In[9]:


# show missing values (missing values - if present - will be displayed in yellow )
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# In[10]:


print(df.isnull().sum())


# In[11]:


df = df.dropna()


# In[12]:


print(df.isnull().sum())


# ## Transform data

# In[13]:


# summary statistics for all numerical columns
round(df.describe(),2)


# In[14]:


# summary statistics for all categorical columns
df.describe(include=['category'])


# # Regression diagnostics

# In[15]:


# fit linear model with statsmodels.formula.api (with R-style formulas) 
lm = smf.ols(formula ='mpg ~ horsepower', data=df).fit()
#lm.summary()


# # 1. Non-linearity of the response-predictor relationships

# ###  Harvey-Collier multiplier test
# Harvey-Collier multiplier test for Null hypothesis that the linear specification is correct. This test is a t-test that the mean of the recursive ols residuals is zero. 
# 
# A significant result (rejecting the null) occurs when the fit is better with a range restriction (which is what happens if the model is nonlinear).

# In[16]:


name = ['t value', 'p value']
test = sm.stats.linear_harvey_collier(lm)
lzip(name, test)


# ---
# Residual plots are also a very useful graphical tool for identifying non-linearity:

# ###  Residuals vs fitted plot

# In[17]:


# fitted values
model_fitted_y = lm.fittedvalues;
# Basic plot
plot = sns.residplot(model_fitted_y, 'mpg', data=df, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 
                               'lw': 1, 'alpha': 0.8});

plot.set_title('Residuals vs Fitted');
plot.set_xlabel('Fitted values');
plot.set_ylabel('Residuals');


# The residuals are not equally spread around a horizontal line which is an indication for a non-linear relationship. This means there seems to be a non-linear relationship between the predictor and the response variable which the model doesn’t capture.

# **Advanced Plots:**
# 
# Besides basic plots, we will also cover some more advanced plots (similar to the R regression diagnostic plots) which flag certain observations. See here for a [description of the R diagnsotic plots](https://data.library.virginia.edu/diagnostic-plots/). 
# 
# The code for the advanced plots was obtained from [here](https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034) 

# **Advanced Residuals vs fitted plot (not necessary)**

# In[18]:


# Necessary values for our advanced plots:
# fitted values
model_fitted_y = lm.fittedvalues;
# model residuals
model_residuals = lm.resid
# normalized residuals
model_norm_residuals = lm.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)


# In[19]:


# Advanced plot (1)
# figure size
plot = plt.figure(1)
plot.set_figheight(8)
plot.set_figwidth(12)
# generate figure with sns.residplot 
plot.axes[0] = sns.residplot(model_fitted_y, 'mpg', data=df, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
# label axes
plot.axes[0].set_title('Residuals vs Fitted')
plot.axes[0].set_xlabel('Fitted values')
plot.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot.axes[0].annotate(i, 
                        xy=(model_fitted_y[i], 
                        model_residuals[i]));


# #### POSSIBLE SOLUTION FOR IDENTIFIED NON-LINEARITY

# We can fit a non-linear function (polynomial regression)

# In[20]:


lm_2 = smf.ols(formula='mpg ~ horsepower + I(horsepower**2)', data=df).fit()
#lm_2.summary()


# In[21]:


# fitted values
model_fitted_y_2 = lm_2.fittedvalues;
# Basic plot
plot = sns.residplot(model_fitted_y_2, 'mpg', data=df, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 
                               'lw': 1, 'alpha': 0.8});

plot.set_title('Residuals vs Fitted');
plot.set_xlabel('Fitted values');
plot.set_ylabel('Residuals');


# # 2. Normality of the residuals
# 
# It can be helpful if the residuals in the model are random, normally distributed variables with a mean of 0. 
# 
# This assumption means that the differences between the predicted and observed data are most frequently zero or very close to zero, and that differences much greater than zero happen only occasionally.
# 
# Some people confuse this assumption with the idea that predictors have to be normally distributed, which they don’t. In small samples a lack of normality invalidates confidence intervals and significance tests, whereas in large samples it will not because of the **central limit theorem**. 
# 
# If you are concerned only with estimating the model parameters (and not significance tests and confidence intervals) then this assumption barely matters. If you bootstrap confidence intervals then you can ignore this assumption.

# ### Jarque-Bera test
# 
# The Jarque–Bera test is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution. 
# 
# The null hypothesis is a joint hypothesis of the skewness being zero and the excess kurtosis being zero. 
# 
# Samples from a normal distribution have an expected skewness of 0 and an expected excess kurtosis of 0 (which is the same as a kurtosis of 3). As the definition of JB shows, any deviation from this increases the JB statistic.

# In[22]:


name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sm.stats.jarque_bera(lm.resid)
lzip(name, test)


# ### Omnibus normtest
# 
# Test for normal distribution of residuals. In this case, we use the $Chi^2$-Test. The Chi-Square Test for normality allows us to check whether or not the model residuals follow an approximately normal distribution.
# 
# Our null hypothesis is that the residuals are from a normal distribution.

# In[23]:


name = ['Chi^2', 'Two-tail probability']
test = sm.stats.omni_normtest(lm.resid)
lzip(name, test)


# ## Notes on outliers.
# 
# An outlier is a point for which $y_i$ is far from the value predicted by the model. Outliers can arise for a variety of reasons, such as incorrect recording of an observation during data collection.
# 
# In practice, it can be difficult to decide how large a residual needs to be before we consider the point to be an outlier. To address this problem, instead of plotting the residuals, we can plot the studentized residuals, computed by dividing each residual by its estimated standard error. 
# 
# Observations whose studentized residuals are greater than 3 in absolute value are possible outliers. 
# 
# If we believe that an outlier has occurred due to an error in data collection or recording, then one solution is to simply remove the observation. However, care should be taken, since an outlier may instead indicate a deficiency with the model, such as a missing predictor.

# ### Normal Q-Q-Plot
# 
# This plots the standardized (z-score) residuals against the theoretical normal quantiles. Anything quite off the diagonal lines may be a concern for further investigation.

# In[24]:


# Use standardized residuals
sm.qqplot(lm.get_influence().resid_studentized_internal);


# This plot shows if residuals are normally distributed. If a normal distribution is present, the residuals should (more or less) follow a straight line. 
# We can observe that only some residuals (in the lower left and the upper right corner) deviate from the straight line. 
# 
# We could obtain the 3 observations with the largest deviations from our advanced plot below (observations 330, 327 and 320). 

# **Advanced QQ-Plot**

# In[25]:


# Advanced plot (2)
# ProbPlot and its qqplot method from statsmodels graphics API. 
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
# figure size
plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)
# figure labels
plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]
# label 3 largest deviations
for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]));


# # 3. Correlation of error terms.
# 
# An important assumption of the linear regression model is that the error terms are uncorrelated.
# 
# Such correlations frequently occur in the context of time series data, which consists of observations for which measurements are obtained at discrete points in time. In many cases, observations that are obtained at adjacent time points will have positively correlated errors. In order to determine if this is the case for a given data set, we can plot the residuals from our model as a function of time. If the errors are uncorrelated, then there should be no discernible pattern.
# 
# Correlation among the error terms can also occur outside of time series data. For instance, consider a study in which individuals’ heights are predicted from their weights. The assumption of uncorrelated errors could be violated if some of the individuals in the study are members of the same family, or eat the same diet, or have been exposed to the same environmental factors. 
# 
# In general, the assumption of uncorrelated errors is extremely important for linear regression as well as for other statistical methods, and good experimental design is crucial in order to mitigate the risk of such correlations.
# 
# A test of autocorrelation that is designed to take account of the regression model is the **Durbin-Watson test**. It is used to test the hypothesis that there is no **lag one autocorrelation** in the residuals. If there is no autocorrelation, the Durbin-Watson distribution is symmetric around 2. 
# 
# A small p-value indicates there is significant autocorrelation remaining in the residuals.
# 
# As a rough rule of thumb, if Durbin–Watson is less than 1.0, there may be cause for alarm. Small values of d indicate successive error terms are positively correlated. If d > 2, successive error terms are negatively correlated.

# In[26]:


sm.stats.durbin_watson(lm.resid)


# # 4. Non-constant Variance of Error Terms
# 

# Another important assumption of the linear regression model is that the error terms have a constant variance. 
# 
# For instance, the variances of the error terms may increase with the value of the response. One can identify non-constant variances in
# the errors, or **heteroscedasticity**, from the presence of a funnel shape in the residual plot. 
# 
# When faced with this problem, one possible solution is to transform the response Y using a concave function such as log Y or √Y . Such a transformation results in a greater amount of shrinkage of the larger responses, leading to a reduction in heteroscedasticity.

# ### Breusch-Pagan test:
# 
# Test assumes homoskedasticity (null hypothesis). If one of the test statistics is significant, then you have evidence of heteroskedasticity. 

# In[27]:


name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sm.stats.het_breuschpagan(lm.resid, lm.model.exog)
lzip(name, test)


# ### Scale-Location plot

# In[28]:


# Scale Location plot
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});


# This plot shows if residuals are spread equally along the ranges of predictors. This is how we can check the assumption of equal variance (**homoscedasticity**). It’s good if we observe a horizontal line with equally (randomly) spread points.
# 
# In our model the residuals begin to spread wider along the y-axis as it passes the x value of around 18. Because the residuals spread wider and wider with an increase of x, the red smooth line is not horizontal and shows a positive angle. This is an indication of **heteroskedasticity**.

# # 5. High-leverage points.
# 
# We just saw that outliers are observations for which the response $y_i$ is unusual given the predictor $x_i$. In contrast, observations with high leverage have an unusual value for $x_i$. 
# 
# In a simple linear regression, high leverage observations are fairly easy to identify, since we can simply look for observations for which the predictor value is outside of the normal range of the observations. But in a multiple linear regression with many predictors, it is possible to have an observation that is well within the range of each individual predictor’s values, but that is unusual in terms of the full set of predictors.
# 
# A general rule of thumb is that observations with a **Cook’s D** over 4/n, where n is the number of observations, is a possible outlier. 

# ### Statsmodel influence
# 
# Once created, an object of class OLSInfluence holds attributes and methods that allow users to assess the influence of each observation. 

# In[29]:


# obtain statistics
infl = lm.get_influence()


# In[30]:


lm_cooksd = lm.get_influence().cooks_distance[0]
n = len(df["name"])
critical_d = 4/n
print('Critical Cooks d:', critical_d)
#identification of potential outliers
out_d = lm_cooksd > critical_d
# Output potential outliers
df.index[out_d],lm_cooksd[out_d]


# In[31]:


# Show summary frame of leverage statistics
print(infl.summary_frame().filter(["student_resid","dffits","cooks_d"]))


# ### Residuals vs leverage plot

# Plots leverage statistics vs. normalized residuals squared. See [statsmodel documentation](http://www.statsmodels.org/0.6.1/generated/statsmodels.graphics.regressionplots.plot_leverage_resid2.html)

# In[32]:


fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(lm, ax = ax)


# ---
# ---

# # 6. Multicollinearity.
# 
# Collinearity refers to the situation in which two or more predictor variables collinearity are closely related to one another.
# 
# The presence of collinearity can pose problems in the regression context, since it can be difficult to separate out the individual effects of collinear variables on the response. 
# 
# In other words, since limit and rating tend to increase or decrease together, it can be difficult to determine how each one separately is associated with the response, balance.

# In[33]:


# plot all variables in a scatter matrix
pd.plotting.scatter_matrix(df, alpha=0.8, figsize=(10, 10), diagonal='kde');


# In[34]:


# Inspect correlation
# Calculate correlation using the default method ( "pearson")
corr = df.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,  square=True, annot_kws={"size": 12});


# A simple way to detect collinearity is to look at the **correlation matrix** of the predictors. An element of this matrix that is large in absolute value indicates a pair of highly correlated variables, and therefore a collinearity problem in the data. 
# 
# Unfortunately, not all collinearity problems can be detected by inspection of the correlation matrix: it is possible for collinearity to exist between three or more variables even if no pair of variables has a particularly high correlation. We call this situation **multicollinearity**. 
# 
# Instead of inspecting the correlation matrix, a better way to assess multicollinearity is to compute the condition number test. If the condition number is above 30, the regression may have significant multicollinearity.

# In[35]:


# makes here no sense since we only have one predictor...
np.linalg.cond(lm.model.exog)


# Instead of inspecting the correlation matrix, a better way to assess multicollinearity is to compute the variance inflation factor (VIF). The smallest possible value for VIF is 1, which indicates the complete absence of collinearity. Typically in practice there is a small amount of collinearity among the predictors. As a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity.

# In[36]:


y, X = dmatrices('mpg ~ horsepower+ cylinders + displacement', df, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.round(2)


# When faced with the problem of collinearity, there are two simple solutions: 
# 
# The first is to drop one of the problematic variables from the regression. This can usually be done without much compromise to the regression fit, since the presence of collinearity implies that the information that this variable provides about the response is redundant in the presence of the other variables. 
# 
# The second solution is to combine the collinear variables together into a single predictor. For instance, we might take the average of standardized versions of limit and rating in order to create a new variable that measures credit worthiness.
