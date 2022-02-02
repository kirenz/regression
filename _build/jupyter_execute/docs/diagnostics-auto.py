#!/usr/bin/env python
# coding: utf-8

# # Diagnostics
# 
# When we fit a linear regression model to a particular data set, many problems may occur. Most common among these are the following:
# 
# 1. Non-linearity of the response-predictor relationships 
# 2. Non-normally distributed errors
# 3. Correlation of error terms
# 4. Non-constant variance of error terms (heteroskedasticity)
# 5. High-leverage points
# 6. Multicollinearity
# 
# In many cases of statistical analysis, we are not sure whether our statistical model is correctly specified. For example when using OLS, linearity and homoscedasticity are assumed. Some test statistics additionally assume that the errors are normally distributed or that we have a large sample. Since our results depend on these statistical assumptions, the results are only correct if our assumptions hold (at least approximately).
# 
# Therefore, we need to test whether our sample is consistent with these assumptions, which we cover in this application. Alternatively, we could use robust methods, for example [robust regression](https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html#Using-robust-regression-to-correct-for-outliers.) or robust covariance estimators, which we'll not cover in this application. 
# 
# Let's use the Auto dataset to perform a linear regression:
# 
# - Dependent variable: `mpg` 
# - Features: `horsepower`, `weight` and `acceleration`  
# 
# Source:
# 
# - Statsmodels [regression diagnostic page](https://www.statsmodels.org/stable/diagnostic.html) 
# - [Statsmodel examples with code](https://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html)

# ## Python setup

# In[1]:


import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from patsy import dmatrices

import matplotlib.pyplot as plt
import seaborn as sns  

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=14)
sns.set() 


# ## Import data

# In[2]:


# Load the csv data files into pandas dataframes
ROOT = "https://raw.githubusercontent.com/kirenz/datasets/master/"
DATA = "Auto.csv"

df = pd.read_csv(ROOT + DATA)


# In[3]:


# show df
df


# In[4]:


df.info()


# ## Tidying data

# In[5]:


# change data type
df['origin'] = pd.Categorical(df['origin'])
df['year'] = pd.Categorical(df['year'], ordered=True)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce') 


# ### Handle missing values

# In[6]:


# show missing values (missing values - if present - will be displayed in yellow )
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# In[7]:


print(df.isnull().sum())


# In[8]:


# there are only 5 missing values therefore we simply delete the rows
df = df.dropna()


# In[9]:


print(df.isnull().sum())


# In[10]:


sns.pairplot(data=df);


# ## Regression model

# In[11]:


# fit linear model with statsmodels.formula.api (with R-style formulas) 
lm = smf.ols(formula ='mpg ~ horsepower + weight + acceleration', data=df).fit()
lm.summary()


# ## Diagnostics
# 
# ### plot_regress_exog
# 
# The `plot_regress_exog` function is a convenience function that gives a 2x2 plot containing 
# 
# 1. the dependent variable and fitted values with confidence intervals vs. the independent variable chosen, 
# 2. the residuals of the model vs. the chosen independent variable, 
# 3. a partial regression plot, 
# 4. and a CCPR plot. 
# 
# This function can be used for quickly checking modeling assumptions with respect to a single regressor ([see statsmodels documentation](https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html))

# In[12]:


fig = sm.graphics.plot_regress_exog(lm, "horsepower")
fig.tight_layout(pad=1.0)


# Let's take a look at the different plots.
# 
# 1. **Y and fitted vs X**: plots the fitted values versus a chosen independent variable. It includes prediction confidence intervals and plots the true dependent variable.
# 
# 
# 2. **Residuals versus horsepower**: The residuals are not equally spread around a horizontal line which is an indication for a non-linear relationship. This means there seems to be a non-linear relationship between the predictor and the response variable which the model doesn’t capture.
# 
# 3. **Partial regression plot**: attempts to show the effect of adding another variable to a model that already has one or more independent variables. 

# In[13]:


fig = sm.graphics.plot_fit(lm, "horsepower")
fig.tight_layout(pad=1.0)


# #### Residuals vs horsepower
# 
# 

# ### Non-linearity

# ####  Harvey-Collier multiplier test
# 
# Harvey-Collier multiplier test for Null hypothesis that the linear specification is correct. This test is a t-test that the mean of the recursive ols residuals is zero. 
# 
# A significant result (rejecting the null) occurs when the fit is better with a range restriction (which is what happens if the model is nonlinear).

# In[14]:


name = ['t value', 'p value']
test = sm.stats.linear_harvey_collier(lm)

# show result
lzip(name, test)


# ###  Residuals vs fitted plot
# 
# Residual plots are also a very useful graphical tool for identifying non-linearity:

# In[15]:


# fitted values
model_fitted_y = lm.fittedvalues

#  Plot
plot = sns.residplot(x=model_fitted_y, y='mpg', data=df, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

# Titel and labels
plot.set_title('Residuals vs Fitted')
plot.set_xlabel('Fitted values')
plot.set_ylabel('Residuals');


# The residuals are not equally spread around a horizontal line which is an indication for a **non-linear** relationship. 
# 
# This means there seems to be a non-linear relationship between the predictor and the response variable which the model doesn’t capture.

# **Advanced Plots:**
# 
# Besides basic plots, we will also cover some more advanced plots (similar to the R regression diagnostic plots) which flag certain observations. See here for a [description of the R diagnsotic plots](https://data.library.virginia.edu/diagnostic-plots/). 
# 
# The code for the advanced plots was obtained from [here](https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034) 

# Advanced Residuals vs fitted plot

# In[16]:


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


# In[17]:


# Advanced plot (1)
# figure size
plot = plt.figure(1)
plot.set_figheight(8)
plot.set_figwidth(12)
# generate figure with sns.residplot 
plot.axes[0] = sns.residplot(x=model_fitted_y, y='mpg', data=df, 
                          lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
# label axes
plot.axes[0].set_title('Residuals vs Fitted')
plot.axes[0].set_xlabel('Fitted values')
plot.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]));


# Deal with non-linearity 

# We can fit a non-linear function (polynomial regression)

# In[18]:



lm_2 = smf.ols(formula='mpg ~ horsepower + I(horsepower**2)', data=df).fit()
lm_2.summary()


# In[19]:


# fitted values
model_fitted_y_2 = lm_2.fittedvalues;
# Basic plot
plot = sns.residplot(x=model_fitted_y_2, y='mpg', data=df, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});

plot.set_title('Residuals vs Fitted');
plot.set_xlabel('Fitted values');
plot.set_ylabel('Residuals');


# ### Normality
# 
# It can be helpful if the residuals in the model are random, normally distributed variables with a mean of 0. 
# 
# This assumption means that the differences between the predicted and observed data are most frequently zero or very close to zero, and that differences much greater than zero happen only occasionally.
# 
# Some people confuse this assumption with the idea that predictors have to be normally distributed, which they don’t. In small samples a lack of normality invalidates confidence intervals and significance tests, whereas in large samples it will not because of the **central limit theorem**. 
# 
# If you are concerned only with estimating the model parameters (and not significance tests and confidence intervals) then this assumption barely matters. If you bootstrap confidence intervals then you can ignore this assumption.

# #### Jarque-Bera test
# 
# The Jarque–Bera test is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution. 
# 
# The null hypothesis is a joint hypothesis of the skewness being zero and the excess kurtosis being zero. 
# 
# Samples from a normal distribution have an expected skewness of 0 and an expected excess kurtosis of 0 (which is the same as a kurtosis of 3). As the definition of JB shows, any deviation from this increases the JB statistic.

# In[20]:


name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sm.stats.jarque_bera(lm.resid)

lzip(name, test)


# #### Omnibus normtest
# 
# Test for normal distribution of residuals. In this case, we use the $Chi^2$-Test. The Chi-Square Test for normality allows us to check whether or not the model residuals follow an approximately normal distribution.
# 
# Our null hypothesis is that the residuals are from a normal distribution.

# In[21]:


name = ['Chi^2', 'Two-tail probability']
test = sm.stats.omni_normtest(lm.resid)
lzip(name, test)


# ### Outliers
# 
# An outlier is a point for which $y_i$ is far from the value predicted by the model. Outliers can arise for a variety of reasons, such as incorrect recording of an observation during data collection.
# 
# In practice, it can be difficult to decide how large a residual needs to be before we consider the point to be an outlier. To address this problem, instead of plotting the residuals, we can plot the studentized residuals, computed by dividing each residual by its estimated standard error. 
# 
# Observations whose studentized residuals are greater than 3 in absolute value are possible outliers. 
# 
# If we believe that an outlier has occurred due to an error in data collection or recording, then one solution is to simply remove the observation. However, care should be taken, since an outlier may instead indicate a deficiency with the model, such as a missing predictor.

# #### Normal Q-Q-Plot
# 
# This plots the standardized (z-score) residuals against the theoretical normal quantiles. Anything quite off the diagonal lines may be a concern for further investigation.

# In[22]:


# Use standardized residuals
sm.qqplot(lm.get_influence().resid_studentized_internal);


# This plot shows if residuals are normally distributed. If a normal distribution is present, the residuals should (more or less) follow a straight line. 
# We can observe that only some residuals (in the lower left and the upper right corner) deviate from the straight line. 
# 
# We could obtain the 3 observations with the largest deviations from our advanced plot below (observations 330, 327 and 320). 

# **Advanced QQ-Plot**

# In[23]:


# Advanced plot (2)
# ProbPlot and its qqplot method from statsmodels graphics API. 
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, lw=1)
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


# ### Correlation
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

# In[24]:


sm.stats.durbin_watson(lm.resid)


# ### Variance of error terms
# 

# Another important assumption of the linear regression model is that the error terms have a constant variance. 
# 
# For instance, the variances of the error terms may increase with the value of the response. One can identify non-constant variances in
# the errors, or **heteroscedasticity**, from the presence of a funnel shape in the residual plot. 
# 
# When faced with this problem, one possible solution is to transform the response Y using a concave function such as log Y or √Y . Such a transformation results in a greater amount of shrinkage of the larger responses, leading to a reduction in heteroscedasticity.

# #### Breusch-Pagan test:
# 
# Test assumes homoskedasticity (null hypothesis). If one of the test statistics is significant, then you have evidence of heteroskedasticity. 

# In[25]:


name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sm.stats.het_breuschpagan(lm.resid, lm.model.exog)
lzip(name, test)


# #### Scale-Location plot

# In[26]:


# Scale Location plot
plt.scatter(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});


# This plot shows if residuals are spread equally along the ranges of predictors. This is how we can check the assumption of equal variance (**homoscedasticity**). It’s good if we observe a horizontal line with equally (randomly) spread points.
# 
# In our model the residuals begin to spread wider along the y-axis as it passes the x value of around 18. Because the residuals spread wider and wider with an increase of x, the red smooth line is not horizontal and shows a positive angle. This is an indication of **heteroskedasticity**.

# ### Leverage
# 
# We just saw that outliers are observations for which the response $y_i$ is unusual given the predictor $x_i$. In contrast, observations with high leverage have an unusual value for $x_i$. 
# 
# In a simple linear regression, high leverage observations are fairly easy to identify, since we can simply look for observations for which the predictor value is outside of the normal range of the observations. But in a multiple linear regression with many predictors, it is possible to have an observation that is well within the range of each individual predictor’s values, but that is unusual in terms of the full set of predictors.
# 
# A general rule of thumb is that observations with a **Cook’s D** over 4/n, where n is the number of observations, is an possible outlier with leverage. 

# #### Statsmodel influence
# 
# Once created, an object of class OLSInfluence holds attributes and methods that allow users to assess the influence of each observation. 

# In[27]:


# obtain statistics
infl = lm.get_influence()

lm_cooksd = lm.get_influence().cooks_distance[0]
n = len(df["name"])
critical_d = 4/n
print('Critical Cooks d:', critical_d)
#identification of potential outliers
out_d = lm_cooksd > critical_d
# Output potential outliers
df.index[out_d],lm_cooksd[out_d]


# In[28]:


# Show summary frame of leverage statistics
print(infl.summary_frame().filter(["student_resid","dffits","cooks_d"]))


# #### Residuals vs leverage

# Plots leverage statistics vs. normalized residuals squared. See [statsmodel documentation](http://www.statsmodels.org/0.6.1/generated/statsmodels.graphics.regressionplots.plot_leverage_resid2.html)

# In[29]:


fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(lm, ax = ax)


# ### Multicollinearity
# 
# Collinearity refers to the situation in which two or more predictor variables collinearity are closely related to one another.
# 
# The presence of collinearity can pose problems in the regression context, since it can be difficult to separate out the individual effects of collinear variables on the response. 

# In[30]:


# plot all variables in a scatter matrix
pd.plotting.scatter_matrix(df, alpha=0.8, figsize=(10, 10), diagonal='kde');


# In[31]:


# Inspect correlation
# Calculate correlation using the default method ( "pearson")
corr = df.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,  square=True, annot_kws={"size": 12});


# A simple way to detect collinearity is to look at the **correlation matrix** of the predictors. An element of this matrix that is large in absolute value indicates a pair of highly (linear) correlated variables, and therefore a collinearity problem in the data. 
# 
# Unfortunately, not all collinearity problems can be detected by inspection of the correlation matrix: it is possible for collinearity to exist between three or more variables even if no pair of variables has a particularly high correlation. We call this situation **multicollinearity**. 

# Instead of inspecting the correlation matrix, a better way to assess multicollinearity is to compute the variance inflation factor (VIF). The smallest possible value for VIF is 1, which indicates the complete absence of collinearity. Typically in practice there is a small amount of collinearity among the predictors. As a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity.

# In[32]:


X = df[['horsepower', 'cylinders', 'displacement']]

# the VIF function needs a constant
X = add_constant(X)

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.round(2)


# When faced with the problem of collinearity, there are two simple solutions: 
# 
# The first is to drop one of the problematic variables from the regression. This can usually be done without much compromise to the regression fit, since the presence of collinearity implies that the information that this variable provides about the response is redundant in the presence of the other variables. 
# 
# The second solution is to combine the collinear variables together into a single predictor.
