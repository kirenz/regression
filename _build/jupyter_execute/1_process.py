#!/usr/bin/env python
# coding: utf-8

# # Programming Process
# 
# The goal of this section is to give you a first impression of some important steps and tools in Python when using linear regression models. A typical project looks something like this {cite:p}`Wickham2016`:
# 

# 
# 
# ```{image} _static/img/data-science.png
# :alt: datascience
# :class: bg-primary mb-1
# :width: 600px
# :align: center
# ```

# First you must **import** your data into Python. This typically means that you take data stored in a file, database, or web API, and load it into a DataFrame in Python (using Pandas).
# 
# Once you’ve imported your data, it is a good idea to tidy it.
# **Tidying** your data means storing it in a consistent form that matches the semantics of the dataset with
# the way it is stored. In brief, when your data is tidy, each column is a variable, and each row is an observation. Tidy data is important because the consistent structure lets you focus your struggle on questions about the data.
# 
# Once you have tidy data, a common first step is to transform it. **Transformation** includes narrowing in on observations of interest (like all people in one city, or all data from the last year), creating new variables that are functions of existing variables (like computing velocity from speed and time), and calculating a set of summary statistics (like counts or means). Together, tidying and transforming are called wrangling, because getting your data in a form that’s natural to work with often feels like a fight!
# 
# Once you have tidy data with the variables you need, there are two main engines of knowledge generation: visualisation and modelling. These have complementary strengths and weaknesses so any real analysis will iterate between them many times. **Visualisation** is a fundamentally human activity. A good visualisation will show you things that you did not expect, or raise new questions about the data. A good visualisation might also hint that you’re asking the wrong question, or you need to collect different data. Visualisations can surprise you, but don’t scale particularly well because they require a human to interpret them.
# 
# **Models** are complementary tools to visualisation. Once you have made your questions sufficiently precise,
# you can use a model to answer them. Models are a fundamentally mathematical or computational tool,
#  so they generally scale well. 
# 
# The last step is **communication**, an absolutely critical part of any data analysis project.
# It doesn't matter how well your models and visualisation have led you to understand the data unless
# you can also communicate your results to others.
# 
# 
# Surrounding all the data science steps covered above is **programming**. Programming is a cross-cutting tool that you use in every part of the project. You don’t need to be an expert programmer to be a data scientist, but learning more about programming pays off because becoming a better programmer allows you to automate
# common tasks, and solve new problems with greater ease.
# 
# To demonstrate the programming process, we examine a dataset which contains variables that could relate to **wages** for a group of males. The data is obtained from James et al. (2021) and consists of 12 variables for 3,000 people, so we have n = 3,000 observations and p = 12 variables (such as year, age, and more).
# 
# First, we take a look at some important Python modules and functions.
# 
# **Module overview:**
# 
#   * **Pandas** provides a powerful set of methods to manipulate, filter, group, and
# transform data. To learn more about pandas, review this [short introduction to pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html), geared mainly for new users. You can see more complex recipes in the Cookbook.
# 
#   * **Seaborn** is a Python data visualization library.

# In[1]:


# Customarily, we import as follows:
import pandas as pd
import seaborn as sns

# seaborn settings
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, palette='winter')

# show plots in jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import and inspect data

# In[2]:


# Load csv data from GitHub into pandas dataframes
ROOT = "https://raw.githubusercontent.com/kirenz/datasets/master/"
DATA = "wage.csv"

df = pd.read_csv(ROOT + DATA)


# Let's take a look at the variables (also called columns or features) in the data set.

# In[3]:


# show the first rows (i.e. head of the DataFrame)
df.head()


# In[4]:


# show the last rows (i.e. tail of the DataFrame)
df.tail()


# In[5]:


# show all variables in the data set
df.columns


# In[6]:


# data overview (with meta data)
df.info()


# ## Tidying data
# 
# The variable "Unnamed: 0" seems to be some kind of identification number per employee. Let's rename the variable and check if we have duplicates in our dataset.

# In[7]:


# rename variable "Unnamed: 0" to "id"
df = df.rename(index=str, columns={"Unnamed: 0": "id"})


# In Jupyter, we have several options to present output. One way is to just run a cod block
# which produces only one output:

# In[8]:


# show the length of the variable id (i.e. the number of observations)
len(df["id"])


# If we need to present multiple outputs or add some text, we can use the print()
# function. See the [Python documentation](https://docs.python.org/3/tutorial/inputoutput.html) for the
# different `print()` options.

# In[9]:


# check for duplicates and print results (if the two numbers match, we have no duplicates)
# show the length of the variable id (i.e. the number of observations)
print("IDs:", len(df["id"]))

# count the number of individual id's
print("Unique IDs:", len(df["id"].value_counts()))


# Suppose we already decided to only use a specific set of variables (`wage`, `year`, `age` and `education`), we can select them
# and drop the other variables.
# 
# Note that the variable selection process should always be based on solid theories and other insights.
# We also keep the `id`-Variable to identify the observations and to be able to merge
# the data with one of the dropped variables if necessary.

# In[10]:


# select variables
df = df[['id','year', 'age', 'education', 'wage']]


# In[11]:


# data overview (with meta data)
df.info()


# In[12]:


# show the first 3 rows
df.head(3)


# In[13]:


# rename variable "education" to "edu"
df = df.rename(index=str, columns={"education": "edu"})


# In[14]:


# check levels and frequency of edu
df['edu'].value_counts() 


# ### Data types
# 
# Pandas offers different options to change the data type of a variable.
# 
# To change data into a categorical format, you can use the following code (see this
# [pandas tutorial to learn more about categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html)):
# 
# `df['variable'] = pd.Categorical(df['variable'])`

# In[15]:


# convert to categorical (nominal)
df['id'] = pd.Categorical(df['id'])
df['year'] = pd.Categorical(df['year'])


# If we need to convert to a ordinal variable with pandas [CategoricalDtype](https://pandas.pydata.org/pandas-docs/stable/categorical.html)

# In[16]:


# convert to ordinal
cat_edu = pd.CategoricalDtype(categories=['1. < HS Grad',
                             '2. HS Grad', 
                             '3. Some College', 
                             '4. College Grad', 
                             '5. Advanced Degree'],
                            ordered=True)

df.edu = df.edu.astype(cat_edu)


# In[17]:


# show levels
df['edu'].cat.categories


# In[18]:


# show datatype
df.dtypes


# If we need to transform variables into a **numerical format**, we can transform the data with
# `pd.to_numeric` [(see Pandas documentation)](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_numeric.html):
# 
# If the numeric data contains strings, we need to replace them with NaN (not a number).
# Otherwise, we get an error message. Therefore, use errors='coerce' ...
# 
# `pandas.to_numeric(arg, errors='coerce', downcast=None)`
# 
# The options to handle errors are as follows:
# 
# - errors : {‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
#     - If `raise`, then invalid parsing will raise an exception
#     - If `coerce`, then invalid parsing will be set as NaN
#     - If `ignore`, then invalid parsing will return the input
# 
# 
# 

# ### Handle missing values
# 
# Next, we need to check if there are missing cases in the data set. By “missing” we simply mean NA (“not available”).
# 
# Many datasets arrive with missing data, either because it exists but was not collected or it never existed.
# Having missing values in a dataset can cause errors with some algorithms. Therefore, we need to take care of this issue (we cover the topic of missing values in one of the following applications in detail).

# In[19]:


# show missing values (missing values - if present - will be displayed in yellow )
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis');


# We can also check the column-wise distribution of null values:

# In[20]:


print(df.isnull().sum())


# In this dataset, there are no missing values present.
# 
# If we observe **missing values** in the df, we could drop them all together with this code:
# 
# `df = df.dropna()`
# 
# However, be careful not to drop many observations if just one variable is the cause for the
#  missing values. In that case, it could be reasonable to only drop the variable:
# 
# `df = df.drop('variable', axis=1)`
# 
# Usually, we will use other methods to fill in missing values (e.g. imputate missing values by using the mean)

# ## Transform data

# ### Descriptive statistics

# First, we obtain some common statistics:

# In[21]:


# summary statistics for all numerical columns
df.describe()


# In[22]:


# summary statistics for all categorical columns
df.describe(include=['category'])


# Compare summary statistics for specific groups in the data:

# In[23]:


# summary statistics by groups
df['age'].groupby(df['edu']).describe()


# Some examples of how to calculate simple statistics:

# In[24]:


# calculation of the mean (e.g. for age)
age_mean = df["age"].mean()

# calculation of the median (e.g. for age)
age_median =  df["age"].median()

# print the result (e.g., age_mean)
print('The precise mean of age is', age_mean)

# print the rounded result
print('The rounded mean of age is', round(age_mean))

# print the round result (to two decimals) (this is the preferred option)
print('The rounded mean of age with two decimals is', round(age_mean, 2))

# use a function inside print()
print('The median of age is', df["age"].median())


# In[25]:


# calculation of the mode
df['age'].mode()


# In[26]:


# quantiles
df['age'].quantile([.25, .5, .75])


# In[27]:


# Range
df['age'].max() - df['age'].min()


# In[28]:


# standard deviation
round(df['age'].std(),2)


# ## Visualize data

# ### Distribution

# How you visualize the distribution of a variable will depend on whether the variable is categorical or continuous.The excellent site [From Data to Viz](https://www.data-to-viz.com/) leads you to the most appropriate graph for your data. It also links to the code to build it and lists common caveats you should avoid.
# 
# For example, to examine the distribution of a categorical variable, we could use a bar or count plot.

# In[29]:


# horizontal count plot (show the counts of observations in each categorical bin)
sns.countplot(y='edu', data=df);


# A variable is **continuous** if it can take any of an infinite set of ordered values. To examine the distribution of a continuous variable, we could use a **histogram**:

# In[30]:


# Pandas histogram of all numerical values
df.hist();


# We can also use Seaborne to plot a histogram with a kernel density estimate:

# In[31]:


# histogram with seaborn 
sns.displot(x='age', data=df, kde=True);


# Another alternative to display the distribution of a continuous variable broken down by a categorical variable is the **boxplot**. 
# 
# Each boxplot consists of:
# 
# - A box that stretches from the 25th percentile of the distribution to the 75th percentile, a distance known as the interquartile range (IQR).
# 
# - In the middle of the box is a line that displays the median, i.e. 50th percentile, of the distribution.
# 
# - These three lines give you a sense of the spread of the distribution
# 
# - Visual points that display observations that fall more than 1.5 times the IQR from either edge of the box. These outlying points are unusual so are plotted individually.
# 
# - A line (or whisker) that extends from each end of the box and goes to the
# farthest non-outlier point in the distribution.

# In[32]:


# boxplot 
sns.boxplot(y='age', data=df);


# In[33]:


# boxplot for different groups
sns.boxplot(y='edu', x='age', data=df);


# ### Relationship

# A great way to visualise the covariation between two continuous variables is to draw a scatterplot.
# You can see covariation as a pattern in the points. We will cover more options to test for relationships in variables
# (e.g., correlation) in the following applications.

# In[34]:


# simple scatterplot
sns.scatterplot(x='age', y='wage', data=df);


# In[35]:


# plot all numeric variables in pairs
sns.pairplot(df);


# ## Model

# We will cover the modelling process in one of the next sections.
# 
