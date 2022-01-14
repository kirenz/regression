#!/usr/bin/env python
# coding: utf-8

# # Spline regression
# 
# Patsy offers a set of specific stateful transforms (for more details about stateful transforms see Stateful transforms) that you can use in formulas to generate splines bases and express non-linear fits.
# 
# ## General B-splines
# 
# B-spline bases can be generated with the bs() stateful transform. The spline bases returned by bs() are designed to be compatible with those produced by the R bs function. The following code illustrates a typical basis and the resulting spline:

# In[3]:


import matplotlib.pyplot as plt
import numpy as np

plt.title("B-spline basis example (degree=3)");
x = np.linspace(0., 1., 100)

y = dmatrix("bs(x, df=6, degree=3, include_intercept=True) - 1", {"x": x})

b = np.array([1.3, 0.6, 0.9, 0.4, 1.6, 0.7])
plt.plot(x, y*b);
plt.plot(x, np.dot(y, b), color='k', linewidth=3);

