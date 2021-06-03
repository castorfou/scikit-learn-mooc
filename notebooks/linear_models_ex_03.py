#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M4.03
# 
# In all previous notebooks, we only used a single feature in `data`. But we
# have already shown that we could add new features to make the model more
# expressive by deriving new features, based on the original feature.
# 
# The aim of this notebook is to train a linear regression algorithm on a
# dataset more than a single feature.
# 
# We will load a dataset about house prices in California.
# The dataset consists of 8 features regarding the demography and geography of
# districts in California and the aim is to predict the median house price of
# each district. We will use all 8 features to predict the target, the median
# house price.

# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# In[1]:


from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$
data.head()


# Now this is your turn to train a linear regression model on this dataset.
# You will need to:
# * create a linear regression model;
# * execute a cross-validation with 10 folds and use the mean absolute error
#   (MAE) as metric. Ensure to return the fitted estimators;
# * compute mean and std of the MAE in thousands of dollars (k$);
# * show the values of the coefficients for each feature using a boxplot by
#   inspecting the fitted model returned from the cross-validation. Hint: you
#   use the function
#   [`df.plot.box()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.box.html)
#   to plot a box plot.

# In[14]:


# Write your code here.: make the exercise
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
import numpy as np 

linear_regression = LinearRegression()

cv_results = cross_validate(
    linear_regression, data, target, cv=10, scoring="neg_mean_absolute_error", return_estimator=True)
print(f'Mean absolute error {np.mean(-cv_results["test_score"]):0.02f} k$,  +/- {np.std(cv_results["test_score"]):0.02f} k$')


# In[34]:


cv_results['estimator']
coef = [model.coef_ for model in cv_results['estimator']]
coef_detailed = []
for ligne in range(len(coef)):
    coef_detailed.append([cof for cof in coef[ligne]])
intercept = [model.intercept_ for model in cv_results['estimator']]
import pandas as pd
full_data = pd.DataFrame(coef_detailed)


# In[36]:


full_data.plot.box()


# # correction bcp plus smart

# In[37]:


import pandas as pd

weights = pd.DataFrame(
    [est.coef_ for est in cv_results["estimator"]], columns=data.columns)

import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
weights.plot.box(color=color, vert=False)
_ = plt.title("Value of linear regression coefficients")


# In[ ]:




