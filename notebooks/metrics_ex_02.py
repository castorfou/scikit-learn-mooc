#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M7.03
# 
# As with the classification metrics exercise, we will evaluate the regression
# metrics within a cross-validation framework to get familiar with the syntax.
# 
# We will use the Ames house prices dataset.

# In[1]:


import pandas as pd
import numpy as np

ames_housing = pd.read_csv("../datasets/house_prices.csv")
data = ames_housing.drop(columns="SalePrice")
target = ames_housing["SalePrice"]
data = data.select_dtypes(np.number)
target /= 1000


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# The first step will be to create a linear regression model.

# In[2]:


# Write your code here.
from sklearn.linear_model import LinearRegression

lr = LinearRegression()


# Then, use the `cross_val_score` to estimate the statistical performance of
# the model. Use a `KFold` cross-validation with 10 folds. Make the use of the
# $R^2$ score explicit by assigning the parameter `scoring` (even though it is
# the default score).

# In[4]:


# Write your code here.
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=10)

test_score = cross_val_score(lr, data, target, cv=cv, scoring='r2')
print(f'tests score {test_score.mean():0.02f} +/- {test_score.std():0.02f}')


# Then, instead of using the $R^2$ score, use the mean absolute error. You need
# to refer to the documentation for the `scoring` parameter.

# In[5]:


# Write your code here.
test_score = cross_val_score(lr, data, target, cv=cv, scoring='neg_mean_absolute_error')
print(f'tests score {-test_score.mean():0.02f} +/- {-test_score.std():0.02f}')


# Finally, use the `cross_validate` function and compute multiple scores/errors
# at once by passing a list of scorers to the `scoring` parameter. You can
# compute the $R^2$ score and the mean absolute error for instance.

# In[8]:


# Write your code here.
from sklearn.model_selection import cross_validate

test_score = cross_validate(lr, data, target, cv=cv, scoring=['r2', 'neg_mean_absolute_error'])
print(test_score)


# In[ ]:




