#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M4.04
# 
# In the previous notebook, we saw the effect of applying some regularization
# on the coefficient of a linear model.
# 
# In this exercise, we will study the advantage of using some regularization
# when dealing with correlated features.
# 
# We will first create a regression dataset. This dataset will contain 2,000
# samples and 5 features from which only 2 features will be informative.

# In[1]:


from sklearn.datasets import make_regression

data, target, coef = make_regression(
    n_samples=2_000, n_features=5, n_informative=2, shuffle=False,
    coef=True, random_state=0, noise=30,
)


# When creating the dataset, `make_regression` returns the true coefficient
# used to generate the dataset. Let's plot this information.

# In[2]:


import pandas as pd

feature_names = [f"Features {i}" for i in range(data.shape[1])]
coef = pd.Series(coef, index=feature_names)
coef.plot.barh()
coef


# Create a `LinearRegression` regressor and fit on the entire dataset and
# check the value of the coefficients. Are the coefficients of the linear
# regressor close to the coefficients used to generate the dataset?

# In[3]:


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(data, target)
linear_regression.coef_


# In[4]:


feature_names = [f"Features {i}" for i in range(data.shape[1])]
coef = pd.Series(linear_regression.coef_, index=feature_names)
_ = coef.plot.barh()


# We see that the coefficients are close to the coefficients used to generate
# the dataset. The dispersion is indeed cause by the noise injected during the
# dataset generation.

# Now, create a new dataset that will be the same as `data` with 4 additional
# columns that will repeat twice features 0 and 1. This procedure will create
# perfectly correlated features.

# In[20]:


# Write your code here.
# data['Features 5']=data['Features 0']
# data['Features 6']=data['Features 0']
# data['Features 7']=data['Features 1']
# data['Features 8']=data['Features 1']
new_data = [[dat[0], dat[1], dat[2], dat[3], dat[4], dat[0], dat[0], dat[1], dat[1]] for dat in data]
import numpy as np
new_data = np.array(new_data)
new_data.shape


# Fit again the linear regressor on this new dataset and check the
# coefficients. What do you observe?

# In[23]:


# Write your code here.
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(new_data, target)
print(f' coefficients: {linear_regression.coef_}')
feature_names = [f"Features {i}" for i in range(new_data.shape[1])]
coef = pd.Series(linear_regression.coef_, index=feature_names)
_ = coef.plot.barh()


# Create a ridge regressor and fit on the same dataset. Check the coefficients.
# What do you observe?

# In[33]:


# Write your code here.
from sklearn.linear_model import RidgeCV
model = RidgeCV( alphas=[0.001, 0.1, 1, 10, 1000], store_cv_values=True )
model.fit(new_data, target)
print(model.alpha_)
print(f' coefficients: {model.coef_}')
feature_names = [f"Features {i}" for i in range(new_data.shape[1])]
coef = pd.Series(model.coef_, index=feature_names)
_ = coef.plot.barh()


# Can you find the relationship between the ridge coefficients and the original
# coefficients?

# In[ ]:


# Write your code here.

