#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M3.02
# 
# The goal is to find the best set of hyperparameters which maximize the
# statistical performance on a training set.
# 
# Here again with limit the size of the training set to make computation
# run faster. Feel free to increase the `train_size` value if your computer
# is powerful enough.

# In[1]:


import numpy as np
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)


# Create your machine learning pipeline
# 
# You should:
# * preprocess the categorical columns using a `OneHotEncoder` and use a
#   `StandardScaler` to normalize the numerical data.
# * use a `LogisticRegression` as a predictive model.

# Start by defining the columns and the preprocessing pipelines to be applied
# on each columns.

# In[2]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

categorical_preprocessor = OneHotEncoder()
numerical_preprocessor = StandardScaler()


# Subsequently, create a `ColumnTransformer` to redirect the specific columns
# a preprocessing pipeline.

# In[3]:


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

preprocessor = ColumnTransformer([
    ('cat-preprocessor', categorical_preprocessor, selector(dtype_include=object)),
    ('num-preprocessor', numerical_preprocessor, selector(dtype_include='number'))],
    remainder='passthrough', sparse_threshold=0)


# Finally, concatenate the preprocessing pipeline with a logistic regression.

# In[4]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(preprocessor, LogisticRegression())


# Use a `RandomizedSearchCV` to find the best set of hyperparameters by tuning
# the following parameters of the `model`:
# 
# - the parameter `C` of the `LogisticRegression` with values ranging from
#   0.001 to 10. You can use a log-uniform distribution
#   (i.e. `scipy.stats.loguniform`);
# - the parameter `with_mean` of the `StandardScaler` with possible values
#   `True` or `False`;
# - the parameter `with_std` of the `StandardScaler` with possible values
#   `True` or `False`.
# 
# Once the computation has completed, print the best combination of parameters
# stored in the `best_params_` attribute.

# In[6]:


from sklearn import set_config
set_config(display='diagram')
model


# In[7]:


model.get_params()


# In[26]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# the parameter C of the LogisticRegression with values ranging from 0.001 to 10. You can use a log-uniform distribution (i.e. scipy.stats.loguniform);
# the parameter with_mean of the StandardScaler with possible values True or False;
# the parameter with_std of the StandardScaler with possible values True or False.
param_distributions = {
    "logisticregression__C": loguniform(0.001, 10),
    "columntransformer__num-preprocessor__with_mean": [True, False],
    "columntransformer__num-preprocessor__with_std": [True, False],
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions,
    n_iter=20, error_score=np.nan, n_jobs=-1, verbose=1)
model_random_search.fit(data_train, target_train)
model_random_search.best_params_


# In[ ]:




