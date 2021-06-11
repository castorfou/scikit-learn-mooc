#!/usr/bin/env python
# coding: utf-8

# # 🏁 Wrap-up quiz
# 
# **This quiz requires some programming to be answered.**
# 
# Open the dataset `house_prices.csv` with the following command:

# In[1]:


import pandas as pd

ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values="?")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]


# `ames_housing` is a pandas dataframe. The column "SalePrice" contains the
# target variable. Note that we instructed pandas to treat the character "?" as a
# marker for cells with missing values also known as "null" values.
# 
# To simplify this exercise, we will only used the numerical features defined
# below:

# In[2]:


numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]


# We will compare the statistical performance of a decision tree and a linear
# regression. For this purpose, we will create two separate predictive models
# and evaluate them by 10-fold cross-validation.
# 
# Thus, use `sklearn.linear_model.LinearRegression` and
# `sklearn.tree.DecisionTreeRegressor` to create the model. Use the default
# parameters for both models.
# 
# **Note**: missing values should be handle with a scikit-learn
# `sklearn.impute.SimpleImputer` and the default strategy (`"mean"`). Be also
# aware that a linear model requires to scale the data. You can use a
# `sklearn.preprocessing.StandardScaler`.

# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_validate

linear_model = make_pipeline(StandardScaler(), SimpleImputer(), LinearRegression())
tree_model = make_pipeline(SimpleImputer(), DecisionTreeRegressor())

linear_cv_result = cross_validate(linear_model, data_numerical, target, cv=10, return_estimator=True, scoring='r2')
print(f'Score for linear model is {linear_cv_result["test_score"].mean():0.2f} +/- {linear_cv_result["test_score"].std():0.2f}')

tree_cv_result = cross_validate(tree_model, data_numerical, target, cv=10, return_estimator=True, scoring='r2')
print(f'Score for tree model is {tree_cv_result["test_score"].mean():0.2f} +/- {tree_cv_result["test_score"].std():0.2f}')


# # Question 1
# Is the decision tree model better in terms of $R^2$ score than the linear
# regression?
# 
# - a) Yes
# - b) No
# 
# _Select a single answer_

# Instead of using the default parameter for decision tree regressor, we will
# optimize the depth of the tree. Using a grid-search
# (`sklearn.model_selection.GridSearchCV`) with a 10-fold cross-validation,
# answer to the questions below. Vary the `max_depth` from 1
# level up to 15 levels.

# In[10]:


tree_model.get_params()


# In[15]:


from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {"decisiontreeregressor__max_depth": np.arange(1, 15, 1)}
tree_reg = GridSearchCV(tree_model, param_grid=param_grid, cv=10, scoring='r2')
tree_reg.fit(data_numerical, target)


# In[17]:


tree_reg.best_params_['decisiontreeregressor__max_depth']


# # Question 2
# What is the optimal tree depth for the current problem?
# 
# - a) The optimal depth is ranging from 3 to 5
# - b) The optimal depth is ranging from 5 to 8
# - c) The optimal depth is ranging from 8 to 11
# - d) The optimal depth is ranging from 11 to 15
# 
# _Select a single answer_
# 

# In[19]:


tree_reg.best_score_


# # Question 3
# A tree with an optimal depth is performing:
# 
# - a) better than a linear model
# - b) equally to a linear model
# - c) worse than a linear model
# 
# _Select a single answer_

# Instead of using only the numerical dataset (which was the variable
# `data_numerical`), use the entire dataset available in the variable `data`.
# 
# Create a preprocessor by dealing separately with the numerical and categorical
# columns. For the sake of simplicity, we will define the categorical columns as
# the columns with an `object` data type.
# 
# **Do not optimize the `max_depth` parameter for this exercise.**

# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_validate

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


from sklearn.compose import make_column_selector as selector

from sklearn.pipeline import Pipeline

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent", verbose=1)),
    ('onehotencoder', OneHotEncoder(handle_unknown = 'ignore'))])

linear_preprocessor = ColumnTransformer(transformers=[
     ("num-preprocessor", numeric_transformer, numerical_features),
    ("cat-preprocessor", categorical_transformer, categorical_columns)
])

tree_preprocessor = ColumnTransformer(transformers=[
     ("num-preprocessor", SimpleImputer(), numerical_features),
    ("cat-preprocessor", categorical_transformer, categorical_columns)
])


linear_model = make_pipeline(linear_preprocessor, LinearRegression())
linear_cv_result = cross_validate(linear_model, data, target, cv=10, return_estimator=True, scoring='r2')

print(f'Score is {linear_cv_result["test_score"].mean():0.2f} +/- {linear_cv_result["test_score"].std():0.2f}')


tree_model = make_pipeline(tree_preprocessor, DecisionTreeRegressor())
tree_cv_result = cross_validate(tree_model, data, target, cv=10, return_estimator=True, scoring='r2')
print(f'Score for tree model is {tree_cv_result["test_score"].mean():0.2f} +/- {tree_cv_result["test_score"].std():0.2f}')



# # Question 4
# Are the performance in terms of $R^2$ better by incorporating the categorical
# features in comparison with the previous tree with the optimal depth?
# 
# - a) No the statistical performance are the same: ~0.7
# - b) The statistical performance is slightly better: ~0.72
# - c) The statistical performance is better: ~0.74
# 
# _Select a single answer_
# 

# In[ ]:




