#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.05
# 
# The goal of this exercise is to evaluate the impact of feature preprocessing
# on a pipeline that uses a decision-tree-based classifier instead of logistic
# regression.
# 
# - The first question is to empirically evaluate whether scaling numerical
#   feature is helpful or not;
# - The second question is to evaluate whether it is empirically better (both
#   from a computational and a statistical perspective) to use integer coded or
#   one-hot encoded categories.

# In[ ]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")


# In[ ]:


target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])


# As in the previous notebooks, we use the utility `make_column_selector`
# to only select column with a specific data type. Besides, we list in
# advance all categories for the categorical columns.

# In[ ]:


from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)
numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)


# ## Reference pipeline (no numerical scaling and integer-coded categories)
# 
# First let's time the pipeline we used in the main notebook to serve as a
# reference:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import cross_validate\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OrdinalEncoder\nfrom sklearn.experimental import enable_hist_gradient_boosting\nfrom sklearn.ensemble import HistGradientBoostingClassifier\n\ncategorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",\n                                          unknown_value=-1)\npreprocessor = ColumnTransformer([\n    (\'categorical\', categorical_preprocessor, categorical_columns)],\n    remainder="passthrough")\n\nmodel = make_pipeline(preprocessor, HistGradientBoostingClassifier())\ncv_results = cross_validate(model, data, target)\nscores = cv_results["test_score"]\nprint("The mean cross-validation accuracy is: "\n      f"{scores.mean():.3f} +/- {scores.std():.3f}")')


# ## Scaling numerical features
# 
# Let's write a similar pipeline that also scales the numerical features using
# `StandardScaler` (or similar):

# In[ ]:


# Write your code here.


# ## One-hot encoding of categorical variables
# 
# For linear models, we have observed that integer coding of categorical
# variables can be very detrimental. However for
# `HistGradientBoostingClassifier` models, it does not seem to be the case as
# the cross-validation of the reference pipeline with `OrdinalEncoder` is good.
# 
# Let's see if we can get an even better accuracy with `OneHotEncoder`.
# 
# Hint: `HistGradientBoostingClassifier` does not yet support sparse input
# data. You might want to use
# `OneHotEncoder(handle_unknown="ignore", sparse=False)` to force the use of a
# dense representation as a workaround.

# In[ ]:


# Write your code here.

