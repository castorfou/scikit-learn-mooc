#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M6.01
# 
# The aim of this notebook is to investigate if we can tune the hyperparameters
# of a bagging regressor and evaluate the gain obtained.
# 
# We will load the California housing dataset and split it into a training and
# a testing set.

# In[1]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# Create a `BaggingRegressor` and provide a `DecisionTreeRegressor`
# to its parameter `base_estimator`. Train the regressor and evaluate its
# statistical performance on the testing set using the mean absolute error.

# In[16]:


# Write your code here.
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

bagged_trees = BaggingRegressor(
                base_estimator=DecisionTreeRegressor())
_ = bagged_trees.fit(data_train, target_train)

y_pred = bagged_trees.predict(data_test)
print(f'MAE: {mean_absolute_error(target_test, y_pred):0.02f} k$')


# Now, create a `RandomizedSearchCV` instance using the previous model and
# tune the important parameters of the bagging regressor. Find the best
# parameters  and check if you are able to find a set of parameters that
# improve the default regressor still using the mean absolute error as a
# metric.
# 
# <div class="admonition tip alert alert-warning">
# <p class="first admonition-title" style="font-weight: bold;">Tip</p>
# <p class="last">You can list the bagging regressor's parameters using the <tt class="docutils literal">get_params</tt>
# method.</p>
# </div>

# In[7]:


bagged_trees.get_params()


# In[12]:


import sklearn
sorted(sklearn.metrics.SCORERS.keys())


# In[19]:


# Write your code here.
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate
from scipy.stats import randint


param_grid = {
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 10, 30, 50],
    "min_samples_leaf": [0.01, 0.05, 0.1, 1]}

param_grid = {
    "n_estimators": randint(10, 30),
    "max_samples": [0.5, 0.8, 1.0],
    "max_features": [0.5, 0.8, 1.0],
    "base_estimator__max_depth": randint(3, 10),
}


search = RandomizedSearchCV(
    bagged_trees, param_grid, n_iter=20, scoring="neg_mean_absolute_error"
)
_ = search.fit(data_train, target_train)


# In[20]:


y_pred = search.predict(data_test)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(target_test, y_pred)


# In[21]:


import pandas as pd

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_score", "std_test_score", "rank_test_score"]
cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results[columns].sort_values(by="rank_test_score")
cv_results["mean_test_score"] = -cv_results["mean_test_score"]
cv_results


# In[22]:


target_predicted = search.predict(data_test)
print(f"Mean absolute error after tuning of the bagging regressor:\n"
      f"{mean_absolute_error(target_test, target_predicted):.2f} k$")


# We see that the bagging regressor provides a predictor in which fine tuning
# is not as important as in the case of fitting a single decision tree.
