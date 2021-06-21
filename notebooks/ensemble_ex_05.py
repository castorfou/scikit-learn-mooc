#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M6.05
# 
# The aim of the exercise is to get familiar with the histogram
# gradient-boosting in scikit-learn. Besides, we will use this model within
# a cross-validation framework in order to inspect internal parameters found
# via grid-search.
# 
# We will use the California housing dataset.

# In[1]:


from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$


# First, create a histogram gradient boosting regressor. You can set the
# trees number to be large, and configure the model to use early-stopping.

# In[7]:


# Write your code here.
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

histogram_gradient_boosting = HistGradientBoostingRegressor(
    max_iter=1000, early_stopping=True)

histogram_gradient_boosting.fit(data, target)
histogram_gradient_boosting.n_iter_


# We will use a grid-search to find some optimal parameter for this model.
# In this grid-search, you should search for the following parameters:
# 
# * `max_depth: [3, 8]`;
# * `max_leaf_nodes: [15, 31]`;
# * `learning_rate: [0.1, 1]`.
# 
# Feel free to explore the space with additional values. Create the
# grid-search providing the previous gradient boosting instance as the model.

# In[10]:


from sklearn.model_selection import GridSearchCV
import pandas as pd

# Write your code here.
param_grid = {
    "max_depth": [3, 8],
    "max_leaf_nodes": [15, 31],
    "learning_rate": [0.1, 1],
}
grid_search = GridSearchCV(
    histogram_gradient_boosting, param_grid=param_grid,
    scoring="neg_mean_absolute_error", n_jobs=-1
)
grid_search.fit(data, target)

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_score", "rank_test_score"]
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results["mean_test_score"] = -cv_results["mean_test_score"]
cv_results[columns].sort_values(by="rank_test_score")


# Finally, we will run our experiment through cross-validation. In this regard,
# define a 5-fold cross-validation. Besides, be sure to shuffle the data.
# Subsequently, use the function `sklearn.model_selection.cross_validate`
# to run the cross-validation. You should also set `return_estimator=True`,
# so that we can investigate the inner model trained via cross-validation.

# In[29]:


# Write your code here.
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_results_hist = cross_validate(
    histogram_gradient_boosting, data, target,  cv=cv, 
    n_jobs=-1, return_estimator=True
)

print("Hist Gradient Boosting Decision Tree")
print(f"Mean absolute error via cross-validation: "
      f"{-cv_results_hist['test_score'].mean():.3f} +/- "
      f"{cv_results_hist['test_score'].std():.3f} k$")
print(f"Average fit time: "
      f"{cv_results_hist['fit_time'].mean():.3f} seconds")
print(f"Average score time: "
      f"{cv_results_hist['score_time'].mean():.3f} seconds")


# Now that we got the cross-validation results, print out the mean and
# standard deviation score.

# In[30]:


# Write your code here.


# Then, inspect the `estimator` entry of the results and check the best
# parameters values. Besides, check the number of trees used by the model.

# In[33]:


# Write your code here.
for est in cv_results_hist['estimator']:
#     print(est.best_params_)
    print(f"# trees: {est.best_estimator_.n_iter_}")


# Inspect the results of the inner CV for each estimator of the outer CV.
# Aggregate the mean test score for each parameter combination and make a box
# plot of these scores.

# In[ ]:


# Write your code here.

