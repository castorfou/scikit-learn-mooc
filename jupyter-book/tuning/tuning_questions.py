#!/usr/bin/env python
# coding: utf-8

# # 🏁 Wrap-up quiz
# 
# **This quiz requires some programming to be answered.**

# Load the dataset file named `penguins.csv` with the following command:

# In[1]:


import pandas as pd


penguins = pd.read_csv("../datasets/penguins.csv")

columns = ["Body Mass (g)", "Flipper Length (mm)", "Culmen Length (mm)"]
target_name = "Species"

# Remove lines with missing values for the columns of interestes
penguins_non_missing = penguins[columns + [target_name]].dropna()

data = penguins_non_missing[columns]
target = penguins_non_missing[target_name]


# `penguins` is a pandas dataframe. The column "Species" contains the target
# variable. We extract through numerical columns that quantify various attributes
# of animals and our goal is try to predict the species of the animal based on
# those attributes stored in the dataframe named `data`.
# 
# Inspect the loaded data to select the correct assertions:

# # Question 1
# Select the correct assertions from the following proposals.
# 
# - a) The problem to be solved is a regression problem
# - b) The problem to be solved is a binary classification problem
#   (exactly 2 possible classes)
# - c) The problem to be solved is a multiclass classification problem
#   (more than 2 possible classes)
# - d) The proportions of the class counts are balanced: there are approximately
#   the same number of rows for each class
# - e) The proportions of the class counts are imbalanced: some classes have more
#   than twice as many rows than others)
# - f) The input features have similar dynamic ranges (or scales)
# 
# _Select several answers_
# 
# Hint: `data.describe()`, and `target.value_counts()` are methods
# that are helpful to answer to this question.

# In[2]:


data.describe()


# In[3]:


target.value_counts()


# 
# +++
# 
# Let's now consider the following pipeline:

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])


# # Question 2
# 
# Evaluate the pipeline using 10-fold cross-validation using the
# `balanced-accuracy` scoring metric to choose the correct statements.
# Use `sklearn.model_selection.cross_validate` with
# `scoring="balanced_accuracy"`.
# Use `model.get_params()` to list the parameters of the pipeline and use
# `model.set_params(param_name=param_value)` to update them.
# 
# - a) The average cross-validated test `balanced_accuracy` of the above pipeline is between 0.9 and 1.0
# - b) The average cross-validated test `balanced_accuracy` of the above pipeline is between 0.8 and 0.9
# - c) The average cross-validated test `balanced_accuracy` of the above pipeline is between 0.5 and 0.8
# - d) Using `n_neighbors=5` is much better than `n_neighbors=51`
# - e) Preprocessing with `StandardScaler` is much better than using the
#      raw features (with `n_neighbors=5`)
# 
# _Select several answers_

# In[25]:


from sklearn.model_selection import cross_validate, ShuffleSplit

model.set_params(classifier__n_neighbors=5)
cv = ShuffleSplit(random_state=0)
cv_results = cross_validate(model, data, target, return_train_score=True,
                            cv=cv, scoring='balanced_accuracy')
cv_results = pd.DataFrame(cv_results)
print(f'model {model.get_params()}')
print(f"{cv_results.mean()}\n")

model.set_params(classifier__n_neighbors=51)
cv_results = cross_validate(model, data, target, return_train_score=True,
                            cv=cv, scoring='balanced_accuracy')
cv_results = pd.DataFrame(cv_results)
print(f'model {model.get_params()}')
print(f"{cv_results.mean()}\n")

model.set_params(classifier__n_neighbors=5)
model.set_params(preprocessor__with_mean=False, preprocessor__with_std=False)
cv_results = cross_validate(model, data, target, return_train_score=True,
                            cv=cv, scoring='balanced_accuracy')
cv_results = pd.DataFrame(cv_results)
print(f'model {model.get_params()}')
print(f"{cv_results.mean()}\n")


# +++
# 
# We will now study the impact of different preprocessors defined in the list below:

# In[27]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


all_preprocessors = [
    None,
    StandardScaler(),
    MinMaxScaler(),
    QuantileTransformer(n_quantiles=100),
    PowerTransformer(method="box-cox"),
]


# The [Box-Cox
# method](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation)
# is common preprocessing strategy for positive values. The other preprocessors
# work both for any kind of numerical features. If you are curious to read the
# details about those method, please feel free to read them up in the
# [preprocessing
# chapter](https://scikit-learn.org/stable/modules/preprocessing.html) of the
# scikit-learn user guide but this is not required to answer the quiz questions.

# # Question 3
# 
# Use `sklearn.model_selection.GridSearchCV` to study the impact of the choice of
# the preprocessor and the number of neighbors on the 10-fold cross-validated
# `balanced_accuracy` metric. We want to study the `n_neighbors` in the range
# `[5, 51, 101]` and `preprocessor` in the range `all_preprocessors`.
# 
# Let us consider that a model is significantly better than another if the its
# mean test score is better than the mean test score of the alternative by more
# than the standard deviation of its test score.
# 
# Which of the following statements hold:
# 
# - a) The best model with `StandardScaler` is significantly better than using any other processor
# - b) Using any of the preprocessors is always better than using no processor, irrespective of the value `of n_neighbors`
# - c) The model with `n_neighbors=5` and `StandardScaler` is significantly better than the model with `n_neighbors=51` and `StandardScaler`.
# - d) The model with `n_neighbors=51` and `StandardScaler` is significantly better than the model with `n_neighbors=101` and `StandardScaler`.
# 
# Hint: pass `{"preprocessor": all_preprocessors, "classifier__n_neighbors": [5, 51, 101]}` for the `param_grid` argument to the `GridSearchCV` class.
# 
# _Select several answers_
# 

# In[38]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

param_grid = {"preprocessor": all_preprocessors, "classifier__n_neighbors": [5, 51, 101]}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=-1, cv=10, scoring='balanced_accuracy')

model_grid_search.fit(data, target)
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
cv_results.head()
column_results = [f"param_{name}" for name in param_grid.keys()]
column_results += [
    "mean_test_score", "std_test_score", "rank_test_score"]
cv_results = cv_results[column_results]


# In[39]:


cv_results


# In[ ]:




