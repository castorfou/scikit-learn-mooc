#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M3.01
# 
# The goal is to write an exhaustive search to find the best parameters
# combination maximizing the model statistical performance.
# 
# Here we use a small subset of the Adult Census dataset to make to code
# fast to execute. Once your code works on the small subset, try to
# change `train_size` to a larger value (e.g. 0.8 for 80% instead of
# 20%).

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42)


# In[2]:


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer(
    [('cat-preprocessor', categorical_preprocessor,
      selector(dtype_include=object))],
    remainder='passthrough', sparse_threshold=0)

# This line is currently required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42))
])


# 
# Use the previously defined model (called `model`) and using two nested `for`
# loops, make a search of the best combinations of the `learning_rate` and
# `max_leaf_nodes` parameters. In this regard, you will need to train and test
# the model by setting the parameters. The evaluation of the model should be
# performed using `cross_val_score`. We will use the following parameters
# search:
# - `learning_rate` for the values 0.01, 0.1, 1 and 10. This parameter controls
#   the ability of a new tree to correct the error of the previous sequence of
#   trees
# - `max_leaf_nodes` for the values 3, 10, 30. This parameter controls the
#   depth of each tree.

# In[3]:


for parameter in model.get_params():
    print(parameter)


# In[5]:


# Write your code here.
from sklearn.model_selection import cross_val_score


learning_rate_range = [0.01, 0.1, 1, 10]
max_leaf_nodes_range = [3, 10, 30]

for learning_rate in learning_rate_range:
    for max_leaf_nodes in max_leaf_nodes_range:
        model.set_params(classifier__learning_rate=learning_rate)
        model.set_params(classifier__max_leaf_nodes=max_leaf_nodes)
        scores = cross_val_score(model, data_test, target_test)
        print(f"Accuracy score via cross-validation using learning_rate:{learning_rate}, max_leaf_nodes:{max_leaf_nodes}:\n"
              f"{scores.mean():.3f} +/- {scores.std():.3f}")


# # correction

# In[6]:


from sklearn.model_selection import cross_val_score

learning_rate = [0.01, 0.1, 1, 10]
max_leaf_nodes = [3, 10, 30]

best_score = 0
best_params = {}
for lr in learning_rate:
    for mln in max_leaf_nodes:
        print(f"Evaluating model with learning rate {lr:.3f}"
              f" and max leaf nodes {mln}... ", end="")
        model.set_params(
            classifier__learning_rate=lr,
            classifier__max_leaf_nodes=mln
        )
        scores = cross_val_score(model, data_train, target_train, cv=2)
        mean_score = scores.mean()
        print(f"score: {mean_score:.3f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'learning-rate': lr, 'max leaf nodes': mln}
            print(f"Found new best model with score {best_score:.3f}!")

print(f"The best accuracy obtained is {best_score:.3f}")
print(f"The best parameters found are:\n {best_params}")


# In[ ]:




