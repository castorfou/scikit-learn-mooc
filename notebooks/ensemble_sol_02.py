#!/usr/bin/env python
# coding: utf-8

# # 📃 Solution for Exercise M6.02
# 
# The aim of this exercise it to explore some attributes available in
# scikit-learn's random forest.
# 
# First, we will fit the penguins regression dataset.

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("../datasets/penguins_regression.csv")
feature_names = ["Flipper Length (mm)"]
target_name = "Body Mass (g)"
data, target = penguins[feature_names], penguins[target_name]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# Create a random forest containing three trees. Train the forest and
# check the statistical performance on the testing set in terms of mean
# absolute error.

# In[2]:


from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=3)
forest.fit(data_train, target_train)
target_predicted = forest.predict(data_test)
print(f"Mean absolute error: "
      f"{mean_absolute_error(target_test, target_predicted):.3f} grams")


# The next steps of this exercise are to:
# 
# - create a new dataset containing the penguins with a flipper length
#   between 170 mm and 230 mm;
# - plot the training data using a scatter plot;
# - plot the decision of each individual tree by predicting on the newly
#   created dataset;
# - plot the decision of the random forest using this newly created dataset.
# 
# <div class="admonition tip alert alert-warning">
# <p class="first admonition-title" style="font-weight: bold;">Tip</p>
# <p class="last">The trees contained in the forest that you created can be accessed
# with the attribute <tt class="docutils literal">estimators_</tt>.</p>
# </div>

# In a first cell, we will collect all the required predictions from the
# different trees and forest.

# In[3]:


import numpy as np

data_ranges = pd.DataFrame(np.linspace(170, 235, num=300),
                           columns=data.columns)
tree_predictions = []
for tree in forest.estimators_:
    tree_predictions.append(tree.predict(data_ranges))

forest_predictions = forest.predict(data_ranges)


# Now, we can plot the predictions that we collected.

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=penguins, x=feature_names[0], y=target_name,
                color="black", alpha=0.5)

# plot tree predictions
for tree_idx, predictions in enumerate(tree_predictions):
    plt.plot(data_ranges, predictions, label=f"Tree #{tree_idx}",
             linestyle="--", alpha=0.8)

plt.plot(data_ranges, forest_predictions, label=f"Random forest")
_ = plt.legend()


# In[ ]:




