#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M7.02
# 
# We presented different classification metrics in the previous notebook.
# However, we did not use it with a cross-validation. This exercise aims at
# practicing and implementing cross-validation.
# 
# We will reuse the blood transfusion dataset.

# In[1]:


import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# First, create a decision tree classifier.

# In[2]:


# Write your code here.
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()


# Create a `StratifiedKFold` cross-validation object. Then use it inside the
# `cross_val_score` function to evaluate the decision tree. We will first use
# the accuracy as a score function. Explicitly use the `scoring` parameter
# of `cross_val_score` to compute the accuracy (even if this is the default
# score). Check its documentation to learn how to do that.

# In[3]:


# Write your code here.

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold()
test_score = cross_val_score(dt, data, target, cv=cv)

print(f'Test score {test_score.mean():0.02f} +/- {test_score.std():0.02f}')


# Repeat the experiment by computing the `balanced_accuracy`.

# In[4]:


# Write your code here.


cv = StratifiedKFold()
test_score = cross_val_score(dt, data, target, cv=cv, scoring='balanced_accuracy')

print(f'Test score {test_score.mean():0.02f} +/- {test_score.std():0.02f}')


# We will now add a bit of complexity. We would like to compute the precision
# of our model. However, during the course we saw that we need to mention the
# positive label which in our case we consider to be the class `donated`.
# 
# We will show that computing the precision without providing the positive
# label will not be supported by scikit-learn because it is indeed ambiguous.

# In[5]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
try:
    scores = cross_val_score(tree, data, target, cv=10, scoring="precision")
except ValueError as exc:
    print(exc)


# <div class="admonition tip alert alert-warning">
# <p class="first admonition-title" style="font-weight: bold;">Tip</p>
# <p class="last">We catch the exception with a <tt class="docutils literal">try</tt>/<tt class="docutils literal">except</tt> pattern to be able to print it.</p>
# </div>
# We get an exception because the default scorer has its positive label set to
# one (`pos_label=1`), which is not our case (our positive label is "donated").
# In this case, we need to create a scorer using the scoring function and the
# helper function `make_scorer`.
# 
# So, import `sklearn.metrics.make_scorer` and
# `sklearn.metrics.precision_score`. Check their documentations for more
# information.
# Finally, create a scorer by calling `make_scorer` using the score function
# `precision_score` and pass the extra parameter `pos_label="donated"`.

# In[7]:


# Write your code here.
from sklearn.metrics import make_scorer, precision_score

scorer = make_scorer(precision_score, pos_label='donated')


# Now, instead of providing the string `"precision"` to the `scoring` parameter
# in the `cross_val_score` call, pass the scorer that you created above.

# In[9]:


# Write your code here.
tree = DecisionTreeClassifier()
try:
    scores = cross_val_score(tree, data, target, cv=10, scoring=scorer)
except ValueError as exc:
    print(exc)
print(scores)


# `cross_val_score` will only compute a single score provided to the `scoring`
# parameter. The function `cross_validate` allows the computation of multiple
# scores by passing a list of string or scorer to the parameter `scoring`,
# which could be handy.
# 
# Import `sklearn.model_selection.cross_validate` and compute the accuracy and
# balanced accuracy through cross-validation. Plot the cross-validation score
# for both metrics using a box plot.

# In[10]:


# Write your code here.
from sklearn.model_selection import cross_validate
scoring = ["accuracy", "balanced_accuracy"]

scores = cross_validate(tree, data, target, cv=cv, scoring=scoring)
scores

import pandas as pd

color = {"whiskers": "black", "medians": "black", "caps": "black"}

metrics = pd.DataFrame(
    [scores["test_accuracy"], scores["test_balanced_accuracy"]],
    index=["Accuracy", "Balanced accuracy"]
).T

import matplotlib.pyplot as plt

metrics.plot.box(vert=False, color=color)
_ = plt.title("Computation of multiple scores using cross_validate")


# In[ ]:




