#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M7.01
# 
# This notebook aims at building baseline classifiers, which we'll use to
# compare our predictive model. Besides, we will check the differences with
# the baselines that we saw in regression.
# 
# We will use the adult census dataset, using only the numerical features.

# In[1]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census-numeric-all.csv")
data, target = adult_census.drop(columns="class"), adult_census["class"]


# First, define a `ShuffleSplit` cross-validation strategy taking half of the
# sample as a testing at each round.

# In[2]:


# Write your code here.
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.5, random_state=0)


# Next, create a machine learning pipeline composed of a transformer to
# standardize the data followed by a logistic regression.

# In[3]:


# Write your code here.
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), LogisticRegression())


# Get the test score by using the model, the data, and the cross-validation
# strategy that you defined above.

# In[10]:


# Write your code here.
from sklearn.model_selection import cross_validate
import pandas as pd

result_classifier = cross_validate(pipeline, data, target, cv=cv)
errors_classifier = pd.Series(result_classifier['test_score'], name='classifier error')
print(f'Test score {errors_classifier.mean():0.02f}')


# Using the `sklearn.model_selection.permutation_test_score` function,
# check the chance level of the previous model.

# In[11]:


# Write your code here.
from sklearn.model_selection import permutation_test_score

score, permutation_score, pvalue = permutation_test_score(
    pipeline, data, target, cv=cv, 
    n_jobs=-1, n_permutations=30)
errors_permutation = pd.Series(permutation_score, name="Permuted error")

print(f'Permutation score {errors_permutation.mean():0.02f}')


# Finally, compute the test score of a dummy classifier which would predict
# the most frequent class from the training set. You can look at the
# `sklearn.dummy.DummyClassifier` class.

# In[12]:


# Write your code here.
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()
result_dummy = cross_validate(dummy, data, target,
                              cv=cv, n_jobs=-1)
errors_dummy = pd.Series(result_dummy["test_score"], name="Dummy error")
print(f'Dummy score {errors_dummy.mean():0.02f}')


# Now that we collected the results from the baselines and the model, plot
# the distributions of the different test scores.

# We concatenate the different test score in the same pandas dataframe.

# In[13]:


# Write your code here.
final_errors = pd.concat([errors_classifier, errors_dummy, errors_permutation],
                         axis=1)


# Next, plot the distributions of the test scores.

# In[14]:


# Write your code here.
import matplotlib.pyplot as plt

final_errors.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Accuracy")
_ = plt.title("Distribution of the testing errors")


# Change the strategy of the dummy classifier to `stratified`, compute the
# results and plot the distribution together with the other results. Explain
# why the results get worse.

# In[15]:


# Write your code here.
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='stratified')
result_dummy = cross_validate(dummy, data, target,
                              cv=cv, n_jobs=-1)
errors_dummy = pd.Series(result_dummy["test_score"], name="Dummy error")
print(f'Dummy score {errors_dummy.mean():0.02f}')


# In[ ]:




