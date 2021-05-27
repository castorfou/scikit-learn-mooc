#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M2.01
# 
# The aim of this exercise is to make the following experiments:
# 
# * train and test a support vector machine classifier through
#   cross-validation;
# * study the effect of the parameter gamma of this classifier using a
#   validation curve;
# * study if it would be useful in term of classification if we could add new
#   samples in the dataset using a learning curve.
# 
# To make these experiments we will first load the blood transfusion dataset.

# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# In[14]:


import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]


# We will use a support vector machine classifier (SVM). In its most simple
# form, a SVM classifier is a linear classifier behaving similarly to a
# logistic regression. Indeed, the optimization used to find the optimal
# weights of the linear model are different but we don't need to know these
# details for the exercise.
# 
# Also, this classifier can become more flexible/expressive by using a
# so-called kernel making the model becomes non-linear. Again, no requirement
# regarding the mathematics is required to accomplish this exercise.
# 
# We will use an RBF kernel where a parameter `gamma` allows to tune the
# flexibility of the model.
# 
# First let's create a predictive pipeline made of:
# 
# * a [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
#   with default parameter;
# * a [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
#   where the parameter `kernel` could be set to `"rbf"`. Note that this is the
#   default.

# In[15]:


# to display nice model diagram
from sklearn import set_config
set_config(display='diagram')


# In[16]:


# Write your code here.

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
model


# Evaluate the statistical performance of your model by cross-validation with a
# `ShuffleSplit` scheme. Thus, you can use
# [`sklearn.model_selection.cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
# and pass a [`sklearn.model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)
# to the `cv` parameter. Only fix the `random_state=0` in the `ShuffleSplit`
# and let the other parameters to the default.

# In[17]:


# Write your code here.

import pandas as pd
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(random_state=0)
cv_results = cross_validate(model, data, target,
                            cv=cv)
cv_results = pd.DataFrame(cv_results)
cv_results


# As previously mentioned, the parameter `gamma` is one of the parameter
# controlling under/over-fitting in support vector machine with an RBF kernel.
# 
# Compute the validation curve
# (using [`sklearn.model_selection.validation_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html))
# to evaluate the effect of the parameter `gamma`. You can vary its value
# between `10e-3` and `10e2` by generating samples on a logarithmic scale.
# Thus, you can use `np.logspace(-3, 2, num=30)`.
# 
# Since we are manipulating a `Pipeline` the parameter name will be set to
# `svc__gamma` instead of only `gamma`. You can retrieve the parameter name
# using `model.get_params().keys()`. We will go more into details regarding
# accessing and setting hyperparameter in the next section.

# In[18]:


# Write your code here.

from sklearn.model_selection import validation_curve
import numpy as np

gamma = np.logspace(-3, 2, num=30)
train_scores, test_scores = validation_curve(
    model, data, target, param_name="svc__gamma", param_range=gamma,
    cv=cv)
train_errors, test_errors = -train_scores, -test_scores


# Plot the validation curve for the train and test scores.

# In[22]:


# Write your code here.

import matplotlib.pyplot as plt

plt.errorbar(gamma, train_scores.mean(axis=1),yerr=train_scores.std(axis=1), label="Training error")
plt.errorbar(gamma, test_scores.mean(axis=1),yerr=test_scores.std(axis=1), label="Testing error")
plt.legend()

plt.xscale("log")

plt.xlabel("Gamma value for SVC")
plt.ylabel("Mean absolute error")
_ = plt.title("Validation curve for SVC")


# Now, you can perform an analysis to check whether adding new samples to the
# dataset could help our model to better generalize. Compute the learning curve
# (using [`sklearn.model_selection.learning_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html))
# by computing the train and test scores for different training dataset size.
# Plot the train and test scores with respect to the number of samples.

# In[12]:


# Write your code here.

from sklearn.model_selection import learning_curve
import numpy as np
train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)
train_sizes

from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)

results = learning_curve(
    model, data, target, train_sizes=train_sizes, cv=cv)
train_size, train_scores, test_scores = results[:3]
# Convert the scores into errors
train_errors, test_errors = -train_scores, -test_scores

import matplotlib.pyplot as plt

plt.errorbar(train_size, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training error")
plt.errorbar(train_size, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Testing error")
plt.legend()

plt.xscale("log")
plt.xlabel("Number of samples in the training set")
plt.ylabel("Mean absolute error")
_ = plt.title("Learning curve for SVC")


# In[ ]:




