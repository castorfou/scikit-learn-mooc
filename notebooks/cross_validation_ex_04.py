#!/usr/bin/env python
# coding: utf-8

# # 📝 Introductory exercise for sample grouping
# 
# This exercise aims at highlighting issues that one could encounter when
# discarding grouping pattern existing in a dataset.
# 
# We will use the digits dataset which includes some grouping pattern.

# In[1]:


from sklearn.datasets import load_digits

data, target = load_digits(return_X_y=True, as_frame=True)


# The first step is to create a model. Use a machine learning pipeline
# composed of a scaler followed by a logistic regression classifier.

# In[2]:


# Write your code here.
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), LogisticRegression())


# Then, create a a `KFold` object making sure that the data will not be
# shuffled during the cross-validation. Use the previous model, data, and
# cross-validation strategy defined to estimate the statistical performance of
# the model.

# In[3]:


# Write your code here.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

cv = KFold(n_splits=3)

score = cross_val_score(pipeline, data, target, cv=cv)

print(score)


# Finally, perform the same experiment by shuffling the data within the
# cross-validation. Draw some conclusion regarding the dataset.

# In[4]:


# Write your code here.
cv = KFold(n_splits=3, shuffle=True)

score = cross_val_score(pipeline, data, target, cv=cv)

print(score)


# In[ ]:




