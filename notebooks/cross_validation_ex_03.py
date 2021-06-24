#!/usr/bin/env python
# coding: utf-8

# # 📝 Introductory exercise regarding stratification
# 
# The goal of this exercise is to highlight one limitation of
# applying blindly a k-fold cross-validation.
# 
# In this exercise we will use the iris dataset.

# In[1]:


from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True, as_frame=True)


# Create a decision tree classifier that we will use in the next experiments.

# In[2]:


# Write your code here.
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()


# As a first experiment, use the utility
# `sklearn.model_selection.train_test_split` to split the data into a train
# and test set. Train the classifier using the train set and check the score
# on the test set.

# In[3]:


# Write your code here.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

dt.fit(data_train, target_train)
prediction = dt.predict(data_test)
print(f'accuracy : {accuracy_score(prediction, target_test):0.02f}')


# Now, use the utility `sklearn.utils.cross_val_score` with a
# `sklearn.model_selection.KFold` by setting only `n_splits=3`. Check the
# results on each fold. Explain the results.

# In[5]:


# Write your code here.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

cv = KFold(n_splits=3)

score = cross_val_score(dt, data, target, cv=cv)

print(score)


# In[ ]:




