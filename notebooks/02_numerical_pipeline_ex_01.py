#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.03
# 
# The goal of this exercise is to compare the statistical performance of our
# classifier (81% accuracy) to some baseline classifiers that would ignore the
# input data and instead make constant predictions.
# 
# - What would be the score of a model that always predicts `' >50K'`?
# - What would be the score of a model that always predicts `' <=50K'`?
# - Is 81% or 82% accuracy a good score for this problem?
# 
# Use a `DummyClassifier` and do a train-test split to evaluate
# its accuracy on the test set. This
# [link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
# shows a few examples of how to evaluate the statistical performance of these
# baseline models.

# In[1]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")


# We will first split our dataset to have the target separated from the data
# used to train our predictive model.

# In[2]:


target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)


# We start by selecting only the numerical columns as seen in the previous
# notebook.

# In[3]:


numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]


# Split the dataset into a train and test sets.

# In[4]:


from sklearn.model_selection import train_test_split

# Write your code here.
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25)


# Use a `DummyClassifier` such that the resulting classifier will always
# predict the class `' >50K'`. What is the accuracy score on the test set?
# Repeat the experiment by always predicting the class `' <=50K'`.
# 
# Hint: you can refer to the parameter `strategy` of the `DummyClassifier`
# to achieve the desired behaviour.

# In[9]:


from sklearn.dummy import DummyClassifier

# Write your code here.

dc_low = DummyClassifier(strategy='constant', constant=' <=50K')
dc_low.fit(data_train, target_train)
dc_low.score(data_test, target_test)


# In[10]:


dc_high = DummyClassifier(strategy='constant', constant=' >50K')
dc_high.fit(data_train, target_train)
dc_high.score(data_test, target_test)


# In[ ]:




