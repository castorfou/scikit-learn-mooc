#!/usr/bin/env python
# coding: utf-8

# # 🏁 Wrap-up quiz
# 
# **This quiz requires some programming to be answered.**
# 
# Open the dataset `blood_transfusion.csv`.
# 
# ```py
# import pandas as pd
# 
# blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
# data = blood_transfusion.drop(columns="Class")
# target = blood_transfusion["Class"]
# ```
# 
# In this dataset, the column `"Class"` is the target vector containing the
# labels that our model should predict.
# 
# For all the questions below, make a cross-validation evaluation using a
# 10-fold cross-validation strategy.
# 
# Evaluate the performance of a `sklearn.dummy.DummyClassifier` that always
# predict the most frequent class seen during the training. Be aware that you can
# pass a list of score to compute in `sklearn.model_selection.cross_validate` by
# setting the parameter `scoring`.
# 
# ```{admonition} Question
# What the accuracy of this dummy classifier?
# 
# - a) ~0.5
# - b) ~0.62
# - c) ~0.75
# 
# _Select a single answer_
# ```
# 
# +++
# 
# ```{admonition} Question
# What the balanced accuracy of this dummy classifier?
# 
# - a) ~0.5
# - b) ~0.62
# - c) ~0.75
# 
# _Select a single answer_
# ```
# 
# +++
# 
# Replace the `DummyClassifier` by a `sklearn.tree.DecisionTreeClassifier` and
# check the statistical performance to answer the question below.
# 
# ```{admonition} Question
# Is a single decision classifier better than a dummy classifier (at least an
# increase of 4%) in terms of balanced accuracy?
# 
# - a) Yes
# - b) No
# 
# _Select a single answer_
# ```
# 
# +++
# 
# Evaluate the performance of a `sklearn.ensemble.RandomForestClassifier` using
# 300 trees.
# 
# ```{admonition} Question
# Is random forest better than a dummy classifier (at least an increase of 4%)
# in terms of balanced accuracy?
# 
# - a) Yes
# - b) No
# 
# _Select a single answer_
# ```
# 
# +++
# 
# Compare a `sklearn.ensemble.GradientBoostingClassifier` and a
# `sklearn.ensemble.RandomForestClassifier` with both 300 trees. Evaluate both
# models with a 10-fold cross-validation and repeat the experiment 10 times.
# 
# ```{admonition} Question
# On average, is the gradient boosting better than the random forest?
# 
# - a) Yes
# - b) No
# - c) Equivalent
# 
# _Select a single answer_
# ```
# 
# +++
# 
# Evaluate the performance of a
# `sklearn.ensemble.HistGradientBoostingClassifier`. Enable early-stopping and
# add as many trees as needed.
# 
# **Note**: Be aware that you need a specific import when importing the
# `HistGradientBoostingClassifier`:
# 
# ```py
# # explicitly require this experimental feature
# from sklearn.experimental import enable_hist_gradient_boosting
# # now you can import normally from ensemble
# from sklearn.ensemble import HistGradientBoostingClassifier
# ```
# 
# ```{admonition} Question
# Is histogram gradient boosting a better classifier?
# 
# - a) Histogram gradient boosting is the best estimator
# - b) Histogram gradient boosting is better than random forest by worse than
#   the exact gradient boosting
# - c) Histogram gradient boosting is better than the exact gradient boosting but
#   worse than the random forest
# - d) Histogram gradient boosting is the worst estimator
# 
# _Select a single answer_
# ```
# 
# +++
# 
# ```{admonition} Question
# With the early stopping activated, how many trees on average the
# `HistGradientBoostingClassifier` needed to converge?
# 
# - a) ~30
# - b) ~100
# - c) ~150
# - d) ~200
# - e) ~300
# 
# _Select a single answer_
# ```
# 
# +++
# 
# [Imbalanced-learn](https://imbalanced-learn.org/stable/) is an open-source
# library relying on scikit-learn and provides methods to deal with
# classification with imbalanced classes.
# 
# Here, we will be using the class
# [`imblearn.ensemble.BalancedBaggingClassifier`](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html)
# to alleviate the issue of class imbalance.
# 
# Use the `BalancedBaggingClassifier` and pass an
# `HistGradientBoostingClassifier` as a `base_estimator`. Fix the hyperparameter
# `n_estimators` to 50.
# 
# **Note**: In case that imbalanced-learn is not available on your laptop, you
# can install it via PyPI or conda-forge channel. Thus in a notebook, you can
# install using:
# 
# ```
# %pip install -U imbalanced-learn
# ```
# 
# or
# 
# ```
# !conda install imbalannced-learn -c conda-forge
# ```
# 
# ```{admonition} Question
# What is a [`BalancedBaggingClassifier`](https://imbalanced-learn.org/stable/ensemble.html#bagging)?
# 
# - a) Is a classifier that make sure that each tree leaves belong to the same
#   depth level
# - b) Is a classifier that explicitly maximizes the balanced accuracy score
# - c) Equivalent to a `sklearn.ensemble.BaggingClassifier` with a resampling of
#      each bootstrap sample to contain a many samples from each class.
# 
# _Select a single answer_
# ```
# 
# +++
# 
# ```{admonition} Question
# Is the balanced accuracy of the `BalancedBaggingClassifier` is
# _choose an answer_ than an `HistGradientBoostingClassifier` alone?
# 
# - a) Worse
# - b) Better
# - c) Equivalent
# 
# _Select a single answer_
# ```
