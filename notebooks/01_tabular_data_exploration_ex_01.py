#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.01

# Imagine we are interested in predicting penguins species based on two of
# their body measurements: culmen length and culmen depth. First we want to do
# some data exploration to get a feel for the data.
# 
# What are the features? What is the target?

# The data is located in `../datasets/penguins_classification.csv`, load it
# with `pandas` into a `DataFrame`.

# In[1]:


# Write your code here.
import pandas as pd

penguins = pd.read_csv('../datasets/penguins_classification.csv')


# Show a few samples of the data
# 
# How many features are numerical? How many features are categorical?

# In[5]:


# Write your code here.
penguins.head()


# In[7]:


penguins.describe()


# What are the different penguins species available in the dataset and how many
# samples of each species are there? Hint: select the right column and use
# the [`value_counts`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) method.

# In[6]:


penguins.Species.value_counts()


# In[ ]:


# Write your code here.


# Plot histograms for the numerical features

# In[9]:


# Write your code here.
_ = penguins.hist(figsize=(20, 14))


# Show features distribution for each class. Hint: use
# [`seaborn.pairplot`](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

# In[10]:


# Write your code here.
import seaborn as sns

columns = ['Culmen Length (mm)', 'Culmen Depth (mm)']
_ = sns.pairplot(data=penguins, vars=columns,
                 hue='Species', plot_kws={'alpha': 0.2},
                 height=3, diag_kind='hist', diag_kws={'bins': 30})


# Looking at these distributions, how hard do you think it will be to classify
# the penguins only using "culmen depth" and "culmen length"?
