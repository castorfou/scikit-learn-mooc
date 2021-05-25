#!/usr/bin/env python
# coding: utf-8

# # Encoding of categorical variables
# 
# In this notebook, we will present typical ways of dealing with
# **categorical variables** by encoding them, namely **ordinal encoding** and
# **one-hot encoding**.

# Let's first load the entire adult dataset containing both numerical and
# categorical data.

# In[ ]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]

data = adult_census.drop(columns=[target_name])


# 
# ## Identify categorical variables
# 
# As we saw in the previous section, a numerical variable is a
# quantity represented by a real or integer number. These variables can be
# naturally handled by machine learning algorithms that are typically composed
# of a sequence of arithmetic instructions such as additions and
# multiplications.
# 
# In contrast, categorical variables have discrete values, typically
# represented by string labels (but not only) taken from a finite list of
# possible choices. For instance, the variable `native-country` in our dataset
# is a categorical variable because it encodes the data using a finite list of
# possible countries (along with the `?` symbol when this information is
# missing):

# In[ ]:


data["native-country"].value_counts().sort_index()


# How can we easily recognize categorical columns among the dataset? Part of
# the answer lies in the columns' data type:

# In[ ]:


data.dtypes


# If we look at the `"native-country"` column, we observe its data type is
# `object`, meaning it contains string values.
# 
# ## Select features based on their data type
# 
# In the previous notebook, we manually defined the numerical columns. We could
# do a similar approach. Instead, we will use the scikit-learn helper function
# `make_column_selector`, which allows us to select columns based on
# their data type. We will illustrate how to use this helper.

# In[ ]:


from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_columns


# Here, we created the selector by passing the data type to include; we then
# passed the input dataset to the selector object, which returned a list of
# column names that have the requested data type. We can now filter out the
# unwanted columns:

# In[ ]:


data_categorical = data[categorical_columns]
data_categorical.head()


# In[ ]:


print(f"The dataset is composed of {data_categorical.shape[1]} features")


# In the remainder of this section, we will present different strategies to
# encode categorical data into numerical data which can be used by a
# machine-learning algorithm.

# ## Strategies to encode categories
# 
# ### Encoding ordinal categories
# 
# The most intuitive strategy is to encode each category with a different
# number. The `OrdinalEncoder` will transform the data in such manner.
# We will start by encoding a single column to understand how the encoding
# works.

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

education_column = data_categorical[["education"]]

encoder = OrdinalEncoder()
education_encoded = encoder.fit_transform(education_column)
education_encoded


# We see that each category in `"education"` has been replaced by a numeric
# value. We could check the mapping between the categories and the numerical
# values by checking the fitted attribute `categories_`.

# In[ ]:


encoder.categories_


# Now, we can check the encoding applied on all categorical features.

# In[ ]:


data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]


# In[ ]:


encoder.categories_


# In[ ]:


print(
    f"The dataset encoded contains {data_encoded.shape[1]} features")


# We see that the categories have been encoded for each feature (column)
# independently. We also note that the number of features before and after the
# encoding is the same.
# 
# However, be careful when applying this encoding strategy:
# using this integer representation leads downstream predictive models
# to assume that the values are ordered (0 < 1 < 2 < 3... for instance).
# 
# By default, `OrdinalEncoder` uses a lexicographical strategy to map string
# category labels to integers. This strategy is arbitrary and often
# meaningless. For instance, suppose the dataset has a categorical variable
# named `"size"` with categories such as "S", "M", "L", "XL". We would like the
# integer representation to respect the meaning of the sizes by mapping them to
# increasing integers such as `0, 1, 2, 3`.
# However, the lexicographical strategy used by default would map the labels
# "S", "M", "L", "XL" to 2, 1, 0, 3, by following the alphabetical order.
# 
# The `OrdinalEncoder` class accepts a `categories` constructor argument to
# pass categories in the expected ordering explicitly. You can find more
# information in the
# [scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
# if needed.
# 
# If a categorical variable does not carry any meaningful order information
# then this encoding might be misleading to downstream statistical models and
# you might consider using one-hot encoding instead (see below).
# 
# ### Encoding nominal categories (without assuming any order)
# 
# `OneHotEncoder` is an alternative encoder that prevents the downstream
# models to make a false assumption about the ordering of categories. For a
# given feature, it will create as many new columns as there are possible
# categories. For a given sample, the value of the column corresponding to the
# category will be set to `1` while all the columns of the other categories
# will be set to `0`.
# 
# We will start by encoding a single feature (e.g. `"education"`) to illustrate
# how the encoding works.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
education_encoded = encoder.fit_transform(education_column)
education_encoded


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p><tt class="docutils literal">sparse=False</tt> is used in the <tt class="docutils literal">OneHotEncoder</tt> for didactic purposes, namely
# easier visualization of the data.</p>
# <p class="last">Sparse matrices are efficient data structures when most of your matrix
# elements are zero. They won't be covered in details in this course. If you
# want more details about them, you can look at
# <a class="reference external" href="https://scipy-lectures.org/advanced/scipy_sparse/introduction.html#why-sparse-matrices">this</a>.</p>
# </div>

# We see that encoding a single feature will give a NumPy array full of zeros
# and ones. We can get a better understanding using the associated feature
# names resulting from the transformation.

# In[ ]:


feature_names = encoder.get_feature_names(input_features=["education"])
education_encoded = pd.DataFrame(education_encoded, columns=feature_names)
education_encoded


# As we can see, each category (unique value) became a column; the encoding
# returned, for each sample, a 1 to specify which category it belongs to.
# 
# Let's apply this encoding on the full dataset.

# In[ ]:


print(
    f"The dataset is composed of {data_categorical.shape[1]} features")
data_categorical.head()


# In[ ]:


data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]


# In[ ]:


print(
    f"The encoded dataset contains {data_encoded.shape[1]} features")


# Let's wrap this NumPy array in a dataframe with informative column names as
# provided by the encoder object:

# In[ ]:


columns_encoded = encoder.get_feature_names(data_categorical.columns)
pd.DataFrame(data_encoded, columns=columns_encoded).head()


# Look at how the `"workclass"` variable of the 3 first records has been
# encoded and compare this to the original string representation.
# 
# The number of features after the encoding is more than 10 times larger than
# in the original data because some variables such as `occupation` and
# `native-country` have many possible categories.

# ### Choosing an encoding strategy
# 
# Choosing an encoding strategy will depend on the underlying models and the
# type of categories (i.e. ordinal vs. nominal).
# 
# Indeed, using an `OrdinaleEncoder` will output ordinal categories. It means
# that there is an order in the resulting categories (e.g. `0 < 1 < 2`). The
# impact of violating this ordering assumption is really dependent on the
# downstream models. Linear models will be impacted by misordered categories
# while tree-based models will not be.
# 
# Thus, in general `OneHotEncoder` is the encoding strategy used when the
# downstream models are **linear models** while `OrdinalEncoder` is used with
# **tree-based models**.
# 
# You still can use an `OrdinalEncoder` with linear models but you need to be
# sure that:
# - the original categories (before encoding) have an ordering;
# - the encoded categories follow the same ordering than the original
#   categories.
# The next exercise highlight the issue of misusing `OrdinalEncoder` with a
# linear model.
# 
# Also, there is no need to use an `OneHotEncoder` even if the original
# categories do not have an given order with tree-based model. It will be
# the purpose of the final exercise of this sequence.

# ## Evaluate our predictive pipeline
# 
# We can now integrate this encoder inside a machine learning pipeline like we
# did with numerical data: let's train a linear classifier on the encoded data
# and check the statistical performance of this machine learning pipeline using
# cross-validation.
# 
# Before we create the pipeline, we have to linger on the `native-country`.
# Let's recall some statistics regarding this column.

# In[ ]:


data["native-country"].value_counts()


# We see that the `Holand-Netherlands` category is occurring rarely. This will
# be a problem during cross-validation: if the sample ends up in the test set
# during splitting then the classifier would not have seen the category during
# training and will not be able to encode it.
# 
# In scikit-learn, there are two solutions to bypass this issue:
# 
# * list all the possible categories and provide it to the encoder via the
#   keyword argument `categories`;
# * use the parameter `handle_unknown`.
# 
# Here, we will use the latter solution for simplicity.

# <div class="admonition tip alert alert-warning">
# <p class="first admonition-title" style="font-weight: bold;">Tip</p>
# <p class="last">Be aware the <tt class="docutils literal">OrdinalEncoder</tt> exposes as well a parameter
# <tt class="docutils literal">handle_unknown</tt>. It can be set to <tt class="docutils literal">use_encoded_value</tt> and by setting
# <tt class="docutils literal">unknown_value</tt> to handle rare categories. You are going to use these
# parameters in the next exercise.</p>
# </div>

# We can now create our machine learning pipeline.

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
)


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">Here, we need to increase the maximum number of iterations to obtain a fully
# converged <tt class="docutils literal">LogisticRegression</tt> and silence a <tt class="docutils literal">ConvergenceWarning</tt>. Contrary
# to the numerical features, the one-hot encoded categorical features are all
# on the same scale (values are 0 or 1), so they would not benefit from
# scaling. In this case, increasing <tt class="docutils literal">max_iter</tt> is the right thing to do.</p>
# </div>

# Finally, we can check the model's statistical performance only using the
# categorical columns.

# In[ ]:


from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, data_categorical, target)
cv_results


# In[ ]:


scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} +/- {scores.std():.3f}")


# As you can see, this representation of the categorical variables is
# slightly more predictive of the revenue than the numerical variables
# that we used previously.

# 
# In this notebook we have:
# * seen two common strategies for encoding categorical features: **ordinal
#   encoding** and **one-hot encoding**;
# * used a **pipeline** to use a **one-hot encoder** before fitting a logistic
#   regression.
