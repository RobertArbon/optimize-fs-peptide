#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This reshapes the results file so that it can be used in a Gaussian Process model. 

# In[33]:


import pandas as pd
import numpy as np
import re


# In[63]:


df_all = pd.read_csv('data/all_methods_results.csv')


# The lists have been converted to strings. Make them lists of floats first.

# In[64]:


def to_float(x):
    nums = re.findall('[0-9]+\\.[0-9]+', string=x)
    nums = [float(y) for y in nums]
    return nums


df_all.loc[:, 'test_scores'] = df_all.loc[:, 'test_scores'].apply(func=to_float)
df_all.loc[:, 'train_scores'] = df_all.loc[:, 'train_scores'].apply(func=to_float)


# Extract only relevant columns

# In[71]:


df = df_all.loc[:, ['method',
                    'test_scores', 
                    'train_scores',
                    'basis','lag_time', 'n_components', 'n_clusters']]


# In[72]:


df.to_csv('results/results_clean.csv', index=False)

