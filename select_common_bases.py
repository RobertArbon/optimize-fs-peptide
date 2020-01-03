#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# # Load data

# In[5]:


df = pd.read_csv('results/results_clean.csv')


# In[6]:


df.head()


# # Subset 

# Not all bases ran for each method. Reason was probably that BC failed and I had to run some of them on Blaze at the last minute. 

# In[7]:


bases = [(k, set(v['basis'].unique())) for k, v in df.groupby('method')]

common_bases = set(bases[0][1])
for k, v in bases:
    common_bases = common_bases & v
common_bases = list(common_bases)
common_bases


# Keep only these bases

# In[11]:


df = df.loc[df['basis'].isin(common_bases), :]

df.to_csv('results/results_clean_common.csv', index=False)


# In[ ]:




