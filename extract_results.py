#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install statsmodels
# Imports
from osprey.config import Config
import pandas as pd

# # Introduction

# This workbook loads the optimization results and saves them as a dataframe. 

# # Load the data

# In[4]:


root_dir = 'fs-peptide'
# Load Configuation Files
databases = {'bayesian':root_dir+'/gp-m52-ei-tica-indv/config-all_tor.yaml', 
             'random':root_dir+'/rand-tica-indv/config_random-all_tor.yaml', 
             'sobol':root_dir+'/sobol-tica-indv/config-all_tor.yaml', 
             'tpe':root_dir+'/tpe-s20-g25-tica-indv/config-all_tor.yaml'}

all_dfs = []
for k, v in databases.items():
    config = Config(v)
    df = config.trial_results()
    df['method'] = k
    all_dfs.append(df)
    
df_all = pd.concat(all_dfs)


# In[7]:


df_all.head()


# # Drop unnecessary columns and rename

# In[67]:


df = df_all.loc[:, ['parameters', 'project_name','mean_test_score', 'mean_train_score', 'test_scores', 'train_scores', 'method', 'completed']]
df.rename(columns={'project_name':'basis'}, inplace=True)


# # Extract parameter values 

# In[5]:


params = ['cluster__n_clusters', 'tica__n_components', 'tica__lag_time']
for param in params:
    df[param.split('__')[1]] = df['parameters'].apply(lambda x: x[param])


# # Save data

# In[6]:


df.to_csv('./data/all_methods_results.csv')

