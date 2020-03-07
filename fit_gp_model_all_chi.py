#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This workbook fits a Gaussian Process model to the Random and Sobol data. 

# In[1]:


import GPy
import pandas as pd
import patsy as pt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import re
import pymc3 as pm
import matplotlib.ticker as tk
import re
from sklearn.model_selection import KFold, StratifiedKFold
import pickle


# In[2]:


df = pd.read_csv('results/all_results_average.csv')
df = df.rename(columns={'lag_time': 't', 'n_components': 'm', 'n_clusters': 'n', 'test_score': 'y'}).loc[:, ['basis', 'method', 'y', 't', 'm', 'n']]

to_scale = ['m', 'n', 't']
scaler = preprocessing.MinMaxScaler()
vars_scaled = pd.DataFrame(scaler.fit_transform(df.loc[:, to_scale]), columns=[x+'_s' for x in to_scale])
df = df.join(vars_scaled)

idx = (df['method'] == 'random') & (~df['basis'].isin(['ff_all_ang', 'ff_bb_ang', 'ff_re_ang', 'chi_tor']))
X = df.loc[idx, :].filter(regex='(_s$|basis)')
y = df.loc[idx, 'y']

X_c = pt.dmatrix('0+ t_s + m_s + n_s+C(basis)', data=X, return_type='dataframe')
X_c = X_c.rename(columns=lambda x: re.sub('C|\\(|\\)|\\[|\\]','',x))


## Priors
def gamma(alpha, beta):
    def g(x):
        return pm.Gamma(x, alpha=alpha, beta=beta)
    return g

def hcauchy(beta):
    def g(x):
        return pm.HalfCauchy(x, beta=beta)
    return g

def fit_model_1(y, X, kernel_type='rbf'):
    """
    function to return a pymc3 model
    y : dependent variable
    X : independent variables
    prop_Xu : number of inducing varibles to use
    
    X, y are dataframes. We'll use the column names. 
    """
    with pm.Model() as model:
        # Covert arrays
        X_a = X.values
        y_a = y.values
        X_cols = list(X.columns)
        
        # Globals
        prop_Xu = 0.1
        l_prior = gamma(1, 0.05)
        eta_prior = hcauchy(2)
        sigma_prior = hcauchy(2)

        # Kernels
        # 3 way interaction
        eta = eta_prior('eta')
        cov = eta**2
        for i in range(X_a.shape[1]):
            var_lab = 'l_'+X_cols[i]
            if kernel_type=='RBF':
                cov = cov*pm.gp.cov.ExpQuad(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type=='Exponential':
                cov = cov*pm.gp.cov.Exponential(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type=='M52':
                cov = cov*pm.gp.cov.Matern52(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type=='M32':
                cov = cov*pm.gp.cov.Matern32(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])

        # Covariance model
        cov_tot = cov 

        # Model
        gp = pm.gp.MarginalSparse(cov_func=cov_tot, approx="FITC")

        # Noise model
        sigma_n =sigma_prior('sigma_n')

        # Inducing variables
        num_Xu = int(X_a.shape[0]*prop_Xu)
        Xu = pm.gp.util.kmeans_inducing_points(num_Xu, X_a)

        # Marginal likelihood
        y_ = gp.marginal_likelihood('y_', X=X_a, y=y_a,Xu=Xu, noise=sigma_n)
        mp = pm.find_MAP()
        
    return gp, mp, model


# Inputs
exp_labs = ['Categorical']
predictors = [X_c]
targets = [y]
kernels =  ['M32' ]

# Outputs
pred_dfs = []

# iterator
kf = StratifiedKFold(n_splits=10)

for i in range(len(exp_labs)):
    
    for idx, (train_idx, test_idx) in enumerate(kf.split(X.values, X['basis'])):
        # subset dataframes
        y_train = targets[i].iloc[train_idx]
        X_train = predictors[i].iloc[train_idx, :]
        y_test = targets[i].iloc[test_idx]
        X_test = predictors[i].iloc[test_idx, :]  
        
        gp, mp, model = fit_model_1(y=y_train, X=X_train, kernel_type=kernels[i])

        # Test
        with model:
            # predict latent
            mu, var = gp.predict(X_test.values, point=mp, diag=True,pred_noise=False)
            sd_f = np.sqrt(var)
            # predict target
            _, var = gp.predict(X_test.values, point=mp, diag=True,pred_noise=True)
            sd_y = np.sqrt(var)
            
        res = pd.DataFrame({'f_pred': mu, 'sd_f': sd_f, 'sd_y': sd_y, 'y': y_test.values})
        res.loc[:, 'kernel'] = kernels[i]
        res.loc[:, 'm_type'] = exp_labs[i]
        res.loc[:, 'fold_num'] = idx
        
        pred_dfs.append(pd.concat([X_test.reset_index(), res.reset_index()], axis=1))

pred_dfs = pd.concat(pred_dfs)

null_mu = np.mean(y)
null_sd = np.std(y)


def ll(f_pred, sigma_pred, y_true):
    tmp = 0.5*np.log(2*np.pi*sigma_pred**2)
    tmp += (f_pred-y_true)**2/(2*sigma_pred**2)
    return tmp


sll = ll(pred_dfs['f_pred'], pred_dfs['sd_y'], pred_dfs['y'])
sll = sll - ll(null_mu, null_sd, pred_dfs['y'])
pred_dfs['msll'] = sll
pred_dfs['smse'] = (pred_dfs['f_pred']-pred_dfs['y'])**2/np.var(y)
pred_dfs.to_pickle('results/all_chi_gp_cross_validation.p')

msll = pred_dfs.groupby(['m_type', 'kernel'])['msll'].mean()
smse = pred_dfs.groupby(['m_type', 'kernel'])['smse'].mean()

summary = pd.DataFrame(smse).join(other=pd.DataFrame(msll), on=['m_type', 'kernel'], how='left')
summary.to_csv('results/all_chi_gp_fit_summary.csv')


# ## Model fit
gp, mp, model = fit_model_1(y=y, X=X_c, kernel_type='M32')
pickle.dump(
    {'gp': gp, 'mp': mp, 'model': model},
    open('results/all_chi_mml_model.p', 'wb')
)
# Create new matrix
chi_new, m_new, n_new, t_new = np.meshgrid(np.unique(X['basis'].values), np.linspace(0, 1, 10), np.linspace(0, 1, 20), np.linspace(0, 1, 20))
X_new = np.array([chi_new.flatten(), m_new.flatten(), n_new.flatten(), t_new.flatten()]).T

# One hot encoding
X_new = pd.DataFrame(X_new, columns=['basis', 'm_s', 'n_s', 't_s'])
for x in ['m_s', 'n_s', 't_s']:
    X_new.loc[:, x] = X_new.loc[:, x].astype(float) 
X_new_c = pt.dmatrix('0 + t_s + m_s + n_s+ C(basis)', data=X_new, return_type='dataframe')
X_new_c = X_new_c.rename(columns=lambda x: re.sub('C|\\(|\\)|\\[|\\]','',x))

# Make predictions
with model:
    # predict latent
    mu, var = gp.predict(X_new_c.values, point=mp, diag=True,pred_noise=True)
    _, var2 = gp.predict(X_new_c.values, point=mp, diag=True,pred_noise=False)
# put predictions in dataframe
pred_df = X_new_c.copy(deep=True)
pred_df['y'] = mu
pred_df['y_err'] = 2*np.sqrt(var)
pred_df['f_err'] = 2*np.sqrt(var2)
pred_df['type'] = 'prediction'

# put original observations in same format as above
obs_df = X_c.copy(deep=True)
obs_df['y'] = y.values
obs_df['y_err'] = 0
obs_df['f_err'] = 0
obs_df['type'] = 'observed'

# Combine predictions and observations and rescale variables. 
all_df = pd.concat([pred_df,obs_df], axis=0)
tmp = scaler.inverse_transform(all_df.loc[:, ['m_s', 'n_s', 't_s']])
for i, x in enumerate(to_scale):
    all_df.loc[:, x] = tmp[:, i]

all_df.to_csv('results/all_chi_psi_and_obs.csv')

# ## Pred vs Obs
with model:
    mu, var = gp.predict(X_c.values, point=mp, diag=True,pred_noise=True)
    _, var2 = gp.predict(X_c.values, point=mp, diag=True,pred_noise=False)

with sns.plotting_context(font_scale=1.25):
    lims = (1.1, 2.4)
    cols = sns.color_palette('colorblind')    
#     fig, ax = plt.subplots(1)
    obs_preds = pd.DataFrame({'y observed':y.values, 'y prediction': mu, 'yerr':2*np.sqrt(var),
                                         'ferr': 2*np.sqrt(var2), 'basis': X['basis']})
    g = sns.FacetGrid(data=obs_preds,
                     col='basis', col_wrap=5)
    g.map(plt.errorbar, 'y observed', 'y prediction', 'yerr', color=cols[0], alpha=0.5,
                lw=0, elinewidth=1)
    g.map(plt.errorbar, 'y observed', 'y prediction', 'ferr', color=cols[0], marker='o',
                lw=0, elinewidth=2, alpha=0.5)
    g.set_titles(r'$\chi$: {col_name}')
    for ax in g.axes.flatten():
        ax.plot(lims, lims, label='$y=x$', color='k')
        ax.set_ylim(lims)
        ax.set_xlim(lims)
    plt.savefig('figures/all_chi_gp_pred-vs-obs.pdf', bbox_inches='tight')

obs_preds.to_csv('results/all_chi_preds_and_obs.csv')

# ## Fit Bayesian model

def fit_model_2(y, X, kernel_type='M32', n=1000, n_chains=2):
    """
    function to return a pymc3 model
    y : dependent variable
    X : independent variables
    prop_Xu : number of inducing varibles to use
    
    X, y are dataframes. We'll use the column names. 
    """
    with pm.Model() as model:
        # Covert arrays
        X_a = X.values
        y_a = y.values
        X_cols = list(X.columns)
        
        # Globals
        prop_Xu = 0.01
        l_prior = gamma(1, 0.05)
        eta_prior = hcauchy(2)
        sigma_prior = hcauchy(2)

        # Kernels
        # 3 way interaction
        eta = eta_prior('eta')
        cov = eta**2
        for i in range(X_a.shape[1]):
            var_lab = 'l_'+X_cols[i]
            if kernel_type=='RBF':
                cov = cov*pm.gp.cov.ExpQuad(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type=='Exponential':
                cov = cov*pm.gp.cov.Exponential(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type=='M52':
                cov = cov*pm.gp.cov.Matern52(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type=='M32':
                cov = cov*pm.gp.cov.Matern32(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])

        # Covariance model
        cov_tot = cov 

        # Model
        gp = pm.gp.MarginalSparse(cov_func=cov_tot, approx="FITC")

        # Noise model
        sigma_n =sigma_prior('sigma_n')

        # Inducing variables
        num_Xu = int(X_a.shape[0]*prop_Xu)
        Xu = pm.gp.util.kmeans_inducing_points(num_Xu, X_a)

        # Marginal likelihood
        y_ = gp.marginal_likelihood('y_', X=X_a, y=y_a,Xu=Xu, noise=sigma_n)
        trace = pm.sample(draws=n, chains=n_chains, cores=1)
        
    return gp, trace, model


# In[39]:


gp, trace, model = fit_model_2(y=y, X=X_c, kernel_type='M32')
pickle.dump(
    {'gp': gp, 'trace': trace, 'model': model},
    open('results/all_chi_bayesian_model.p', 'wb')
)


df_trace = pd.DataFrame({x : trace.get_values(x) for x in trace.varnames if x[-5:]!='log__'})


len_labs = [x for x in list(df_trace.columns) if x[0]=='l']

relevance = pd.DataFrame(1/df_trace.loc[:, len_labs].values, columns=len_labs)
relevance_m = relevance.melt(var_name='Hyperparameter', value_name='Relevance')
with sns.plotting_context('paper', font_scale=1.25):
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=relevance_m, x='Relevance', y='Hyperparameter', whis=2)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(tk.StrMethodFormatter('{x:4.2f}'))
    ax.xaxis.set_minor_locator(tk.LogLocator(base=10.0, subs='auto', numdecs=4))
    ax.tick_params(which='minor', axis='x', bottom=True, direction='in')
    plt.savefig('figures/all_chi_gp_bayes_relevance.png', dpi=450 )


# In[132]:


all_params = relevance.join(df_trace.loc[:, [r'$\eta$', r'$\sigma_n$']])

all_params.melt(var_name='Parameter').groupby('Parameter')['value'].aggregate(**{"Median": lambda x: "{:4.2f}".format(np.median(x)), 
             "95% CI": lambda x: "({0:4.2f}, {1:4.2f})".format(np.quantile(x, 0.025),np.quantile(x, 0.975))}).\
reset_index().\
to_csv('results/ppo_gp_bayes_posterior.csv', index=False)




