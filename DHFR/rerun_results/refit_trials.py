from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from msmbuilder.feature_selection import VarianceThreshold, FeatureSelector
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from os.path import join
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import sys

"""
Refits trials with all the data and records the timescales.
"""

# Globals
#traj_dir = '/mnt/storage/home/ra15808/scratch/train'
# traj_dir = '/Users/robert_arbon/Datasets/DHFR/train'
traj_dir = '/panfs/panasas01/chem/ra15808/Datasets/DHFR/train'

trial_db = 'best_trials.pickl'
output_db = trial_db.split('.')[0]+'-'+'all_ts'+'.pickl'

# Pipelines
pipe = Pipeline([
            ('variance_cut', VarianceThreshold()),
           ('tica', tICA(kinetic_mapping=True)),
           ('cluster', MiniBatchKMeans()),
            ('msm', MarkovStateModel(n_timescales=2, lag_time=50, verbose=True))])

# Get old results
best = pd.read_pickle(trial_db)
best.sort_values(by='feature', inplace=True)

# Setup results dictionary
results = {'id': [], 'strategy': [], 'all_timescales': []}

# Loop
old_feature = 'none'
for i, row in best.iterrows():
    print('---Running {}---'.format(i))

    # Get dataset
    if row['feature'] != old_feature:
        traj_paths = glob(join(traj_dir, row['feature'], '*.npy'))
        old_feature = row['feature']
        trajs = [np.load(traj_path) for traj_path in traj_paths]

        if not len(trajs) > 0:
            print('Error!  No trajectories')
            print(traj_paths)
            continue

    # Set params
    params = row.filter(regex='.*__.*').to_dict()
    pipe.set_params(**params)
    pipe.set_params(msm__n_timescales=None)
    pipe.fit(trajs)

    results['id'].append(row['id'])
    results['strategy'].append(row['strategy'])
    all_ts = pipe.named_steps['msm'].timescales_
    results['all_timescales'].append(all_ts)

# Save results
new_results = pd.DataFrame(data=results)
new_results = new_results.merge(right=best, on=['id','strategy'], how='inner')
new_results.to_pickle(output_db)
