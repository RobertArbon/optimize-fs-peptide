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

# Globals
num_procs = 5
# traj_dir = '/mnt/storage/home/ra15808/scratch/train'
traj_dir = '/Users/robert_arbon/Datasets/DHFR/train'

# Pipelines
pipe = Pipeline([
            ('variance_cut', VarianceThreshold()),
           ('tica', tICA(kinetic_mapping=True)),
           ('cluster', MiniBatchKMeans()),
            ('msm', MarkovStateModel(n_timescales=2, lag_time=50, verbose=True))])

# Get old results
best = pd.read_pickle('best_trials.pickl')
best.sort_values(by='feature', inplace=True)

# New timescales to try
new_timescales = [3, 4, 5, 10, 20]

# Setup results dictionary
results = {'id': [], 'strategy': []}
for n_ts in new_timescales:
    results['test_scores-{}'.format(n_ts)] = []
results['final_timescales'] = []

# Loop:
old_feature = 'none'
for i, row in best.iterrows():

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

    results['id'].append(row['id'])
    results['strategy'].append(row['strategy'])

    # This is really inefficient
    for n_ts in new_timescales:
        cv = ShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
        pipe.set_params(msm__n_timescales=n_ts)
        try:
            test_scores = cross_val_score(pipe, trajs, cv=cv, n_jobs=num_procs, pre_dispatch=num_procs, verbose=1)
        except:
            test_scores = [None]

        results['test_scores-{}'.format(n_ts)].append(test_scores)

new_results = pd.DataFrame(data=results)
new_results = new_results.merge(right=best, on=['id','strategy'], how='inner')
new_results.to_pickle('best_trials_rescored.pickl')