from osprey.config import Config
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from msmbuilder.feature_selection import VarianceThreshold, FeatureSelector
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from os.path import join
import os
from glob import glob
import numpy as np
from multiprocessing import Pool
import pandas as pd
from msmbuilder.featurizer import DihedralFeaturizer, KappaAngleFeaturizer
from sklearn.model_selection import cross_val_score, cross_val_predict


# Globals
# traj_dir = '/mnt/storage/home/ra15808/scratch/train'
traj_dir = '/Users/robert_arbon/Datasets/DHFR/train'

cv = ShuffleSplit(n_splits=2, test_size=0.5)


pipe_fixed = Pipeline([
            ('variance_cut', VarianceThreshold()),
           ('tica', tICA(kinetic_mapping=True)),
           ('cluster', MiniBatchKMeans()),
           ('msm', MarkovStateModel(n_timescales=2, lag_time=50, verbose=True))])

pipe_csp = Pipeline([
            ('variance_cut', VarianceThreshold()),
           ('tica', tICA(kinetic_mapping=True)),
           ('cluster', MiniBatchKMeans()),
           ('msm', MarkovStateModel(use_gap='timescales', lag_time=50, verbose=True))])

best = pd.read_pickle('best_trials.pickl')

best.sort_values(by='rank', inplace=True)

for i, row in best.iterrows():
    # Choose pipeline
    if row['strategy'] == 'fixed':
        pipe = pipe_fixed
    elif row['strategy'] == 'csp':
        pipe = pipe_csp

    # Get dataset
    traj_paths = glob(join(traj_dir, row['feature'], '*.npy'))

    trajs = [np.load(traj_path) for traj_path in traj_paths]
    if not len(trajs) > 0:
        print('Error!  No trajectories')
        print(traj_paths)
        continue

    # Set params
    params = row.filter(regex='.*__.*').to_dict()
    print(params)
    pipe.set_params(**params)

    test_scores = cross_val_score(pipe, trajs, cv=cv, n_jobs=2, verbose=1)
    print(test_scores)