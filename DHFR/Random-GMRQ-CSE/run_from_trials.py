"""
This takes and old database of trials and re-runs the various models using a different scoring method. 
"""
from osprey.config import Config
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from msmbuilder.feature_selection import VarianceThreshold
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from os.path import join
from glob import glob
import numpy as np

# Globals
config_path = '../../Trial Data/DHFR/Random-GMRQ-2/alpha_angle.yaml'
db_path = '../../Trial Data/DHFR/Random-GMRQ-2/osprey-trials.db'
traj_dir = '/home/robert/Datasets/DHFR/train'


def get_pipeline(parameters):
    """
    Wrapper so that new instance of a pipeline can be instantiated for every fold. 
    :return: sklean.pipeline.Pipeline object
    """
    pipe = Pipeline([('variance_cut', VarianceThreshold()),
                     ('tica', tICA(kinetic_mapping=True)),
                     ('cluster', MiniBatchKMeans()),
                     ('msm', MarkovStateModel(use_gap='timescales', lag_time=50, verbose=True))])
    pipe.set_params(**parameters)

    return pipe


# cross validation iterator
# TODO get this from the config file
# cv:
#     name: shufflesplit
#     params:
#       n_splits: 5
#       test_size: 0.5
cv = ShuffleSplit(n_splits=5, test_size=0.5)


def get_trajectories(feat):
    """
    Gets the trajctories associated with a feature
    :param feat: 
    :return: 
    """
    traj_paths = glob(join(traj_dir, feat, '*'))
    trajs = [np.load(traj_path) for traj_path in traj_paths]
    return trajs


def get_parameters(irow):
    """
    Gets the parameters for running a new model. 
    :return: dictionary of parameters
    """
    i, row = irow
    params_dict = row['parameters']

    params = {}
    trial_config = {}

    params['tica__lag_time'] = params_dict['tica__lag_time']
    params['tica__n_components'] = params_dict['tica__n_components']
    params['cluster__n_clusters'] = params_dict['cluster__n_clusters']

    trial_config['params'] = params
    trial_config['feature'] = row['project_name']
    trial_config['row'] = i

    return trial_config


if __name__ == "__main__":
    np.random.seed(42)
    config = Config(config_path)
    trials = config.trial_results()
    trials = trials.iloc[:5,:]
    print('original df: ')
    print(trials.head())

    ## Do it in parallel
    # with Pool() as pool:
    #     dihed_trajs = dict(pool.imap_unordered(feat, meta.iterrows()))
    new_trial_params = [get_parameters(irow) for irow in trials.iterrows()]

    trial_config = new_trial_params[0]

    feat = trial_config['feature']
    X = get_trajectories(feat)

    row_num = trial_config['row']
    params = trial_config['params']

    train_timescales = []
    test_timescales = []
    train_gaps = []
    test_gaps = []
    train_scores = []
    test_scores = []

    for train_idx, test_idx in cv.split(X):
        # the pipeline to use
        pipe = get_pipeline(params)

        train = [X[i] for i in train_idx]
        test = [X[i] for i in test_idx]
        pipe.fit(train)
        train_timescales.append(pipe.named_steps['msm'].n_timescales)
        train_gaps.append(pipe.named_steps['msm'].gap_)
        train_scores.append(pipe.score(train))

        # train_scores.append(pipe.score(X[train]))
    print(train_timescales)
    print(train_gaps)
    print(train_scores)

    # trials.ix[row_num,'cse_n_timescales'] =





