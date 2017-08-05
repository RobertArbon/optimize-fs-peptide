"""
This takes and old database of trials and re-runs the various models using a different scoring method. 
"""
from osprey.config import Config
from sklearn.pipeline import Pipeline
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

# the pipeline to use
pipe = Pipeline([('variance_cut', VarianceThreshold()),
          ('tica', tICA(kinetic_mapping=True)),
          ('cluster', MiniBatchKMeans()),
          ('msm', MarkovStateModel(use_gap='timescales', lag_time=50, verbose=True))])


def load_trajectoris(feat):
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

    config = Config(config_path)
    trials = config.trial_results()

    ## Do it in parallel
    # with Pool() as pool:
    #     dihed_trajs = dict(pool.imap_unordered(feat, meta.iterrows()))
    new_trial_params = [get_parameters(irow) for irow in trials.iterrows()]
    print(new_trial_params[0])






