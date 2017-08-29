from multiprocessing import Pool, cpu_count
from functools import reduce
import os
from osprey.config import Config
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from msmbuilder.feature_selection import VarianceThreshold
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from os.path import join
import os
from glob import glob
import numpy as np
from multiprocessing import Pool
import pandas as pd

config_path = '../Random-GMRQ-2/alpha_angle.yaml'
db_path = '../Random-GMRQ-2/osprey-trials.db'
traj_dir = '/mnt/storage/home/ra15808/Datasets/DHFR/train'

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
cv = ShuffleSplit(n_splits=2, test_size=0.5)

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
    trial_config['id'] = row['id']

    return trial_config

def run_trial(trial_config):

    id_num = trial_config['id']
    print('Running trial {}'.format(id_num))
    #X = get_trajectories(trial_config['feature'])
    #
    #train_scores = []
    #train_gaps = []
    #train_n_timescales = []
    #test_scores = []

    #for train_idx, test_idx in cv.split(X):
    #    pipe = get_pipeline(trial_config['params'])

    #    train = [X[idx] for idx in train_idx]
    #    try:
    #        pipe.fit(train)
    #        train_n_timescales.append(pipe.named_steps['msm'].n_timescales)
    #        train_gaps.append(pipe.named_steps['msm'].gap_)
    #        train_scores.append(pipe.score(train))
    #    except:
    #        print('Error in training trial {} - setting results to None'.format(id_num))
    #        train_n_timescales.append(None)
    #        train_gaps.append(None)
    #        train_scores.append(None)

    #    test = [X[idx] for idx in test_idx]
    #    try:
    #        score = pipe.score(test)
    #    except:
    #        print('Error in test trial {} - setting results to None'.format(id_num))
    #        score = None
    #    test_scores.append(score)

    #results = {'id': id_num, 'cse_train_scores': train_scores, 'cse_train_gaps': train_gaps,
    #           'cse_train_n_timescales': train_n_timescales, 'cse_test_scores': test_scores }
    #return results


def square(x):
    """Function to return the square of the argument"""
    return x*x

def output(x):
    print(x)

if __name__ == "__main__":
    # print the number of cores
    print("Number of cores available equals %d" % cpu_count())

    np.random.seed(42)
    config = Config(config_path)
    trials = config.trial_results()
    trials = trials.iloc[:10,:]
    trial_configs = [get_parameters(irow) for irow in trials.iterrows()]

    # create a pool of workers
    n_cpu=int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    print('Number of cpus being used {}'.format(n_cpu))

    pool = Pool(n_cpu)
    print(pool)

    ### Test    

    results = pool.imap_unordered(run_trial, trial_configs)


    ### These work: 

    #pool.map(output, trial_configs)

    #a = range(1,50001)
    #result = pool.map( square, a )
    #total = reduce( lambda x,y: x+y, result )
    #print("The sum of the square of the first 50000 integers is %d" % total)

    ### Dustbin
    # pool.imap_unordered(output, trial_configs)
    # pool.imap(output, trial_configs)
