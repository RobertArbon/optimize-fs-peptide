### The model to be pickles and then saved.
from sklearn.pipeline import Pipeline
from msmbuilder.feature_extraction import *
from msmbuilder.feature_selection import FeatureSelector, VarianceThreshold
from msmbuilder.preprocessing import RobustScaler
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from msmbuilder.io import save_generic
import numpy as np
import mdtraj as md

# (a) α-angles,
# (b) α-carbon contact distances,
# (c) pairwise α-carbon RMSD,
# (d) tICs from α-angles, and
# (e) tICs from α-carbon contact distances.
# All clustering was performed with the mini-batch k-medoids
# Optimized parameter selection reveals trends in Markov state models for protein folding
# Brooke E. Husic, Robert T. McGibbon, Mohammad M. Sultan, and Vijay S. Pande
# Citation:

#
# TIMESCALES
#
# The data will be loaded with a stride of 10 frames.  Each fame is 50ps, so the time per frame will be
# 500ps/frame or 0.5ns/frame.
# Each trajectory is 1000 frames long
# Lag time will be 40 frames (20 ns)  based on a visual inspection of /Misc/MSM_lag_time.ipynb
to_ns = 0.5
msm_lag = int(40/to_ns)

#
# FEATURE INDICES
#
all_idx = np.load('indices_all.npy')

#
# OTHER PARAMETERS
#
ref_traj = md.load('../Data/data/trajectory-1.xtc', top='../Data/data/fs-peptide.pdb')



featurizer = FeatureSelector(features=feats)

pipe = Pipeline([('features', featurizer),
                 ('variance_cut', VarianceThreshold()),
                 ('scaling', RobustScaler()),
                 ('cluster', MiniBatchKMeans()),
                 ('msm', MarkovStateModel(lag_time=msm_lag, verbose=False))])

save_generic(pipe, 'model.pickl')




