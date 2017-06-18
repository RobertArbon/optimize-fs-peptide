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
ref_traj = md.load('../Data/fs_peptide/trajectory-1.xtc', top='../Data/fs_peptide/fs-peptide.pdb')

#
# FEATURES
#
# Put all the features here but only select one at a time.  Specify variables to be optimized in config file only.
# Not including bonds as they are contstrained

feats = [('superposed', SuperposeFeaturizer(atom_indices=all_idx, reference_traj=ref_traj)),
         ('rmsd', RMSDFeaturizer(reference_traj=ref_traj[0], atom_indices=all_idx)),
         ('kern-rmsd', LandMarkRMSDFeaturizer(reference_traj=ref_traj[0], atom_indices=all_idx, sigma=1)),
         ('')
         ('pp_tor', DihedralFeaturizer(types=['phi', 'psi'])),
         ('ppo_tor', DihedralFeaturizer(types=['phi', 'psi', 'omega'])),
         ('chi_tor', DihedralFeaturizer(types=['chi1', 'chi2', 'chi3', 'chi4'])),
         ('contacts', ContactFeaturizer()),
         ]

featurizer = FeatureSelector(features=feats)

pipe = Pipeline([('features', featurizer),
                 ('variance_cut', VarianceThreshold()),
                 ('scaling', RobustScaler()),
                 ('tica', tICA(kinetic_mapping=True)),
                 ('cluster', MiniBatchKMeans()),
                 ('msm', MarkovStateModel(lag_time=msm_lag, verbose=False))])

save_generic(pipe, 'model.pickl')




