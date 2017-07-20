import mdtraj as md
import numpy as np
#    trajectories: ~/Datasets/DHFR/train/5dfr-trajectory-300K-*.dcd
#    topology: ~/Datasets/DHFR/train/top.pdb
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from msmbuilder.feature_selection import VarianceThreshold
from sklearn.pipeline import  Pipeline


pipe = Pipeline([('variance_cut', VarianceThreshold()),
          ('tica', tICA(kinetic_mapping=True, n_components=8, lag_time=358)),
          ('cluster', MiniBatchKMeans(n_clusters=383)),
          ('msm', MarkovStateModel(n_timescales=2, lag_time=50, verbose=True))])


ftrajs  = [np.load('../../Data/DHFR/ftraj-{}.npy'.format(i)) for i in range(51)]

pipe.fit(ftrajs)

print(pipe.score(ftrajs))

# tica = tICA(n_components=8, lag_time=358)
# ttrajs = tica.fit_transform(ftrajs)
#
# cluster = MiniBatchKMeans(n_clusters=383)
# ctrajs = cluster.fit_transform(ttrajs)
#
# msm = MarkovStateModel(n_timescales=2, lag_time=50, verbose=False)
#
# msm.fit(ctrajs)
#
# print(msm.score(ctrajs))