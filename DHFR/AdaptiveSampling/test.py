import mdtraj as md
import numpy as np
#    trajectories: ~/Datasets/DHFR/train/5dfr-trajectory-300K-*.dcd
#    topology: ~/Datasets/DHFR/train/top.pdb
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel

ftrajs  = [np.load('/panfs/panasas01/chem/ra15808/Datasets/DHFR/train/ftraj-{}.npy'.format(i)) for i in range(51)] 

tica = tICA(n_components=8, lag_time=358)
ttrajs = tica.fit_transform(ftrajs)

cluster = MiniBatchKMeans(n_clusters=383)
ctrajs = cluster.fit_transform(ttrajs)

msm = MarkovStateModel(n_timescales=2, lag_time=50, verbose=False)

msm.fit(ctrajs)

print(msm.score(ctrajs))


