### The model to be pickles and then saved.
from sklearn.pipeline import Pipeline
from msmbuilder.feature_extraction import DihedralFeaturizer, ContactFeaturizer
from msmbuilder.feature_selection import FeatureSelector, VarianceThreshold
from msmbuilder.preprocessing import RobustScaler
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from msmbuilder.io import save_generic
from sklearn.base import clone, BaseEstimator
from six import iteritems


# The data will be loaded with a stride of 10 frames.  Each fame is 50ps, so the time per frame will be
# 500ps/frame or 0.5ns/frame.
# Each trajectory is 1000 frames long
# Lag time will be 40 frames (20 ns)  based on a visual inspection of /Misc/MSM_lag_time.ipynb
to_ns = 0.5
msm_lag = int(40/to_ns)

# Put all the features here but only select one at a time

feats = [('bb1_dihed', DihedralFeaturizer(types=['phi', 'psi'])),
         ('bb2_dihed', DihedralFeaturizer(types=['phi', 'psi', 'omega'])),
         ('res_dihed', DihedralFeaturizer(types=['chi1', 'chi2', 'chi3', 'chi4'])),
         ('contacts', ContactFeaturizer())]

featurizer = FeatureSelector(features=feats)

pipe = Pipeline([('features', featurizer),
                 ('variance_cut', VarianceThreshold()),
                 ('scaling', RobustScaler()),
                 ('tica', tICA(kinetic_mapping=True)),
                 ('cluster', MiniBatchKMeans()),
                 ('msm', MarkovStateModel(lag_time=msm_lag, verbose=False))])

save_generic(pipe, 'model.pickl')




