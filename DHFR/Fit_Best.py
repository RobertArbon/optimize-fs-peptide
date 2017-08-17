from msmbuilder.feature_extraction import AlphaAngleFeaturizer, KappaAngleFeaturizer, DihedralFeaturizer
from msmbuilder.feature_selection import VarianceThreshold
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from sklearn.pipeline import Pipeline


# index	id	feature	rank	cse_rank	gaps_rank	cluster__n_clusters	tica__n_components	tica__lag_time
# 0	0	340	alpha_angle	1	566	1078	415	3	35
# 2	2	994	alpha_angle	3	621	617	957	4	458
# 1	1	1513	kappa_angle	2	505	1029	965	5	175


# index	id	feature	rank	cse_rank	gaps_rank	cluster__n_clusters	tica__n_components	tica__lag_time
# 316	156	270	psi-o_tor	311	2	1068	148	4	91
# 1325	1165	623	psi_tor	1276	3	692	854	10	1
# 54	54	701	kappa_angle	51	1	737	680	6	412

# model_params = {'340': [AlphaAngleFeaturizer(), 415, 3, 35]}
#
# pipe = Pipeline([ ('features', )
#             ('variance_cut', VarianceThreshold()),
#            ('tica', tICA(kinetic_mapping=True)),
#            ('cluster', MiniBatchKMeans()),
#            ('msm', MarkovStateModel(n_timescales=2, lag_time=50, verbose=True))])