### The model to be pickles and then saved.
from os.path import join


from msm_functions import angles
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.decomposition import tICA
from msmbuilder.feature_extraction import *
from msmbuilder.feature_selection import FeatureSelector, VarianceThreshold
from msmbuilder.io import save_generic
from msmbuilder.msm import MarkovStateModel
from msmbuilder.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


#
# FUNCTIONS
#


def print_feature_names(feature_list, path):
    "    - ['something']"
    with open(path, 'w') as fh:
        for feat in feature_list:
            fh.write("        - ['{}']\n".format(feat[0]))


#
# FEATURE INDICES
#
all_idx = np.load('indices_all.npy') # all atoms
nb_pairs_idx = np.load('nb_pairs.npy')  # All non-bonded pairs (bond distances are fixed)
hb_pairs_idx = np.load('hbonds_0pc.npy')  # hydrogen bond triples (D-H-A).  Hydrogen bonds must be present >10% time
bb_dihed_idx = np.load('dihed_bb.npy')  # backbone torsions from forcefield
re_dihed_idx = np.load('dihed_re.npy')  # residue torsions from forcefield
bb_angle_idx = np.load('angles_bb.npy')  # backbone angles from forcefield
re_angle_idx = np.load('angles_re.npy')  # residue angles from forcefield
ca_pairs_idx = np.load('ca_pairs.npy')  # pairs of alpha-carbons
all_angle_idx = np.load('angles_all.npy')  # residue angles from forcefield

#
# OTHER PARAMETERS
#
ref_traj = md.load('data/helix-reference.xtc', top='data/fs-peptide.pdb')

#
# FEATURES
#
# Put all the features here but only select one at a time.  Specify variables to be optimized in config file only.
# Not including bonds as they are constrained
tica_unstructured_features = [('superposed', SuperposeFeaturizer(atom_indices=all_idx, reference_traj=ref_traj)),
            ('hb_pairs', AtomPairsFeaturizer(pair_indices=hb_pairs_idx[:,[1,2]], periodic=True)),
            ('pp_tor', DihedralFeaturizer(types=['phi', 'psi'])),
            ('ppo_tor', DihedralFeaturizer(types=['phi', 'psi', 'omega'])),
            ('chi_tor', DihedralFeaturizer(types=['chi1', 'chi2', 'chi3', 'chi4'])),
            ('all_tor', DihedralFeaturizer(types=['phi', 'psi', 'omega','chi1', 'chi2', 'chi3', 'chi4'])),
            ('alpha_angle', AlphaAngleFeaturizer()),
            ('kappa_angle', KappaAngleFeaturizer()),
            ('ca_cont', ContactFeaturizer(contacts='all', scheme='ca')),
            ('close_cont', ContactFeaturizer(contacts='all', scheme='closest')),
            ('close-h_cont', ContactFeaturizer(contacts='all', scheme='closest-heavy')),
            ('raw_pos', RawPositionsFeaturizer(ref_traj=ref_traj)),
            ('drid', DRIDFeaturizer(atom_indices=all_idx)),
            ('ff_bb_ang', FunctionFeaturizer(angles, func_args={'indices': bb_angle_idx})),
            ('ff_re_ang', FunctionFeaturizer(angles, func_args={'indices': re_angle_idx})),
            ('ff_all_ang', FunctionFeaturizer(angles, func_args={'indices': all_angle_idx}))]

#
# TIMESCALES
#
# The data will be loaded with a stride of 10 frames.  Each fame is 50ps, so the time per frame will be
# 500ps/frame or 0.5ns/frame.
# Each trajectory is 1000 frames long
# Lag time will be 40 frames (20 ns)  based on a visual inspection of /Misc/MSM_lag_time.ipynb

features = tica_unstructured_features
to_ns = 0.5
msm_lag = int(40/to_ns)

#
# MODEL
#
pipe = Pipeline([('features', FeatureSelector(features=tica_unstructured_features)),
                 ('variance_cut', VarianceThreshold()),
                 ('scaling', RobustScaler()),
                 ('tica', tICA(kinetic_mapping=True)),
                 ('cluster', MiniBatchKMeans()),
                 ('msm', MarkovStateModel(lag_time=msm_lag, verbose=False, n_timescales=2))])
#
# SAVE MODEL
#
savedir = 'rand-tica-all'
save_generic(pipe, '{}/model.pickl'.format(savedir))
print_feature_names(features, join(savedir, 'feature_list.txt'))




