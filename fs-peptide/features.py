import mdtraj as md
import numpy as np
from msm_functions import angles, pair_rmsd
from msmbuilder.feature_extraction import *

# Citation:
# (a) α-angles,
# (b) α-carbon contact distances,
# (c) pairwise α-carbon RMSD,
# (d) tICs from α-angles, and - (Y)
# (e) tICs from α-carbon contact distances. - (Y)
# All clustering was performed with the mini-batch k-medoids
# Optimized parameter selection reveals trends in Markov state models for protein folding
# Brooke E. Husic, Robert T. McGibbon, Mohammad M. Sultan, and Vijay S. Pande

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

# NEED HTICA:
# ('nb_pairs', AtomPairsFeaturizer(pair_indices=nb_pairs_idx, periodic=True)),

# ('ff_bb_tor', FunctionFeaturizer(dihedrals, func_args={'indices': bb_dihed_idx})),
# ('ff_re_tor', FunctionFeaturizer(dihedrals, func_args={'indices': re_dihed_idx})),

tica_unstructured_features = \
        [
            ('superposed', SuperposeFeaturizer(atom_indices=all_idx, reference_traj=ref_traj)),
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
            ('ff_all_ang', FunctionFeaturizer(angles, func_args={'indices': all_angle_idx}))
         ]

tica_structured_features = \
    [
        ('vM_pp_tor', VonMisesFeaturizer(types=['phi', 'psi'],n_bins=18, kappa=20)),
        ('vM_ppo_tor', VonMisesFeaturizer(types=['phi', 'psi', 'omega'], n_bins=18, kappa=20)),
        ('vM_chi_tor', VonMisesFeaturizer(types=['chi1', 'chi2', 'chi3', 'chi4'], n_bins=18, kappa=20)),
        ('ca_log-cont', LogisticContactFeaturizer(contacts='all', scheme='ca', center=0.8, steepness=20)),
        ('close_log-cont', LogisticContactFeaturizer(contacts='all', scheme='closest', center=0.8, steepness=20)),
        ('close-h_log-cont', LogisticContactFeaturizer(contacts='all', scheme='closest-heavy', center=0.8, steepness=20)),
        ('ca_b-cont', BinaryContactFeaturizer(contacts='all', scheme='ca', cutoff=0.8)),
        ('close_b-cont', BinaryContactFeaturizer(contacts='all', scheme='closest', cutoff=0.8)),
        ('close-h_b-cont', BinaryContactFeaturizer(contacts='all', scheme='closest-heavy', cutoff=0.8))
    ]

# These features don't go into tICA
non_tica_features = \
        [('rmsd', RMSDFeaturizer(reference_traj=ref_traj[0], atom_indices=all_idx)),
         ('kern-rmsd', LandMarkRMSDFeaturizer(reference_traj=ref_traj[0], atom_indices=all_idx, sigma=1)),
         ('atom-sasa', SASAFeaturizer(mode='atom', probe_radius=0.14, n_sphere_points=960)),
         ('res-sasa', SASAFeaturizer(mode='residue', probe_radius=0.14, n_sphere_points=960)),
         # ('ca_cont_sm', ContactFeaturizer(contacts='all', scheme='ca', soft_min=True, soft_min_beta=20)),
         ('ca_cont', ContactFeaturizer(contacts='all', scheme='ca')),
         ('rmsd_ca_pair', FunctionFeaturizer(pair_rmsd, func_args={'pairs': ca_pairs_idx}))]