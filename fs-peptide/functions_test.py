import mdtraj as md
import msm_functions as f
import numpy as np
from msmbuilder.feature_extraction import FunctionFeaturizer, RMSDFeaturizer

ref_traj = md.load('data/trajectory-1.xtc', top='data/fs-peptide.pdb', stride=10)


dihed_idx = np.load('dihed_bb.npy')
angles_idx = np.load('angles_bb.npy')
ca_pairs_idx = np.load('ca_pairs.npy')
all_idx = np.load('indices_all.npy')

print('Dihedrals shape:', dihed_idx.shape)
ftraj = f.dihedrals(traj=ref_traj, indices=dihed_idx)
print('ftraj shape: ', ftraj.shape)
feat = FunctionFeaturizer(f.dihedrals, func_args={'indices': dihed_idx})
ftraj = feat.fit_transform([ref_traj])
print('FF ftraj shape: ', ftraj[0].shape)
print()
print('Angles shape:', angles_idx.shape)
ftraj = f.angles(traj=ref_traj, indices=angles_idx)
print('ftraj shape: ', ftraj.shape)
feat = FunctionFeaturizer(f.angles, func_args={'indices': angles_idx})
ftraj = feat.fit_transform([ref_traj])
print('FF ftraj shape: ', ftraj[0].shape)
print()

print('CA shape', ca_pairs_idx.shape)
ftraj = f.pair_rmsd(ref_traj, pairs=ca_pairs_idx)
print('ftraj shape', ftraj.shape)
feat = FunctionFeaturizer(f.pair_rmsd, func_args={'pairs': ca_pairs_idx})
ftraj = feat.fit_transform([ref_traj])
print('FF ftraj shape: ', ftraj[0].shape)
print()

feat = RMSDFeaturizer(reference_traj=ref_traj[0], atom_indices=all_idx)
ftraj = feat.fit_transform([ref_traj])
print('RMSD featurizier shape ', ftraj[0].shape)

