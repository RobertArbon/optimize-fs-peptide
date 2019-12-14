import mdtraj as md
import numpy as np


def dihedrals(traj, indices=None):
    x = []
    y = md.compute_dihedrals(traj, indices=indices, periodic=True)
    x.extend([np.sin(y), np.cos(y)])
    return np.hstack(x)


def angles(traj, indices=None):
    x = []
    y = md.compute_angles(traj, angle_indices=indices, periodic=True)
    x.extend([np.sin(y), np.cos(y)])
    return np.hstack(x)

def pair_rmsd(traj, pairs=None):
    y = md.compute_distances(traj, atom_pairs=pairs)
    rmsd = np.sqrt(np.mean(y**2, axis=1))
    rmsd = rmsd[:, np.newaxis]
    return rmsd