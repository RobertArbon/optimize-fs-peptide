from mdtraj import compute_distances, baker_hubbard
from msmbuilder.feature_extraction import Featurizer


class HBondFeaturizer(Featurizer):
    def __init__(self, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False, indices=[]):
        '''Parameters:	
        freq = float, default 0.1, returns the Hbonds that occur in greater this fraction of the frames in the traj
        exclude_water = bool, default True
        periodic = bool, default True, set to True to calculate displacements and angles across periodic box boundaries
        sidechain_only = bool, default False

        returns: A 2D array of hydrogen atom-acceptor distances in each frame (in nm), array n_frames x n_bonds for 
         each traj '''
        self.freq = freq
        self.exclude_water = exclude_water
        self.periodic = periodic
        self.sidechain_only = sidechain_only
        self.indices = indices

    def fit(self, ):

    def partial_transform(self, traj):
        f = md.baker_hubbard(traj, freq=self.freq, exclude_water=self.exclude_water, periodic=self.periodic,
                             sidechain_only=self.sidechain_only)
        # TODO need to use the same indicies for each trajectory.
        Hbond_pairs = f[:, -2:]
        result = md.compute_distances(traj, Hbond_pairs, periodic=self.periodic, opt=True)
        return result