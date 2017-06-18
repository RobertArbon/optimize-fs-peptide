from featureselector import FeatureSelector
from mdtraj import load
from msmbuilder.io import load_meta, preload_tops

from Old_Functions.hbonds import HBondFeaturizer

meta = load_meta('meta.pandas.pickl')
tops = preload_tops(meta)
trajs = [load(row['traj_fn'], top=tops[row['top_fn']], stride=10)
         for i, row in meta.iterrows()]

def traj_summary(ftrajs):
    print('Length of ftraj {}'.format(len(ftrajs)))
    for traj in ftrajs:
        print('\t Shape: {}'.format(traj.shape))

def test_HBondFeaturizer(traj_list):

    feat = HBondFeaturizer(freq=0.0)
    feat.fit(traj_list)
    ftraj = feat.transform(traj_list)
    traj_summary(ftraj)


def test_HBondsFeatExtr(traj_list):

    features = [('hbonds', HBondFeaturizer())]
    feat = FeatureSelector(features, which_feat=['hbonds'])
    feat.fit(traj_list)
    ftrajs = feat.transform(traj_list)
    traj_summary(ftrajs)

# test_HBondFeaturizer(trajs)
test_HBondsFeatExtr(trajs)

