#!/usr/bin/env bash


rsync -av *.npy ra15808@bluecrystalp3.bris.ac.uk:~/hyperparamopt/fs-peptide/
rsync -av data/helix-reference.xtc ra15808@bluecrystalp3.bris.ac.uk:~/hyperparamopt/fs-peptide/data
