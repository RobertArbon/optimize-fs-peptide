#!/bin/bash


#    trajectories: ../data/trajectory-*.xtc
#    topology: ../data/fs-peptide.pdb
for f in `ls *.yaml`
do 
    #sed -i -e 's/..\/data/~\/msmbuilder_data\/fs_peptide/g' $f 
    W=${f%.*}
    V=${W#config_random-}
    echo $V
    sed -i -e "s/  project_name: random/  project_name: $V/g" $f 
done
#V=testing
#sed -i -e "s/  project_name: random/  project_name: $V/g" config_random 
