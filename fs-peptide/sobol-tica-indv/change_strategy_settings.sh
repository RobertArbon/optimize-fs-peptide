#!/bin/bash


#for f in `ls *.yaml`
#do
#    W=${f%.*}
#    cp $f $W
#    sed -i '.original' 's/name\: hyperopt_tpe/name\: sobol/' $f
#    sed -i '.original' '/params:/d' $f
#    sed -i '.original' '/seeds:/d' $f
#    sed -i '.original' '/gamma:/d' $f
#
#done

for f in `ls *.yaml`
do
    sed -i '.original' 's/    name\: shufflesplit/    name: shufflesplit\
    params\:/' $f
    sed -i '.original' 's/  name\: mdtraj/  name\: mdtraj\
  params\:/' $f

done
