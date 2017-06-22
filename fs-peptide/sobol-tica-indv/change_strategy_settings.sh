#!/bin/bash


for f in `ls *.yaml`
do
    W=${f%.*}
    cp $f $W
    sed -i '.original' 's/name\: hyperopt_tpe/name\: sobol/' $f
    sed -i '.original' '/params:/d' $f
    sed -i '.original' '/seeds:/d' $f
    sed -i '.original' '/gamma:/d' $f

done
