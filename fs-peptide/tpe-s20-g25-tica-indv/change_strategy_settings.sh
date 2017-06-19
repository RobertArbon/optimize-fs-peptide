#!/bin/bash


for f in `ls *.yaml.original`
do
    W=${f%.*}
    cp $f $W
    sed -i '.original' 's/name\: random/name\: hyperopt_tpe\
    params\:\
        seeds\: 20\
        gamma\: 0.25/' $W
done
