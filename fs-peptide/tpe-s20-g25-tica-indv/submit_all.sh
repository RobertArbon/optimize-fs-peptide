#!/bin/bash

i=1
for f in `ls *.yaml`; do
    name=${f%.*}
    name=${name#config-}
    name=tpe-$name
    echo $f, $name, $i
    qsub -v INPUT=$f,SEEDNUM=$i -N $name submit_indv.sh 
    (( i += 1 ))
done
#        qsub -v RESTART=$rstfile,INPUT=$inpfile -N 100ns-$dir submit.sh


