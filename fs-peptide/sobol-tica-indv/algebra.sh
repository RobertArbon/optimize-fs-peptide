#!/bin/bash

NO_OF_CORES=14
i=1
for f in `ls *.yaml`
do 
  if [ $i -le $NO_OF_CORES ]
  then
     echo $f, $i 
  fi
  (( i += 1 )) 
done 
