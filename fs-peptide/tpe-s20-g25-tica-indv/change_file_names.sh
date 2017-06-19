#!/usr/bin/env bash

for f in `ls *.yaml`
do
#    W=${f%.*}
    V=${f#config_random-}
    cp $f config-$V
done