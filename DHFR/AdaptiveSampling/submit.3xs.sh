#!/bin/bash


for i in {0..3}
do
   nohup osprey worker $1 -n 50 --seed $i > $1.$i.log & 
done
