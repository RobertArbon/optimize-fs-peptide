#!/bin/bash

for f in `ls *.yaml`
do
   name=${f%.yaml}
   echo $name
   cp submit.sh submit-$name.sh
   sed -i "s/f\=xxx/f\=$f/g" submit-$name.sh 
   qsub -N $name submit-$name.sh 
done
