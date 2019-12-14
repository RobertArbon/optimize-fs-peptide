#!/bin/bash
#kappa_angle    200
#phi_tor        199
#alpha_angle    198
#psi_tor        195
#omega_tor      193
#phi-o_tor      184
#psi-o_tor      168
#pp_tor          34

name=kappa_angle    
sbatch --export=config=$name,num=11 --job-name=$name submit.sh
name=phi_tor       
sbatch --export=config=$name,num=11 --job-name=$name submit.sh
name=alpha_angle   
sbatch --export=config=$name,num=11 --job-name=$name submit.sh
name=psi_tor       
sbatch --export=config=$name,num=11 --job-name=$name submit.sh
name=omega_tor     
sbatch --export=config=$name,num=11 --job-name=$name submit.sh
name=phi-o_tor     
sbatch --export=config=$name,num=12 --job-name=$name submit.sh
name=psi-o_tor     
sbatch --export=config=$name,num=13 --job-name=$name submit.sh
name=pp_tor        
sbatch --export=config=$name,num=40 --job-name=$name submit.sh


#for f in `ls *.yaml`
#do
#   name=${f%.yaml}
#   echo $name
#   sbatch --export=config=$name,num=40 --job-name=$name submit.sh
#done
