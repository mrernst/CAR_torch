#!/bin/bash 
# 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --time=700:00:00 
#SBATCH --mem=30GB 
#SBATCH --exclude vane 
#SBATCH --partition=sleuths 
#SBATCH --job-name=linear_evaluation 
#SBATCH --mail-type=END 
#SBATCH --mail-user=mernst@fias.uni-frankfurt.de 
#SBATCH --output=/home/mernst/git/titan/network_engine/utilities/noname_experiment_slurm_%j.out 

j=$((SLURM_ARRAY_TASK_ID)) 
for i in `seq 1 1 1` 
do 
    echo "iteration $i" 
    echo "job $j" 
    srun python3 linear_eval.py
done 
# --- end of experiment --- 
