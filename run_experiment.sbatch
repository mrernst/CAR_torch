#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=700:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --partition=sleuths
#SBATCH --job-name=protoengine_experiment_1
#SBATCH --mail-type=END
#SBATCH --mail-user=mernst@fias.uni-frankfurt.de
#SBATCH --output=/home/mernst/git/titan/experiments/protoengine_experiment_1/files/slurm_output/noname_experiment_slurm_%j.out
#SBATCH --array=0%1
#SBATCH --reservation triesch-shared


j=${job_array[$((SLURM_ARRAY_TASK_ID))]}
for i in `seq 1 1 1`
do
    echo "iteration $i"
    echo "job $j"
    srun python3 protoengine.py
done

# TODO: Make this more elegant
