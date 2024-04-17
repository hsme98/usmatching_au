#!/bin/bash

#SBATCH -o logs/series_unsupervised.log-%j
#SBATCH -a 0-9
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
module load cuda/11.8

python /home/gridsan/herol/usmatching_au_new_change/examples/stl10/main_unsupervised.py --num_workers 20 --iter $SLURM_ARRAY_TASK_ID