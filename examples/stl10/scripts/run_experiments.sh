#!/bin/bash

#SBATCH -o logs/series_unsupervisedcond.log-%j-%a
#SBATCH -a 0-9  # Adjust this to match the number of files
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
module load cuda/11.8


python /home/gridsan/herol/usmatching_au_new_change/examples/stl10/main_unsupervised_cond.py --iter $SLURM_ARRAY_TASK_ID --num_workers 20
# python /home/gridsan/herol/usmatching_au_change/examples/stl10/main_$1.py experiments/experiment_$file_idx.json --iter $iter_idx