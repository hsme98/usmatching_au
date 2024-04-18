#!/bin/bash

#SBATCH -o logs/eval_unsupervisedcond.log-%j-%a
#SBATCH -a 0-6  # Adjust this to match the number of files n_eval x tot_files - 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
n_eval=7

let "file_idx = SLURM_ARRAY_TASK_ID / n_eval"
let "iter_idx = SLURM_ARRAY_TASK_ID % n_eval"

echo $file_idx
echo $iter_idx


python /home/gridsan/herol/usmatching_au_nn/examples/stl10/main_unsupervised.py --dataset imagenet --iter $iter_idx --num_workers 20
