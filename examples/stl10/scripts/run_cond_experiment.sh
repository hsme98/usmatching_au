#!/bin/bash

#SBATCH -o logs/single_unsupervisedcond_exp.log-%j
#SBATCH -c 64
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
module load cuda/11.8

python /home/gridsan/herol/usmatching_au_change/examples/stl10/main_unsupervised_cond.py --feat_dim 32 --num_workers 64 --batch_size 768