#!/bin/bash

# SBATCH -o logs/series.log-%j-%a
# SBATCH -a 0-39  # Adjust this to match the number of files
# SBATCH -c 4
# SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
module load cuda/11.8


n_eval=5

let "file_idx = SLURM_ARRAY_TASK_ID / n_eval"
let "iter_idx = SLURM_ARRAY_TASK_ID % n_eval"

python /home/gridsan/herol/usmatching_au_change/examples/stl10/main_$1.py --iter $SLURM_ARRAY_TASK_ID
# python /home/gridsan/herol/usmatching_au_change/examples/stl10/main_$1.py experiments/experiment_$file_idx.json --iter $iter_idx
