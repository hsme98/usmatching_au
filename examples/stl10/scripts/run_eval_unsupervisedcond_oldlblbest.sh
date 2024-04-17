#!/bin/bash

# SBATCH -o logs/eval_unsupervisedcond.log-%j-%a
# SBATCH -a 0-44  # Adjust this to match the number of files n_eval x tot_files - 1
# SBATCH -c 20
# SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
n_eval=5

let "file_idx = SLURM_ARRAY_TASK_ID / n_eval"
let "iter_idx = SLURM_ARRAY_TASK_ID % n_eval"

echo $file_idx
echo $iter_idx

python /home/gridsan/herol/usmatching_au_new_change/examples/stl10/main_linear_eval.py /home/gridsan/herol/usmatching_au_new_change/examples/stl10/results/cifar100_series_unsupervisedcond_newlbls_${file_idx}_200/encoder.pth --iter $iter_idx --num_workers 10

python /home/gridsan/herol/usmatching_au_new_change/examples/stl10/main_linear_eval.py /home/gridsan/herol/usmatching_au_new_change/examples/stl10/results/cifar100_series_unsupervisedcond_newlbls_${file_idx}_200/encoder_best.pth --iter $iter_idx --num_workers 10


