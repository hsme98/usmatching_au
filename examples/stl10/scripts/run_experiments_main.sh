#!/bin/bash

#SBATCH -o train_teacher.log-%j-%a
#SBATCH -a 1-50  # Adjust this to match the number of files
#SBATCH -c 4
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
module load cuda/11.8
source activate distil_env6

SHARED=""
TRANSFORMS="/home/gridsan/herol/usmatching_au/examples/stl10/transforms/transforms.py"

python /home/gridsan/herol/usmatching_au/examples/stl10/main.py --result_folder /home/gridsan/herol/usmatching_au/experiments  --iter $SLURM_ARRAY_TASK_ID > training_$SLURM_ARRAY_TASK_ID