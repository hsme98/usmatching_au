#!/bin/bash

#SBATCH -o train_teacher.log-%j-%a
#SBATCH -a 1-4  # Adjust this to match the number of files
#SBATCH -c 4
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
module load cuda/11.8
source activate distil_env6

SHARED="--shared"
TRANSFORMS="/home/gridsan/herol/usmatching_au/examples/stl10/transforms/transforms.py"
AUG="--aug double"
FOLD="--result_folder /home/gridsan/herol/usmatching_au/experiments_0"

python /home/gridsan/herol/usmatching_au/examples/stl10/main_multi.py $SHARED $AUG --transforms $TRANSFORMS --iter $SLURM_ARRAY_TASK_ID $FOLD $AUG