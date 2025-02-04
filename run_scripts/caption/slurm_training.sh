#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-00:15:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=47GB

# set up job
module load python/3.9.12 cuda/11.6.2
source ../../env/bin/activate
module load python/3.9.12
# pushd /home/kpyu/great-lakes-tutorial

# run job
sh train_caption_stage1_base.sh # inference & evaluate
