#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-00:05:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB

# set up job
module load python/3.9.12 cuda/11.6.2
source ../../env/bin/activate
module load python/3.9.12
# pushd /home/kpyu/great-lakes-tutorial

# run job
sh evaluate_caption_base.sh  # inference & evaluate
