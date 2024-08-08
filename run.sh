#!/bin/bash
#SBATCH -A cli138
#SBATCH -p gpu
#SBATCH -J SRGAN
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niuh@ornl.gov
#SBATCH -t 12:00:00
#SBATCH -o ./slurm-output.txt
#SBATCH -e ./slurm-error.txt

module load cuda/11.0.2
conda init bash
source ~/.bashrc
conda activate srgan

python -u train.py > ./slurm-output.txt 2> ./slurm-error.txt
