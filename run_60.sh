#!/bin/bash
#SBATCH -A cli138
#SBATCH -p gpu
#SBATCH -J SRGAN_60
#SBATCH -N 1
#SBATCH --ntasks-per-node=4          # 4 tasks per node, one for each GPU
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 4
#SBATCH -t 48:00:00
#SBATCH -o ./slurm_output/slurm_60-output.txt
#SBATCH -e ./slurm_output/slurm_60-error.txt

module load cuda/11.0.2
conda init bash
source ~/.bashrc
conda activate srgan

python -u train_60.py

