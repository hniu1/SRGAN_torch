#!/bin/bash
#SBATCH -A cli138
#SBATCH -p gpu
#SBATCH -J SRGAN_72
#SBATCH -N 1
#SBATCH --ntasks-per-node=4          # 4 tasks per node, one for each GPU
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 4
#SBATCH -t 48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niuh@ornl.gov
#SBATCH -o ./slurm_output/slurm_72-output_e.txt
#SBATCH -e ./slurm_output/slurm_72-error_e.txt

module load cuda/11.0.2
conda init bash
source ~/.bashrc
conda activate srgan

# python -u train_72.py
python -u train_72.py --mode='eval'

