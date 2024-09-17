#!/bin/bash
#SBATCH -A cli138
#SBATCH -p gpu
#SBATCH -J SRGAN_50
#SBATCH -N 1
#SBATCH --ntasks-per-node=4          # 4 tasks per node, one for each GPU
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niuh@ornl.gov
#SBATCH -t 48:00:00
#SBATCH -o ./slurm_output/slurm-output_50.txt
#SBATCH -e ./slurm_output/slurm-error_50.txt

module load cuda/11.0.2
conda init bash
source ~/.bashrc
conda activate srgan

# srun python -u train_41.py
python -u train_50.py
# python -u train_50.py --mode='eval'

