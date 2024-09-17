#!/bin/bash
#SBATCH -A cli138
#SBATCH -p gpu
#SBATCH -J SRGAN_40
#SBATCH -N 1
#SBATCH --ntasks-per-node=4          # 4 tasks per node, one for each GPU
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niuh@ornl.gov
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/slurm-output_40.txt
#SBATCH -e ./slurm_output/slurm-error_40.txt

module load cuda/11.0.2
conda init bash
source ~/.bashrc
conda activate srgan

python -u train_40.py > ./slurm_output/slurm-output_40.txt 2> ./slurm_output/slurm-error_40.txt
# python -u train_40.py --mode='eval'

