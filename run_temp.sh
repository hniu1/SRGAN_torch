#!/bin/bash
#SBATCH -A cli138
#SBATCH -p gpu
#SBATCH -J SRGAN_temp
#SBATCH -N 1
#SBATCH --ntasks-per-node=1          # 4 tasks per node, one for each GPU
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 8
#SBATCH -t 48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niuh@ornl.gov
#SBATCH -o ./slurm_output/slurm_03-output.txt
#SBATCH -e ./slurm_output/slurm_03-error.txt

module purge
# module load cuda/11.2.2   # load CUDA only

# activate conda first, then force its libs to the front:
source ~/.bashrc
conda activate srgan
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# python -u train_3h_temp.py
python -u train_3h_prcp.py

