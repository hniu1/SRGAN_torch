#!/bin/bash
#SBATCH -A cli138
#SBATCH -J data_tmin
#SBATCH -o logs/data_tmin-%j.out
#SBATCH -e logs/data_tmin-%j.err
#SBATCH -p batch
#SBATCH -N 1               # for 16 GPUs total
#SBATCH -t 01:00:00


module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm


srun \
     python -u pipeline_01_prepare_daymet.py \
    --version dy_0.8 \
    --var tmin \
    --year-start 1980 \
    --year-end 2020 \
    --high-deg
