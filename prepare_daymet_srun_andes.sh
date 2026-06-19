#!/bin/bash
#SBATCH -A cli138
#SBATCH -J data_tmin
#SBATCH -o logs/data_tmin-%j.out
#SBATCH -e logs/data_tmin-%j.err
#SBATCH -p batch
#SBATCH -N 1               # for 16 GPUs total
#SBATCH -t 2:00:00
#SBATCH --mem=180G


conda init bash
source ~/.bashrc
conda activate srgan

python -u prepare_daymet.py \
    --version dy_v0.8 \
    --var tmin \
    --year-start 1980 \
    --year-end 2020 \
    --high-deg