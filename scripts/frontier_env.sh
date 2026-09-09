#!/bin/bash
# Source this script on Frontier.
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
conda activate "${REFINE_ENV:-/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm}"
