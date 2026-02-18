srun -A cli138 \
     -p batch \
     -q debug \
     -N 1 \
     -n 2 \
     --gpus-per-task=1 \
     -c 7 \
     -t 02:00:00 \
     --pty bash