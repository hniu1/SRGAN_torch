# for dir in /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/tmax/*; do
#   scp "$dir/y_*.npy" haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/tmax/"$(basename "$dir")"/
# done

# "ACCESS-CM2","BCC-CSM2-MR","CNRM-ESM2-1","MPI-ESM1-2-HR","MRI-ESM2-0","NorESM2-MM"

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/BCC-CSM2-MR/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/BCC-CSM2-MR
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/CNRM-ESM2-1/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/CNRM-ESM2-1
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/MPI-ESM1-2-HR/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/MPI-ESM1-2-HR
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/MRI-ESM2-0/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/MRI-ESM2-0
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/NorESM2-MM/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/NorESM2-MM

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/BCC-CSM2-MR/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/BCC-CSM2-MR
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/CNRM-ESM2-1/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/CNRM-ESM2-1
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/MPI-ESM1-2-HR/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/MPI-ESM1-2-HR
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/MRI-ESM2-0/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/MRI-ESM2-0
scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/NorESM2-MM/y_*.npy haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/NorESM2-MM


rsync -av --include='*/' --include='y_*' --exclude='*' --relative /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/tmax/./*/ haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/tmax/
rsync -av --include='*/' --include='y_*' --exclude='*' --relative /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/tmin/./*/ haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/tmin/

rsync -av --include='*/' --include='y_*' --exclude='*' --relative /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/tmax/./*/ haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/tmax/
rsync -av --include='*/' --include='y_*' --exclude='*' --relative /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/tmin/./*/ haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/tmin/

rsync -av --include='*/' --include='y_*' --exclude='*' --relative /mnt/data/home/7hn/project/SRGAN_torch/models/2.6/ haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/

rsync -avz /mnt/data/home/7hn/project/SRGAN_torch/output/v2.6/* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/SRGAN_torch/output/v2.6/

rsync -avz /mnt/data/home/7hn/project/SRGAN_torch/output/v5.4/* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/SRGAN_torch/output/v5.4/

rsync -avz /mnt/data/home/7hn/project/SRGAN_torch/output/v7.7/* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/SRGAN_torch/output/v7.7/

rsync -avz /mnt/data/home/7hn/project/SRGAN_torch/output/v7.6/* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/SRGAN_torch/output/v7.6/

rsync -avz /mnt/data/home/7hn/project/SRGAN_torch/output/v8.5/* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/SRGAN_torch/output/v8.5/

rsync -avz /mnt/data/home/7hn/project/SRGAN_torch/output/v8.6/* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/SRGAN_torch/output/v8.6/

rsync -avz /mnt/data/home/7hn/project/SRGAN_torch/output/* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/SRGAN_torch/output/

# prcp: version 2.6 and 5.4 downscale 40 year gcm data
# tmax: version 7.7 and 7.6 downscale 40 year gcm data
# tmin: version 8.5 and 8.6 downscale 40 year gcm data