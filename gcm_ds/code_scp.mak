scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/EC-Earth3-CC/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/EC-Earth3-CC

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/EC-Earth3-Veg/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/EC-Earth3-Veg

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/GFDL-CM4/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/GFDL-CM4

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/pr/TaiESM1/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.6/pr/TaiESM1

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/EC-Earth3-CC/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/EC-Earth3-CC

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/EC-Earth3-Veg/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/EC-Earth3-Veg

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/GFDL-CM4/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/GFDL-CM4

scp -r /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/pr/TaiESM1/y_* haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/0.7/pr/TaiESM1

var="tmax"
version="0.6"
rsync -avz --progress \
  --include="TaiESM1/y_*" \
  --include="*/" \
  --exclude="*" \
  --rsync-path="mkdir -p /lustre/orion/proj-shared/cli138/7hn/BC/$version/$var && rsync" \
  /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/$version/$var/ \
  haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/$version/$var/


var="pr"
version="0.8"
rsync -avz --progress \
  --include="ACCESS-CM2/y_*" \
  --include="BCC-CSM2-MR/y_*" \
  --include="CNRM-ESM2-1/y_*" \
  --include="MPI-ESM1-2-HR/y_*" \
  --include="MRI-ESM2-0/y_*" \
  --include="NorESM2-MM/y_*" \
  --include="*/" \
  --exclude="*" \
  --rsync-path="mkdir -p /lustre/orion/proj-shared/cli138/7hn/BC/$version/$var && rsync" \
  /mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/$version/$var/ \
  haoranniu@andes.olcf.ornl.gov:/lustre/orion/proj-shared/cli138/7hn/BC/$version/$var/