# SRGAN Batch Loading

This repository contains scripts to:
- prepare Daymet training data caches, and
- train daily SRGAN downscaling models on HPC systems (Frontier/Andes style workflows), and
- run GCM-to-fine-grid downscaling using trained daily models.

## Main scripts

### Prepare Daymet
- `prepare_daymet.py`: prepares cached arrays and scaler files for a given variable and year range.
- `prepare_daymet_srun_frontier.sh`: Frontier batch launcher example.
- `prepare_daymet_srun_andes.sh`: Andes/local batch launcher example.

### Train Daily
- `train_daily_frontier_srun.py`
- `train_daily_frontier_srun_step2.py`: 0.25° → 0.0416° (second step).
- `srgan_srun.sh`: Frontier SLURM launcher example for daily training.

### GCM Downscaling
- `gcm_downscaling.py`: final 100 km → 25 km → 4 km downscaling workflow.
- `gcm_downscaling_srun.sh`: SLURM launcher for downscaling.
- `gcm_downscling_scale.py`: utility script for inverse-scaling a saved downscaling array.

---

## 1) Prepare Daymet data

### What it creates
For a selected `--version`, the script writes:
- `models/<version>/scaler.pkl`
- `output/<version>/x_train.npy`
- `output/<version>/x_test.npy`
- `output/<version>/y_train.npy`
- `output/<version>/y_test.npy`

### Run directly
```bash
export SRGAN_BASE_DIR=/path/to/SRGAN_batch_loading

python -u prepare_daymet.py \
  --base-dir ${SRGAN_BASE_DIR} \
  --version dy_v0.8 \
  --var tmin \
  --year-start 1980 \
  --year-end 2020 \
  --high-deg
```

### Run with SLURM
Use one of:
- `prepare_daymet_srun_frontier.sh`
- `prepare_daymet_srun_andes.sh`

Submit with:
```bash
sbatch prepare_daymet_srun_frontier.sh
# or
sbatch prepare_daymet_srun_andes.sh
```

---

## 2) Train daily SRGAN

Daily training expects prepared arrays under `output/<version>/` and scaler/model files under `models/<version>/`.

### Run directly (single node / manual launch)
```bash
export SRGAN_BASE_DIR=/path/to/SRGAN_batch_loading

python3 -u train_daily_frontier_srun_step2.py \
  --master_addr <MASTER_ADDR> \
  --master_port 3442 \
  --mode train \
  --batch-size 4 \
  --version dy_v0.8 \
  --base-dir ${SRGAN_BASE_DIR} \
  --var tmax \
  --w1-fn1 1e-5 \
  --w2-fn2 1e3
```

Optional flags:
- `--initial-training` to run generator pretrain phase.
- `--amp` to enable bfloat16 autocast.
- `--mode eval` for evaluation mode.

### Patch Training
Patch training is supported for both daily training scripts with `--patch-training`.
The cached full fields are kept on disk, and random paired LR/HR crops are sampled
inside the PyTorch dataset at training time.

Recommended starting point:
```bash
# Stage 1: 1 deg -> 0.25 deg
python3 -u train_daily_frontier_srun.py \
  --master_addr <MASTER_ADDR> \
  --master_port 3442 \
  --mode train \
  --batch-size 64 \
  --version <stage1_version> \
  --base-dir ${SRGAN_BASE_DIR} \
  --var tmax \
  --patch-training \
  --lr-patch-size 8 \
  --scale-factor 4 \
  --patches-per-image 4

# Stage 2: 0.25 deg -> 0.0416 deg
python3 -u train_daily_frontier_srun_step2.py \
  --master_addr <MASTER_ADDR> \
  --master_port 3442 \
  --mode train \
  --batch-size 4 \
  --version <stage2_version> \
  --base-dir ${SRGAN_BASE_DIR} \
  --var tmax \
  --patch-training \
  --lr-patch-size 32 \
  --scale-factor 6 \
  --patches-per-image 4
```

Train Stage 1 and Stage 2 as separate models. They use different scale factors
and different physical context per grid cell, so separate training keeps each
model focused on its own mapping.

### Run with SLURM
Use `srgan_srun.sh` as the template launcher:
```bash
sbatch srgan_srun.sh
```

---

## 3) GCM downscaling

This stage applies trained daily SRGAN generators to bias-corrected GCM inputs and writes predictions under `gcm_ds/<downscale_version>/<var>/<gcm>/`.

### Main script
Use `gcm_downscaling.py` (this is the current entrypoint used by `gcm_downscaling_srun.sh`).

### Run directly
```bash
python -u gcm_downscaling.py
```

### Run with SLURM
```bash
sbatch gcm_downscaling_srun.sh
```

### Typical outputs
- `y_gcm_100.npy`
- `y_pred_25.npy`
- `y_pred_4.npy` (or `y_pred_4_test.npy` depending on script branch)

### Important
- `gcm_downscaling.py` currently sets variable list, GCM list, scenario, and model versions inside `__main__` (no CLI args yet), so edit those values in the script before submitting jobs.

---

## Notes
- Set `SRGAN_BASE_DIR` to your repo root to avoid editing hardcoded paths.
- Keep `--version` consistent between `prepare_daymet.py` and training scripts.
- Common variables are `tmax`, `tmin`, and precipitation (`prcp` in data-prep scripts).
