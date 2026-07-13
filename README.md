# SRGAN Patch Training

This branch trains daily SRGAN downscaling models with spatial patches instead
of full-domain fields. The goal is to make the North America workflow more
scalable while keeping the same two-stage downscaling structure:

| Stage | Mapping | LR patch | HR patch | Scale |
|---|---|---:|---:|---:|
| Stage 1 | 1 deg -> 0.25 deg | `8 x 8` | `32 x 32` | `4x` |
| Stage 2 | 0.25 deg -> 0.0416 deg | `32 x 32` | `192 x 192` | `6x` |

Patch training is lazy: cached full-field arrays stay on disk, and paired
LR/HR crops are sampled inside the PyTorch dataset during training. The code
does not pre-write every sliding-window patch to disk.

## Main Files

- `prepare_daymet.py`: prepares cached Daymet/ERA5 arrays and scalers.
- `patch_dataset.py`: samples paired LR/HR patches from cached arrays.
- `train_daily_frontier_srun.py`: Stage 1 training, 1 deg -> 0.25 deg.
- `train_daily_frontier_srun_step2.py`: Stage 2 training, 0.25 deg -> 0.0416 deg.
- `srgan_torch.py`: generator and discriminator definitions.
- `loss_torch.py`: SRGAN loss wrappers.
- `scalers.py`: custom scaler utilities.
- `srgan_srun.sh`: Frontier SLURM launcher template.
- `gcm_downscaling.py`: applies trained models for GCM downscaling.

## 1. Prepare Data

Data preparation writes cached arrays under `output/<version>/` and a scaler
under `models/<version>/`.

For Stage 1, omit `--high-deg`:

```bash
export SRGAN_BASE_DIR=/lustre/orion/proj-shared/cli138/7hn/SRGAN_patch_training

python -u prepare_daymet.py \
  --base-dir ${SRGAN_BASE_DIR} \
  --version <stage1_version> \
  --var tmax \
  --year-start 1980 \
  --year-end 2020
```

For Stage 2, include `--high-deg`:

```bash
python -u prepare_daymet.py \
  --base-dir ${SRGAN_BASE_DIR} \
  --version <stage2_version> \
  --var tmax \
  --year-start 1980 \
  --year-end 2020 \
  --high-deg
```

SLURM templates:

```bash
sbatch prepare_daymet_srun_frontier.sh
sbatch prepare_daymet_srun_andes.sh
```

## 2. Train With Patches

Both training scripts support:

```text
--patch-training
--lr-patch-size
--scale-factor
--patches-per-image
```

`--patches-per-image` controls how many random spatial crops are drawn from
each timestep per epoch.

### Stage 1

```bash
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
```

### Stage 2

```bash
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

Train Stage 1 and Stage 2 as separate models. They use different scale factors,
different model architectures, and different physical grid spacings.

## 3. Evaluate Or Downscale

Evaluation mode still uses the trained generator checkpoint from
`models/<version>/`.

```bash
python3 -u train_daily_frontier_srun_step2.py \
  --master_addr <MASTER_ADDR> \
  --master_port 3442 \
  --mode eval \
  --batch-size 4 \
  --version <stage2_version> \
  --base-dir ${SRGAN_BASE_DIR} \
  --var tmax
```

GCM downscaling uses:

```bash
python -u gcm_downscaling.py
sbatch gcm_downscaling_srun.sh
```

Typical GCM outputs:

- `y_gcm_100.npy`
- `y_pred_25.npy`
- `y_pred_4.npy`

## Notes

- Use `SRGAN_BASE_DIR` to point scripts at this worktree.
- Keep `--version` consistent between data preparation, training, and evaluation.
- Common variables are `tmax`, `tmin`, and precipitation (`prcp` in data-prep scripts).
- Full-domain inference may still work because the generators are convolutional.
  If memory becomes limiting, add tiled inference with overlap/halo cropping.
