# AI-Based Climate Downscaling with Super-Resolution Models
This repository provides the reference implementation for AI-based statistical downscaling of climate model outputs using convolutional neural networks and generative adversarial networks. The code supports multi-step spatial super-resolution of coarse-resolution Global Climate Model (GCM) data to high-resolution gridded products suitable for climate analysis and impact studies.

Author:
 - Haoran Niu, CSED, ORNL
 - Deeksha Rastogi, CSED, ORNL

## Overview

The workflow implements super-resolution downscaling of climate variables using:

- SRCNN-style convolutional models

- SRGAN-based adversarial super-resolution models

- PyTorch-based training and inference pipelines

- Post-processing and evaluation utilities

The framework is designed to downscale GCM outputs through intermediate resolutions (e.g., ~100 km → 25 km → 4 km), trained using high-resolution observational datasets.

## Repo Structure
```text
.
├── .vscode/                # Editor configuration (optional)
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── environment.yml         # Conda environment specification
├── train.py                # Model training entry point
├── gcm_downscaling.py      # GCM data downscaling and inference
├── srgan_torch.py          # SRGAN generator and discriminator
├── loss_torch.py           # Loss functions (content, adversarial, etc.)
├── scalers.py              # Data normalization and scaling utilities
├── postproc.py             # Post-processing utilities
└── plot_diff.py            # Visualization and diagnostic plotting
```

## Environment Setup

We recommend using Conda to reproduce the software environment.

```bash
conda env create -f environment.yml
conda activate <env_name>
```

## Model Training

Model training is handled through the main training script:

```bash
python train_*.py

```
## Downscaling and Inference

Once trained models are available, GCM data can be downscaled using:

```bash
python gcm_downscaling.py
```

This script performs spatial super-resolution on coarse-resolution climate inputs and produces high-resolution outputs suitable for downstream analysis.

## Post-Processing and Visualization
- postproc.py: Applies post-processing steps such as masking, rescaling, and formatting.

- plot_diff.py: Generates diagnostic plots comparing downscaled outputs with reference data or baseline methods.

These utilities are intended for evaluation, validation, and figure generation.


## BibTeX Citation

If you use this code in your research, please cite our paper:


```bibtex
@article{rastogi2025complementing,
  title={Complementing dynamical downscaling with super-resolution convolutional neural networks},
  author={Rastogi, Deeksha and Niu, Haoran and Passarella, Linsey and Mahajan, Salil and Kao, Shih-Chieh and Vahmani, Pouya and Jones, Andrew D},
  journal={Geophysical Research Letters},
  volume={52},
  number={4},
  pages={e2024GL111828},
  year={2025},
  publisher={Wiley Online Library}
}

@techreport{rastogi2025artificial,
  title={Artificial Intelligence-Enhanced CMIP6 Climate Projections Across the Conterminous United States},
  author={Rastogi, Deeksha and Niu, Haoran and Kao, Shih-Chieh},
  year={2025},
  institution={Oak Ridge National Laboratory (ORNL), Oak Ridge, TN (United States). Oak~…}
}
```
