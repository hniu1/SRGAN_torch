#!/usr/bin/env python3
"""Check local assets, Daymet inputs and pretrained checkpoint; no scheduler actions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-all', action='store_true', help='Read and checksum all bundled binary assets')
    parser.add_argument('--model-smoke', action='store_true', help='Run the real checkpoint on one small Daymet patch on CPU')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'daymet/asset-manifest.json').read_text())['assets']
    for asset in assets:
        path = ROOT / asset['path']
        if not path.is_file() or path.stat().st_size != asset['bytes']:
            raise ValueError(f'Missing or wrong-sized asset: {path}; run scripts/restore_demo.py')
        if args.hash_all or path.suffix == '.pt':
            if digest(path) != asset['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
    command = [sys.executable, str(ROOT / 'skills/refine-downscaling/scripts/validate_inputs.py'), '--strict', '--full-scan']
    for name in ('tmin', 'tmax', 'prcp'):
        command += ['--input', f'{name}={ROOT}/daymet/data/Daymet_ERA5_{name}_dy_1990_0p25deg.nc']
    subprocess.run(command, check=True)
    import numpy as np
    from netCDF4 import Dataset
    # Match actual LR coordinates and time values across all three variables.
    with Dataset(ROOT / 'daymet/data/Daymet_ERA5_tmin_dy_1990_0p25deg.nc') as ref:
        for name in ('tmax', 'prcp'):
            with Dataset(ROOT / f'daymet/data/Daymet_ERA5_{name}_dy_1990_0p25deg.nc') as other:
                for key in ('lat', 'lon', 'time'):
                    if key in ref.variables:
                        np.testing.assert_array_equal(ref[key][:], other[key][:])
                        for attr in ('units', 'calendar'):
                            if getattr(ref[key], attr, None) != getattr(other[key], attr, None):
                                raise ValueError(f'{name}: {key} {attr} differ')
    from refine_downscaling.stage2_data import Stage2FullFieldDataset
    dataset = Stage2FullFieldDataset(ROOT / 'daymet/prepared', 'test')
    if len(dataset) != 365 or dataset.scale_factor != 6:
        raise ValueError('Demo must contain the 365-day 1990 test split at 6x')
    import torch
    from refine_downscaling.model import REFINE, REFINEConfig
    checkpoint = torch.load(ROOT / 'checkpoints/refine_6x.pt', map_location='cpu', weights_only=False)
    config = REFINEConfig.from_dict(checkpoint['model_config'])
    if config.scale_factor != 6 or tuple(config.variable_names) != ('tmin', 'tmax', 'prcp'):
        raise ValueError('Checkpoint model configuration mismatch')
    if checkpoint['data_manifest']['transforms'] != dataset.manifest['transforms']:
        raise ValueError('Checkpoint normalization mismatch')
    if args.model_smoke:
        from refine_downscaling.stage2_data import Stage2NetCDFPatchDataset
        torch.set_num_threads(2)
        patches = Stage2NetCDFPatchDataset(ROOT / 'daymet/prepared', 'test', core_size=8, halo=2, patches_per_day=1, random_patches=False)
        sample = patches[0]
        model = REFINE(config).eval()
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            output = model(*(sample[key][None] for key in ('lr', 'static_lr', 'static_hr', 'season')))
        if tuple(output.shape) != (1, 3, 72, 72) or not torch.isfinite(output).all():
            raise ValueError('Pretrained patch smoke failed')
        print('Pretrained Daymet patch smoke passed:', tuple(output.shape))
    print('Demo assets and inputs verified. Full-domain GPU inference requires a Frontier allocation.')


if __name__ == '__main__':
    main()
