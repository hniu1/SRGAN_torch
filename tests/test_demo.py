"""Exercise the relocated single-stage CLI with real-format NetCDF inputs."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from netCDF4 import Dataset

from test_pipeline import create_synthetic_stage2
from refine_downscaling.model import REFINE, REFINEConfig
from refine_downscaling.stage2_data import Stage2FullFieldDataset
import pipeline_04_infer


class DemoTests(unittest.TestCase):
    def test_relocated_daymet_inference_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'original'
            original.mkdir()
            prepared = create_synthetic_stage2(original)
            manifest = json.loads((prepared / 'manifest.json').read_text())
            self.assertFalse(Path(manifest['source']['data_root']).is_absolute())
            relocated = root / 'relocated'
            shutil.move(original, relocated)
            prepared = relocated / 'prepared'
            self.assertEqual(len(Stage2FullFieldDataset(prepared, 'test')), 2)
            config = REFINEConfig(scale_factor=6, embed_dim=12, num_heads=3,
                                  num_groups=1, blocks_per_group=1, window_size=4)
            model = REFINE(config)
            checkpoint = relocated / 'model.pt'
            torch.save({'model_config':config.to_dict(), 'model':model.state_dict(),
                        'data_manifest':manifest}, checkpoint)
            output = relocated / 'result.nc'
            argv = ['infer', '--data-dir', str(prepared), '--checkpoint', str(checkpoint),
                    '--output', str(output), '--start-date', '2002-01-01', '--end-index', '1',
                    '--enforce-temperature-order']
            for name in ('tmin', 'tmax', 'prcp'):
                path = relocated / 'source' / f'Daymet_ERA5_{name}_dy_2002_0p25deg.nc'
                with Dataset(path, 'a') as nc:
                    variable = nc[f'{name}_dy']
                    if name != 'prcp':
                        variable[:] = variable[:] - 273.15
                        variable.units = 'C'
                    else:
                        variable[:] = np.maximum(variable[:], 0)
                argv += ['--input', f'{name}={path}']
            with patch('sys.argv', argv), patch('torch.cuda.is_available', return_value=False):
                pipeline_04_infer.main()
                with self.assertRaises(FileExistsError):
                    pipeline_04_infer.main()
            with Dataset(output) as nc:
                self.assertEqual(nc['tmin'].shape, (1, 24, 30))
                self.assertTrue(np.isfinite(nc['prcp'][:]).all())
                self.assertTrue(np.all(nc['tmin'][:] <= nc['tmax'][:]))
                self.assertEqual(nc['time'].units, 'days since 2002-01-01')
            self.assertEqual(json.loads(output.with_suffix('.nc.json').read_text())['output_timesteps'], 1)
