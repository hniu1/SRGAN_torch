#!/usr/bin/env python3
"""Build a complete relocatable demo archive, excluding old runs and repository internals."""
import argparse
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/refine-frontier-demo.tar')
    args = parser.parse_args()
    for asset in json.loads((ROOT / 'daymet/asset-manifest.json').read_text())['assets']:
        path = ROOT / asset['path']
        if not path.is_file() or path.stat().st_size != asset['bytes']:
            raise ValueError(f'Missing or incomplete asset: {path}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    roots = ['README.md', '.gitignore', 'requirements.txt', 'requirements-lock.txt',
             'submit_pipeline.sh', 'refine_downscaling', 'scripts', 'slurm', 'skills',
             'skill-authoring-kit', 'tests', 'assets', 'docs', 'daymet', 'checkpoints']
    roots += [p.name for p in ROOT.glob('pipeline_*.py')]
    roots += [p.name for p in ROOT.glob('utility_*.py')]
    def clean(info):
        if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc', '.copying')):
            return None
        return info
    with tarfile.open(args.output, 'x') as archive:
        for name in roots:
            archive.add(ROOT / name, arcname=f'refine-frontier-demo/{name}', filter=clean)
    print(f'Created {args.output} ({args.output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
