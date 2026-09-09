#!/usr/bin/env python3
"""Restore ignored demo assets from recorded Frontier sources without overwriting files."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    for asset in json.loads((ROOT / 'daymet/asset-manifest.json').read_text())['assets']:
        target = ROOT / asset['path']
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.copying')
            print('Copying', asset['path'], flush=True)
            shutil.copyfile(asset['source'], temporary)
            with temporary.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != asset['sha256']:
                    raise ValueError(f'Checksum mismatch: {temporary}')
            temporary.rename(target)
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != asset['sha256']:
                raise ValueError(f'Existing asset differs; preserved: {target}')
    print('Assets restored and checksummed.')


if __name__ == '__main__':
    main()
