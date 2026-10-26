"""Restore immutable audit input caches from Git-friendly gzip copies.

Existing files must match their recorded SHA256 and are never overwritten.
Use --verify-only to verify archive contents without creating any input file.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT / 'results/nurse_forecasting/root_cause_analysis/inputs'

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ARCHIVES / 'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest:
        archive = ARCHIVES / entry['archive']
        target = ROOT / entry['original_path']
        if sha256_file(archive) != entry['archive_sha256']:
            raise ValueError(f'Archive hash mismatch: {archive}')
        h = hashlib.sha256()
        with gzip.open(archive, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != entry['original_sha256']:
            raise ValueError(f'Decompressed hash mismatch: {archive}')
        if args.verify_only:
            print(f'Verified: {entry["original_path"]}')
        elif target.exists():
            if sha256_file(target) != entry['original_sha256']:
                raise ValueError(f'Existing file differs; refusing to overwrite: {target}')
            print(f'Already present, unchanged: {entry["original_path"]}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(archive, 'rb') as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            if sha256_file(target) != entry['original_sha256']:
                raise ValueError(f'Restored file hash mismatch: {target}')
            print(f'Restored: {entry["original_path"]}')

if __name__ == '__main__':
    main()
