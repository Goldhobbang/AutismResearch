"""Package existing audit caches losslessly without changing the originals."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/nurse_forecasting/root_cause_analysis/inputs'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in ['features.csv', 'predictions.csv']:
        source = ROOT / 'results/nurse_forecasting' / name
        archive = OUT / (name + '.gz')
        source_hash = digest(source)
        if not archive.exists():
            with source.open('rb') as raw, archive.open('xb') as compressed:
                with gzip.GzipFile(filename='', mode='wb', compresslevel=6,
                                   fileobj=compressed, mtime=0) as stream:
                    shutil.copyfileobj(raw, stream)
        h = hashlib.sha256()
        with gzip.open(archive, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != source_hash:
            raise ValueError(f'Archive differs from original: {archive}')
        if archive.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f'Archive exceeds GitHub single-file limit: {archive}')
        manifest.append({'original_path': source.relative_to(ROOT).as_posix(),
                         'original_bytes': source.stat().st_size,
                         'original_sha256': source_hash,
                         'archive': archive.name,
                         'archive_bytes': archive.stat().st_size,
                         'archive_sha256': digest(archive)})
        print(f'{name}: {source.stat().st_size} -> {archive.stat().st_size} bytes; SHA256 verified')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
