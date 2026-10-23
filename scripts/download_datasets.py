"""Download official archives only; no mirror fallback or generated replacement data.

Run from project root: py scripts/download_datasets.py --dataset wesad --extract
The WESAD non-commercial research condition and PhysioNet ODC-By terms apply.
Self-computed SHA256 proves local file identity, not publisher authentication.
"""
import argparse
import hashlib
import json
import shutil
import stat
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

PAGES = {
    'wesad': 'https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html',
    'physionet': 'https://physionet.org/content/wearable-device-dataset/1.0.1/',
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def official_archive(dataset):
    page = PAGES[dataset]
    with urllib.request.urlopen(page, timeout=60) as response:
        html = response.read().decode('utf-8')
    parser = Links(); parser.feed(html)
    if dataset == 'wesad':
        links = [urllib.parse.urljoin(page, h) for h in parser.links
                 if 'uni-siegen.sciebo.de/' in h]
    else:
        links = [urllib.parse.urljoin(page, h) for h in parser.links
                 if h.lower().endswith('.zip') and 'wearable-device-dataset' in h]
    links = list(dict.fromkeys(links))
    if len(links) != 1:
        raise RuntimeError(f'Expected one official archive link; found {len(links)}. Inspect {page}')
    return links[0]


def safe_extract(archive, target):
    root = target.resolve()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            output = (target / entry.filename).resolve()
            if not output.is_relative_to(root):
                raise ValueError('Unsafe ZIP path')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('ZIP symbolic links are not accepted')
        if z.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        z.extractall(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=PAGES, required=True)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    raw = Path(args.raw_dir); raw.mkdir(parents=True, exist_ok=True)
    logs = Path('results/logs'); logs.mkdir(parents=True, exist_ok=True)
    item = {'utc': datetime.now(timezone.utc).isoformat(), 'dataset': args.dataset,
            'source_page': PAGES[args.dataset], 'status': 'started'}
    archive = raw / ('WESAD.zip' if args.dataset == 'wesad' else 'PhysioNet_StressExercise_1.0.1.zip')
    partial = archive.with_suffix('.zip.partial')
    started = time.perf_counter()
    try:
        if archive.exists():
            raise FileExistsError(f'{archive} exists; preserve it and audit before downloading again.')
        if shutil.disk_usage(raw).free < 12 * 1024**3 and args.dataset == 'wesad':
            raise RuntimeError('Keep at least 12 GiB free for the WESAD archive and extraction.')
        url = official_archive(args.dataset); item['download_url'] = url
        digest = hashlib.sha256(); count = 0; last_report = 0
        with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as output:
            item['resolved_url'] = response.url
            expected = response.headers.get('Content-Length')
            item['content_length'] = int(expected) if expected else None
            while True:
                block = response.read(2**20)
                if not block:
                    break
                output.write(block); digest.update(block); count += len(block)
                if count - last_report >= 100 * 2**20:
                    print(f'{args.dataset}: {count / 2**20:.0f} MiB', flush=True); last_report = count
        if expected and count != int(expected):
            raise IOError('Content-Length mismatch; partial file retained')
        if not zipfile.is_zipfile(partial):
            raise ValueError('Response is not a ZIP archive; do not treat it as data')
        partial.replace(archive)
        item.update(status='downloaded', bytes=count, sha256=digest.hexdigest(), file=str(archive))
        if args.extract:
            safe_extract(archive, raw)
            item['status'] = 'downloaded_and_extracted'
        print(json.dumps(item, indent=2))
    except Exception as exc:
        item.update(status='blocked_or_failed', error=f'{type(exc).__name__}: {exc}')
        print(json.dumps(item, indent=2))
        raise
    finally:
        item['elapsed_s'] = time.perf_counter() - started
        with (logs / 'download_attempts.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
