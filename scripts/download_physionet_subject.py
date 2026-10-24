"""Download and verify raw PhysioNet 1.0.1 files with official SHA256 checksums.
Usage:
    py scripts/download_physionet_subject.py --subject S01
    py scripts/download_physionet_subject.py --subject S02 S03 S04
    py scripts/download_physionet_subject.py --all-verified
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'raw' / 'physionet_v1'
SUMS_PATH = ROOT / 'data' / 'registry' / 'SHA256SUMS.txt'
SEGMENTS_PATH = ROOT / 'data' / 'processed' / 'physionet' / 'verified_segments.csv'
BASE_URLS = [
    'https://physionet-open.s3.amazonaws.com/wearable-device-dataset/1.0.1/',
    'https://physionet.org/files/wearable-device-dataset/1.0.1/',
]


def load_official_hashes(path=SUMS_PATH):
    hashes = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                h, p = parts
                hashes[p.strip().lstrip('*')] = h.lower()
    return hashes


def download_and_verify_file(rel_path, official_hashes, max_retries=3):
    dest = DATA_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = official_hashes.get(rel_path)

    if dest.exists() and expected_hash:
        with open(dest, 'rb') as f:
            curr_hash = hashlib.sha256(f.read()).hexdigest()
        if curr_hash == expected_hash:
            print(f'[EXISTS & VERIFIED] {rel_path}', flush=True)
            return True
        else:
            print(f'[CORRUPT/MISMATCH] {rel_path} (local {curr_hash[:8]} != {expected_hash[:8]}), re-downloading...', flush=True)
            dest.unlink(missing_ok=True)

    part_path = dest.with_suffix(dest.suffix + '.part')

    for attempt in range(1, max_retries + 1):
        base_url = BASE_URLS[(attempt - 1) % len(BASE_URLS)]
        url = base_url + rel_path
        try:
            existing_bytes = part_path.stat().st_size if part_path.exists() else 0
            headers = {'User-Agent': 'Mozilla/5.0'}
            if existing_bytes > 0:
                headers['Range'] = f'bytes={existing_bytes}-'

            req = urllib.request.Request(url, headers=headers)
            print(f'[DOWNLOADING] {url} -> {dest} (attempt {attempt}/{max_retries}, resume={existing_bytes})', flush=True)
            with urllib.request.urlopen(req, timeout=90) as resp:
                status = getattr(resp, 'status', 200)
                mode = 'ab' if (existing_bytes > 0 and status == 206) else 'wb'
                with open(part_path, mode) as out:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)

            with open(part_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            if expected_hash and file_hash != expected_hash:
                part_path.unlink(missing_ok=True)
                raise ValueError(f'Hash mismatch for {rel_path}: got {file_hash}, expected {expected_hash}')

            part_path.replace(dest)
            print(f'[VERIFIED SHA256] {rel_path} ({file_hash[:16]}...)', flush=True)
            return True
        except Exception as e:
            print(f'[ERROR] attempt {attempt} failed for {rel_path}: {e}', flush=True)
            if attempt == max_retries:
                raise


def get_verified_subjects():
    if not SEGMENTS_PATH.exists():
        return [f'S{i:02d}' for i in range(1, 19)]
    df = pd.read_csv(SEGMENTS_PATH)
    v = df[df['verified'].astype(str).str.lower().isin(['true', '1'])]
    return sorted(v['subject'].unique().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', nargs='+', help='Subject ID(s), e.g. S01 S02')
    parser.add_argument('--all-verified', action='store_true', help='Download all verified subjects (S01-S18)')
    parser.add_argument('--dry-run', action='store_true', help='Only list files to download and size')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel download threads')
    args = parser.parse_args()

    hashes = load_official_hashes()

    if args.all_verified:
        target_subs = get_verified_subjects()
    elif args.subject:
        target_subs = args.subject
    else:
        target_subs = ['S01']

    print(f"Target subjects: {target_subs}", flush=True)

    files = []
    for sub in target_subs:
        for session in ['STRESS', 'AEROBIC']:
            for name in ['ACC.csv', 'BVP.csv', 'EDA.csv', 'TEMP.csv', 'HR.csv', 'IBI.csv', 'tags.csv']:
                rel_path = f"Wearable_Dataset/{session}/{sub}/{name}"
                if rel_path in hashes:
                    files.append(rel_path)

    print(f"Total files registered: {len(files)}", flush=True)

    if args.dry_run:
        missing = 0
        corrupt = 0
        verified = 0
        for p in files:
            dest = DATA_DIR / p
            exp = hashes.get(p)
            if not dest.exists():
                missing += 1
            else:
                with open(dest, 'rb') as f:
                    h = hashlib.sha256(f.read()).hexdigest()
                if h == exp:
                    verified += 1
                else:
                    corrupt += 1
        print(f"Status summary: Verified={verified}, Corrupt={corrupt}, Missing={missing}", flush=True)
        return

    print(f"Starting download with {args.workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_and_verify_file, p, hashes): p for p in files}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[FATAL] Failed to download {p}: {e}", flush=True)
                sys.exit(1)

    print("\nAll target files successfully downloaded and verified with official SHA256!", flush=True)


if __name__ == '__main__':
    main()
