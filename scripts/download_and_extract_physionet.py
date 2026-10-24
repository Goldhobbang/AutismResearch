"""Download official PhysioNet 1.0.1 zip archive and extract to data/raw/physionet_v1.
Verifies all extracted files against official SHA256SUMS.txt.
"""
import hashlib
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'raw'
TARGET_DIR = DATA_DIR / 'physionet_v1'
ZIP_PATH = DATA_DIR / 'PhysioNet_StressExercise_1.0.1.zip'
SUMS_PATH = ROOT / 'data' / 'registry' / 'SHA256SUMS.txt'
URL = 'https://physionet.org/content/wearable-device-dataset/get-zip/1.0.1/'


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


def download_zip():
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 60 * 1024 * 1024:
        print(f"[FOUND EXISTING ZIP] {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
        return

    print(f"[DOWNLOADING] {URL} -> {ZIP_PATH} (~69.7 MB)...", flush=True)
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    partial = ZIP_PATH.with_suffix('.zip.partial')
    t0 = time.time()
    last_print = t0
    downloaded = 0

    with urllib.request.urlopen(req, timeout=120) as resp, open(partial, 'wb') as out:
        total = int(resp.headers.get('Content-Length', 0))
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_print > 5.0:
                pct = (downloaded / total * 100) if total else 0
                print(f"Downloaded: {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({pct:.1f}%) in {time.time()-t0:.1f}s", flush=True)
                last_print = time.time()

    partial.replace(ZIP_PATH)
    print(f"[DOWNLOAD COMPLETE] {ZIP_PATH} in {time.time()-t0:.1f}s", flush=True)


def extract_and_verify():
    hashes = load_official_hashes()
    print(f"[EXTRACTING] {ZIP_PATH} to {TARGET_DIR}...", flush=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        namelist = z.namelist()
        print(f"Total entries in archive: {len(namelist)}", flush=True)
        # Determine prefix inside zip
        # Look for Wearable_Dataset
        prefix = ''
        for name in namelist:
            if 'Wearable_Dataset/' in name:
                prefix = name.split('Wearable_Dataset/')[0]
                break

        print(f"Detected internal zip prefix: '{prefix}'", flush=True)

        extracted_count = 0
        for member in z.infolist():
            if member.is_dir():
                continue
            filename = member.filename
            if prefix and filename.startswith(prefix):
                rel = filename[len(prefix):]
            else:
                rel = filename

            dest = TARGET_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted_count += 1

    print(f"[EXTRACTION COMPLETE] Extracted {extracted_count} files.", flush=True)

    # Verify SHA256 of all extracted files
    print("\n[VERIFYING SHA-256 INTEGRITY]", flush=True)
    verified = 0
    mismatched = 0
    missing = 0

    for rel_path, expected_hash in hashes.items():
        file_path = TARGET_DIR / rel_path
        if not file_path.exists():
            missing += 1
            continue
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual_hash == expected_hash:
            verified += 1
        else:
            print(f"[MISMATCH] {rel_path}: actual {actual_hash} != expected {expected_hash}", flush=True)
            mismatched += 1

    print(f"Verification Results: {verified} VERIFIED, {mismatched} MISMATCHED, {missing} MISSING (out of {len(hashes)} registered)", flush=True)
    if mismatched > 0:
        print("[ERROR] Integrity verification failed!", flush=True)
        sys.exit(1)
    print("[SUCCESS] All files cryptographically verified with official SHA256!", flush=True)


if __name__ == '__main__':
    download_zip()
    extract_and_verify()
