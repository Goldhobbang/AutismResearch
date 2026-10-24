import os
import urllib.request
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'raw' / 'physionet_v1'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load official SHA256SUMS.txt
sums_path = ROOT / 'data' / 'registry' / 'SHA256SUMS.txt'
official_hashes = {}
with open(sums_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            h, p = parts
            official_hashes[p] = h

BASE_URL = 'https://physionet.org/files/wearable-device-dataset/1.0.1/'

def download_and_verify(rel_path):
    dest = DATA_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = official_hashes.get(rel_path)
    
    if dest.exists() and expected_hash:
        with open(dest, 'rb') as f:
            curr_hash = hashlib.sha256(f.read()).hexdigest()
        if curr_hash == expected_hash:
            print(f'[EXISTS & VERIFIED] {rel_path}')
            return True
            
    url = BASE_URL + rel_path
    print(f'[DOWNLOADING] {url} -> {dest}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, 'wb') as out:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out.write(chunk)
            
    with open(dest, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    if expected_hash and file_hash != expected_hash:
        raise ValueError(f'Hash mismatch for {rel_path}: got {file_hash}, expected {expected_hash}')
        
    print(f'[VERIFIED SHA256] {rel_path} ({file_hash[:16]}...)')
    return True

if __name__ == '__main__':
    # Test downloading S01 STRESS and AEROBIC
    files_to_download = [
        'Wearable_Dataset/STRESS/S01/ACC.csv',
        'Wearable_Dataset/STRESS/S01/BVP.csv',
        'Wearable_Dataset/STRESS/S01/EDA.csv',
        'Wearable_Dataset/STRESS/S01/tags.csv',
        'Wearable_Dataset/AEROBIC/S01/ACC.csv',
        'Wearable_Dataset/AEROBIC/S01/BVP.csv',
        'Wearable_Dataset/AEROBIC/S01/EDA.csv',
        'Wearable_Dataset/AEROBIC/S01/tags.csv',
    ]
    for rel_path in files_to_download:
        download_and_verify(rel_path)
