import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.core import basic, spectrum

def parse_header_time(val):
    val = val.strip()
    try:
        return float(val)
    except ValueError:
        pass
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'):
        try:
            return datetime.strptime(val, fmt).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse start time: {val}")

def load_e4_csv(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        line0 = next(r)
        line1 = next(r)
        start_ts = parse_header_time(line0[0])
        fs = float(line1[0])
        data = []
        for row in r:
            if not row:
                continue
            data.append([float(x) for x in row])
    arr = np.array(data, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return start_ts, fs, arr

def compute_physionet_window_features(eda_chunk, bvp_chunk, acc_chunk):
    # eda_chunk: 4Hz, 240 samples for 60s
    # bvp_chunk: 64Hz, 3840 samples for 60s
    # acc_chunk: 32Hz, (1920, 3) samples for 60s
    out = {}
    
    # EDA
    out.update(basic(eda_chunk, 4.0, 'eda_'))
    delta_eda = np.diff(eda_chunk)
    out['eda_positive_diff_sum'] = float(np.maximum(delta_eda[np.isfinite(delta_eda)], 0).sum())
    from scipy.signal import find_peaks
    out['eda_scr_candidate_count'] = float(len(find_peaks(eda_chunk, prominence=0.05, distance=4)[0])) if np.isfinite(eda_chunk).all() else np.nan
    
    # PPG / BVP
    out.update(basic(bvp_chunk, 64.0, 'ppg_'))
    out['ppg_peak_hz'], out['ppg_band_fraction'] = spectrum(bvp_chunk, 64.0, 0.7, 3.5)
    
    # ACC
    mag = np.linalg.norm(acc_chunk, axis=1)
    out.update(basic(mag, 32.0, 'acc_mag_'))
    for i, axis in enumerate('xyz'):
        out.update(basic(acc_chunk[:, i], 32.0, f'acc_{axis}_'))
        
    if np.isfinite(acc_chunk).all():
        dynamic = acc_chunk - acc_chunk.mean(axis=0)
        out['acc_dynamic_energy'] = float(np.mean(np.sum(dynamic * dynamic, axis=1)))
        delta_acc = np.diff(acc_chunk, axis=0)
        out['acc_motion_change_per_s'] = float(np.mean(np.linalg.norm(delta_acc, axis=1)) * 32.0)
        unchanged = np.all(np.abs(delta_acc) < 1e-4, axis=1)
        out['acc_unchanged_fraction'] = float(np.mean(unchanged))
    else:
        out['acc_dynamic_energy'] = np.nan
        out['acc_motion_change_per_s'] = np.nan
        out['acc_unchanged_fraction'] = np.nan
        
    out['acc_peak_hz'], out['acc_band_fraction'] = spectrum(mag, 32.0, 0.5, 10.0)
    return out
