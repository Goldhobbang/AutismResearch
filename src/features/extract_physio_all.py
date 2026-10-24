"""Extract comprehensive physiological features (Wrist + Chest) for Experiment B.
Processes all 15 WESAD subjects in 60s windows with 15s step.
"""
import os
import sys
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / 'data' / 'processed' / 'wesad_physio_features.csv'

# Respiration bandpass filter (0.1 - 0.5 Hz at 700 Hz)
B_RESP, A_RESP = butter(2, [0.1 / 350.0, 0.5 / 350.0], btype='band')
# EMG bandpass filter (10 - 300 Hz at 700 Hz)
B_EMG, A_EMG = butter(2, [10.0 / 350.0, 300.0 / 350.0], btype='band')

def basic_stats(arr, prefix):
    x = np.asarray(arr, dtype=float).reshape(-1)
    ok = np.isfinite(x)
    z = x[ok]
    if len(z) < 3:
        return {f'{prefix}_{k}': np.nan for k in ['mean', 'std', 'range', 'slope', 'energy']}
    t = np.arange(len(z))
    slope = np.polyfit(t, z, 1)[0] if len(z) > 1 else 0.0
    return {
        f'{prefix}_mean': float(np.mean(z)),
        f'{prefix}_std': float(np.std(z)),
        f'{prefix}_range': float(np.ptp(z)),
        f'{prefix}_slope': float(slope),
        f'{prefix}_energy': float(np.mean(z * z))
    }

def extract_window_physio(wrist, chest, start_s, end_s):
    feats = {}
    
    # 1. Wrist EDA (4 Hz)
    w_eda = wrist['EDA'][int(start_s * 4):int(end_s * 4)].reshape(-1)
    feats.update(basic_stats(w_eda, 'w_eda'))
    if len(w_eda) >= 4 and np.isfinite(w_eda).all():
        peaks, props = find_peaks(w_eda, prominence=0.05, distance=4)
        feats['w_eda_scr_count'] = float(len(peaks))
        feats['w_eda_scr_amp_sum'] = float(props['prominences'].sum()) if len(peaks) else 0.0
    else:
        feats['w_eda_scr_count'] = np.nan
        feats['w_eda_scr_amp_sum'] = np.nan
        
    # 2. Wrist BVP (64 Hz)
    w_bvp = wrist['BVP'][int(start_s * 64):int(end_s * 64)].reshape(-1)
    feats.update(basic_stats(w_bvp, 'w_bvp'))
    if len(w_bvp) >= 64 and np.isfinite(w_bvp).all() and np.std(w_bvp) > 1e-4:
        freq, pwr = welch(w_bvp, fs=64.0, nperseg=min(len(w_bvp), 2048), detrend='constant')
        band = (freq >= 0.7) & (freq <= 3.5)
        if band.any() and pwr.sum() > 0:
            peak_hz = float(freq[band][np.argmax(pwr[band])])
            feats['w_bvp_peak_hz'] = peak_hz
            feats['w_bvp_hr_bpm'] = peak_hz * 60.0
            feats['w_bvp_band_fraction'] = float(pwr[band].sum() / pwr.sum())
        else:
            feats['w_bvp_peak_hz'] = np.nan
            feats['w_bvp_hr_bpm'] = np.nan
            feats['w_bvp_band_fraction'] = np.nan
    else:
        feats['w_bvp_peak_hz'] = np.nan
        feats['w_bvp_hr_bpm'] = np.nan
        feats['w_bvp_band_fraction'] = np.nan
        
    # 3. Wrist TEMP (4 Hz)
    w_temp = wrist['TEMP'][int(start_s * 4):int(end_s * 4)].reshape(-1)
    feats.update(basic_stats(w_temp, 'w_temp'))
    
    # 4. Wrist ACC (32 Hz)
    w_acc = wrist['ACC'][int(start_s * 32):int(end_s * 32)]
    if len(w_acc) >= 32 and np.isfinite(w_acc).all():
        mag = np.linalg.norm(w_acc, axis=1)
        feats.update(basic_stats(mag, 'w_acc_mag'))
        dynamic = w_acc - np.mean(w_acc, axis=0)
        feats['w_acc_dynamic_energy'] = float(np.mean(np.sum(dynamic * dynamic, axis=1)))
        delta_acc = np.diff(w_acc, axis=0)
        feats['w_acc_jerk'] = float(np.mean(np.linalg.norm(delta_acc, axis=1)) * 32.0)
    else:
        for k in ['w_acc_mag_mean', 'w_acc_mag_std', 'w_acc_mag_range', 'w_acc_mag_slope', 'w_acc_mag_energy', 'w_acc_dynamic_energy', 'w_acc_jerk']:
            feats[k] = np.nan
            
    # 5. Chest ECG (700 Hz)
    c_ecg = chest['ECG'][int(start_s * 700):int(end_s * 700)].reshape(-1)
    feats.update(basic_stats(c_ecg, 'c_ecg'))
    if len(c_ecg) >= 700 and np.isfinite(c_ecg).all():
        # Adaptive threshold for R-peaks
        q75 = np.percentile(c_ecg, 75)
        peaks, _ = find_peaks(c_ecg, distance=int(700 * 0.4), height=q75, prominence=0.25)
        if len(peaks) >= 5:
            rr = np.diff(peaks) / 700.0 # RR in seconds
            feats['c_ecg_hr_mean'] = float(np.mean(60.0 / rr))
            feats['c_ecg_hr_std'] = float(np.std(60.0 / rr))
            feats['c_ecg_rr_mean_ms'] = float(np.mean(rr) * 1000.0)
            feats['c_ecg_sdnn_ms'] = float(np.std(rr) * 1000.0)
            rr_diff = np.diff(rr) * 1000.0
            feats['c_ecg_rmssd_ms'] = float(np.sqrt(np.mean(rr_diff**2)))
            feats['c_ecg_pnn50'] = float(np.mean(np.abs(rr_diff) > 50.0))
        else:
            for k in ['c_ecg_hr_mean', 'c_ecg_hr_std', 'c_ecg_rr_mean_ms', 'c_ecg_sdnn_ms', 'c_ecg_rmssd_ms', 'c_ecg_pnn50']:
                feats[k] = np.nan
    else:
        for k in ['c_ecg_hr_mean', 'c_ecg_hr_std', 'c_ecg_rr_mean_ms', 'c_ecg_sdnn_ms', 'c_ecg_rmssd_ms', 'c_ecg_pnn50']:
            feats[k] = np.nan
            
    # 6. Chest Resp (700 Hz)
    c_resp = chest['Resp'][int(start_s * 700):int(end_s * 700)].reshape(-1)
    feats.update(basic_stats(c_resp, 'c_resp'))
    if len(c_resp) >= 700 and np.isfinite(c_resp).all():
        resp_filt = filtfilt(B_RESP, A_RESP, c_resp)
        r_peaks, r_props = find_peaks(resp_filt, distance=int(700 * 1.5), prominence=0.3)
        if len(r_peaks) >= 2:
            feats['c_resp_rate_bpm'] = float(len(r_peaks) * (60.0 / (end_s - start_s)))
            feats['c_resp_amp_mean'] = float(np.mean(r_props['prominences']))
            feats['c_resp_amp_std'] = float(np.std(r_props['prominences']))
        else:
            feats['c_resp_rate_bpm'] = np.nan
            feats['c_resp_amp_mean'] = np.nan
            feats['c_resp_amp_std'] = np.nan
    else:
        feats['c_resp_rate_bpm'] = np.nan
        feats['c_resp_amp_mean'] = np.nan
        feats['c_resp_amp_std'] = np.nan
        
    # 7. Chest EMG (700 Hz)
    c_emg = chest['EMG'][int(start_s * 700):int(end_s * 700)].reshape(-1)
    if len(c_emg) >= 700 and np.isfinite(c_emg).all():
        emg_filt = filtfilt(B_EMG, A_EMG, c_emg)
        feats.update(basic_stats(emg_filt, 'c_emg'))
        feats['c_emg_rms'] = float(np.sqrt(np.mean(emg_filt**2)))
    else:
        for k in ['c_emg_mean', 'c_emg_std', 'c_emg_range', 'c_emg_slope', 'c_emg_energy', 'c_emg_rms']:
            feats[k] = np.nan
            
    # 8. Chest EDA (700 Hz)
    c_eda = chest['EDA'][int(start_s * 700):int(end_s * 700)].reshape(-1)
    feats.update(basic_stats(c_eda, 'c_eda'))
    if len(c_eda) >= 700 and np.isfinite(c_eda).all():
        c_peaks, _ = find_peaks(c_eda, prominence=0.05, distance=700)
        feats['c_eda_scr_count'] = float(len(c_peaks))
    else:
        feats['c_eda_scr_count'] = np.nan
        
    # 9. Chest Temp (700 Hz)
    c_temp = chest['Temp'][int(start_s * 700):int(end_s * 700)].reshape(-1)
    feats.update(basic_stats(c_temp, 'c_temp'))
    
    return feats

def process_subject(pkl_path, window=60, step=15):
    sub = pkl_path.stem
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    wrist = data['signal']['wrist']
    chest = data['signal']['chest']
    labels = data['label'].reshape(-1)
    
    total_sec = len(labels) / 700.0
    rows = []
    
    for end_s in range(window, int(total_sec) + 1, step):
        start_s = end_s - window
        
        # Check label purity across window
        lbl_chunk = labels[int(start_s * 700):int(end_s * 700)]
        if len(lbl_chunk) == 0:
            continue
        first_lbl = lbl_chunk[0]
        if np.all(lbl_chunk == first_lbl):
            phase = int(first_lbl)
        else:
            phase = 0 # mixed
            
        y = 1 if phase == 2 else 0 if phase in (1, 3) else -1
        
        feats = extract_window_physio(wrist, chest, start_s, end_s)
        feats['dataset'] = 'WESAD'
        feats['subject'] = sub
        feats['start_s'] = start_s
        feats['end_s'] = end_s
        feats['phase'] = phase
        feats['y'] = y
        rows.append(feats)
        
    return pd.DataFrame(rows)

def main():
    import multiprocessing
    raw_dir = ROOT / 'data' / 'raw' / 'WESAD'
    paths = sorted(list(raw_dir.glob('S*/S*.pkl')))
    print(f"Found {len(paths)} subjects for physiological extraction")
    
    t_start = time.time()
    # Process subjects in parallel using 4 worker processes
    with multiprocessing.Pool(processes=min(4, os.cpu_count() or 4)) as pool:
        dfs = pool.map(process_subject, paths)
        
    df_all = pd.concat(dfs, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_PATH, index=False)
    print(f"\nSaved all physiological features to {OUT_PATH} in {time.time()-t_start:.1f}s (Shape: {df_all.shape})", flush=True)

if __name__ == '__main__':
    main()
