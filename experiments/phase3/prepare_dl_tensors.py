"""Prepare raw time-series tensors for Deep Learning models (Experiment C).
Extracts all 2,031 binary evaluation windows resampled to unified rates:
- Wrist: 6 channels at 32 Hz (1,920 timesteps): EDA, BVP, TEMP, ACC_X, ACC_Y, ACC_Z
- Chest: 2 channels at 35 Hz (2,100 timesteps): ECG, Resp
Saves to data/processed/wesad_dl_tensors.npz
"""
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / 'data' / 'processed' / 'wesad_dl_tensors.npz'

def main():
    feat_path = ROOT / 'data' / 'processed' / 'wesad_features.csv'
    df_feat = pd.read_csv(feat_path)
    # Filter to 2,031 binary evaluation windows
    df_eval = df_feat[df_feat['phase'].isin([1, 2, 3])].copy()
    
    raw_dir = ROOT / 'data' / 'raw' / 'WESAD'
    subjects = sorted(df_eval['subject'].unique())
    
    X_wrist_list = []
    X_chest_list = []
    y_list = []
    sub_list = []
    start_s_list = []
    end_s_list = []
    phase_list = []
    
    t_start = time.time()
    for sub in subjects:
        sub_df = df_eval[df_eval['subject'] == sub].sort_values('end_s')
        pkl_p = raw_dir / sub / f'{sub}.pkl'
        with open(pkl_p, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        wrist = data['signal']['wrist']
        chest = data['signal']['chest']
        
        # Pre-resample continuous signals to reduce computation
        # EDA 4Hz -> 32Hz (upsample 8x)
        eda_32 = np.repeat(wrist['EDA'].reshape(-1), 8)
        # BVP 64Hz -> 32Hz (downsample 2x)
        bvp_32 = resample_poly(wrist['BVP'].reshape(-1), 1, 2)
        # TEMP 4Hz -> 32Hz (upsample 8x)
        temp_32 = np.repeat(wrist['TEMP'].reshape(-1), 8)
        # ACC 32Hz native
        acc_32 = wrist['ACC']
        
        # Chest ECG 700Hz -> 35Hz (downsample 20x)
        ecg_35 = resample_poly(chest['ECG'].reshape(-1), 1, 20)
        # Chest Resp 700Hz -> 35Hz (downsample 20x)
        resp_35 = resample_poly(chest['Resp'].reshape(-1), 1, 20)
        
        for _, row in sub_df.iterrows():
            s = row['start_s']
            e = row['end_s']
            phase = int(row['phase'])
            y = 1 if phase == 2 else 0
            
            # Slice wrist: 60s * 32Hz = 1920 samples
            idx_32_lo = int(round(s * 32))
            idx_32_hi = idx_32_lo + 1920
            
            chunk_eda = eda_32[idx_32_lo:idx_32_hi]
            chunk_bvp = bvp_32[idx_32_lo:idx_32_hi]
            chunk_temp = temp_32[idx_32_lo:idx_32_hi]
            chunk_acc = acc_32[idx_32_lo:idx_32_hi] # (1920, 3)
            
            # Check length
            if len(chunk_eda) == 1920 and len(chunk_acc) == 1920:
                wrist_tensor = np.column_stack([chunk_eda, chunk_bvp, chunk_temp, chunk_acc]).T # (6, 1920)
            else:
                continue
                
            # Slice chest: 60s * 35Hz = 2100 samples
            idx_35_lo = int(round(s * 35))
            idx_35_hi = idx_35_lo + 2100
            chunk_ecg = ecg_35[idx_35_lo:idx_35_hi]
            chunk_resp = resp_35[idx_35_lo:idx_35_hi]
            
            if len(chunk_ecg) == 2100 and len(chunk_resp) == 2100:
                chest_tensor = np.vstack([chunk_ecg, chunk_resp]) # (2, 2100)
            else:
                continue
                
            X_wrist_list.append(wrist_tensor)
            X_chest_list.append(chest_tensor)
            y_list.append(y)
            sub_list.append(sub)
            start_s_list.append(s)
            end_s_list.append(e)
            phase_list.append(phase)
            
        print(f"[{sub}] processed {len(sub_df)} windows in {time.time()-t_start:.1f}s", flush=True)
        del data
        
    X_wrist = np.array(X_wrist_list, dtype=np.float32)
    X_chest = np.array(X_chest_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects_arr = np.array(sub_list)
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X_wrist=X_wrist, # (2031, 6, 1920)
        X_chest=X_chest, # (2031, 2, 2100)
        y=y,
        subjects=subjects_arr,
        start_s=np.array(start_s_list),
        end_s=np.array(end_s_list),
        phase=np.array(phase_list)
    )
    print(f"\nSaved raw DL tensors to {OUT_PATH} in {time.time()-t_start:.1f}s")
    print(f"X_wrist shape: {X_wrist.shape}, X_chest shape: {X_chest.shape}, y shape: {y.shape}")

if __name__ == '__main__':
    main()
