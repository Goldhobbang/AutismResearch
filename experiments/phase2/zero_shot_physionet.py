"""Zero-shot external validation of WESAD models on PhysioNet 1.0.1.
Evaluates:
1. Stress vs Baseline classification on verified cognitive/social stress segments.
2. Exercise Confusion: Stress prediction probability during Aerobic cycling.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loaders.physionet_e4 import load_e4_csv, compute_physionet_window_features

MODELS_DIR = ROOT / 'results' / 'baselines' / 'wesad_20261024' / 'models'
RAW_DIR = ROOT / 'data' / 'raw' / 'physionet_v1' / 'Wearable_Dataset'
OUT_DIR = ROOT / 'results' / 'phase2' / 'physionet_external'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_windows_for_session(subject, session_type, segments):
    sess_dir = RAW_DIR / session_type / subject
    eda_p = sess_dir / 'EDA.csv'
    bvp_p = sess_dir / 'BVP.csv'
    acc_p = sess_dir / 'ACC.csv'
    
    has_eda = eda_p.exists() and eda_p.stat().st_size > 1000
    has_bvp = bvp_p.exists() and bvp_p.stat().st_size > 1000
    has_acc = acc_p.exists() and acc_p.stat().st_size > 1000
    
    if not (has_eda and has_acc):
        print(f"Skipping {subject} {session_type}: missing EDA or ACC (EDA={has_eda}, BVP={has_bvp}, ACC={has_acc})")
        return None
        
    t_eda, fs_eda, arr_eda = load_e4_csv(eda_p)
    t_acc, fs_acc, arr_acc = load_e4_csv(acc_p)
    if has_bvp:
        t_bvp, fs_bvp, arr_bvp = load_e4_csv(bvp_p)
    else:
        t_bvp, fs_bvp, arr_bvp = 0, 64.0, np.zeros(int(len(arr_eda)/fs_eda * 64.0))
        
    window_sec = 60
    step_sec = 15
    
    records = []
    
    for _, seg in segments.iterrows():
        s_start = seg['start_s']
        s_end = seg['end_s']
        label = seg['label']
        task = seg['task']
        
        # Slide windows within [s_start, s_end]
        w_start = s_start
        while w_start + window_sec <= s_end:
            w_end = w_start + window_sec
            
            # Slice signals: start and end sample indices
            idx_eda_lo = int(np.round(w_start * fs_eda))
            idx_eda_hi = int(np.round(w_end * fs_eda))
            chunk_eda = arr_eda[idx_eda_lo:idx_eda_hi]
            
            idx_acc_lo = int(np.round(w_start * fs_acc))
            idx_acc_hi = int(np.round(w_end * fs_acc))
            chunk_acc = arr_acc[idx_acc_lo:idx_acc_hi]
            
            if has_bvp:
                idx_bvp_lo = int(np.round(w_start * fs_bvp))
                idx_bvp_hi = int(np.round(w_end * fs_bvp))
                chunk_bvp = arr_bvp[idx_bvp_lo:idx_bvp_hi]
            else:
                chunk_bvp = np.zeros(window_sec * 64)
                
            if len(chunk_eda) == window_sec * 4 and len(chunk_acc) == window_sec * 32:
                feats = compute_physionet_window_features(chunk_eda, chunk_bvp, chunk_acc)
                feats['subject'] = subject
                feats['session'] = session_type
                feats['task'] = task
                feats['start_s'] = w_start
                feats['end_s'] = w_end
                feats['label'] = label
                feats['has_bvp'] = bool(has_bvp)
                records.append(feats)
                
            w_start += step_sec
            
    if not records:
        return None
    return pd.DataFrame(records)

def evaluate_models_on_windows(df_windows):
    # Model families to test:
    model_families = ['lr_acc', 'lr_eda', 'lr_eda_acc']
    if df_windows['has_bvp'].all():
        model_families.extend(['lr_all', 'lr_ppg', 'rf_all'])
        
    results = {}
    pred_dfs = []
    
    for fam in model_families:
        # Load all 15 folds
        fold_probs = []
        thresholds = []
        for fold in range(15):
            path = MODELS_DIR / f"{fam}_fold{fold}.joblib"
            if not path.exists():
                continue
            model_dict = joblib.load(path)
            pipe = model_dict['model']
            cols = model_dict['columns']
            thresh = model_dict['threshold']
            
            X = df_windows[cols].values
            prob = pipe.predict_proba(X)[:, 1]
            fold_probs.append(prob)
            thresholds.append(thresh)
            
        mean_prob = np.mean(fold_probs, axis=0)
        mean_thresh = np.mean(thresholds)
        
        df_windows[f'prob_{fam}'] = mean_prob
        df_windows[f'pred05_{fam}'] = (mean_prob >= 0.5).astype(int)
        df_windows[f'pred_tuned_{fam}'] = (mean_prob >= mean_thresh).astype(int)
        
    return df_windows

if __name__ == '__main__':
    # Load verified segments
    seg_path = ROOT / 'data' / 'processed' / 'physionet' / 'verified_segments.csv'
    df_segs = pd.read_csv(seg_path)
    
    # Check S01
    s01_segs = df_segs[(df_segs['subject'] == 'S01') & (df_segs['verified'] == True)]
    print(f"Verified segments for S01: {len(s01_segs)}")
    
    stress_segs = s01_segs[s01_segs['session'].str.contains('STRESS')]
    df_stress = extract_windows_for_session('S01', 'STRESS', stress_segs)
    
    # Only try aerobic if ACC and BVP are fully downloaded (not actively writing)
    aerobic_acc_p = RAW_DIR / 'AEROBIC' / 'S01' / 'ACC.csv'
    df_aerobic = None
    if aerobic_acc_p.exists() and aerobic_acc_p.stat().st_size > 1000000:
        try:
            aerobic_segs = s01_segs[s01_segs['session'].str.contains('AEROBIC')]
            df_aerobic = extract_windows_for_session('S01', 'AEROBIC', aerobic_segs)
        except Exception as e:
            print(f"Aerobic session still completing download: {e}")
    
    dfs = [d for d in [df_stress, df_aerobic] if d is not None]
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"Total windows extracted: {len(df_all)}")
        print(df_all.groupby(['session', 'task', 'label']).size())
        
        df_eval = evaluate_models_on_windows(df_all)
        df_eval.to_csv(OUT_DIR / 'physionet_s01_zero_shot_predictions.csv', index=False)
        print(f"Saved predictions to {OUT_DIR / 'physionet_s01_zero_shot_predictions.csv'}")
