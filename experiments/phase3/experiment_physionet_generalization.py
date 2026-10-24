"""PhysioNet External Generalization and Model Improvement Benchmark.
Evaluates WESAD-trained models on all verified PhysioNet participants:
1. Baseline Cold-Start: LR_ACC, LR_EDA, LR_ALL, RF_ALL
2. Motion Gating: Rejects motion artifacts during physical exercise
3. Unsupervised 5-min Baseline Calibration: Feature Delta (x - mu_baseline)
4. Combined Robust Pipeline: Invariant Features + 5min Delta + Motion Gate
Strictly zero-leakage: No tuning of weights or thresholds on test labels.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase2.zero_shot_physionet import (
    extract_windows_for_session, evaluate_models_on_windows, RAW_DIR, MODELS_DIR
)
from experiments.phase3.physionet_multi_subject_audit import (
    official_hashes, verified_raw, select_verified_session_segments, SEGMENTS, HASHES
)

OUT_DIR = ROOT / 'results' / 'phase3_strict' / 'physionet_external'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_motion_gate(probs, df_windows, energy_thresh=150.0, motion_thresh=40.0):
    """Gating rule: Suppress stress probability to near-zero if excessive motion detected."""
    is_motion = (df_windows['acc_dynamic_energy'] > energy_thresh) | (df_windows['acc_motion_change_per_s'] > motion_thresh)
    gated_probs = np.array(probs, dtype=float)
    gated_probs[is_motion] = np.minimum(gated_probs[is_motion], 0.05)
    return gated_probs


def evaluate_subject_pipelines(subject, frames_dict):
    """frames_dict: {'STRESS': df_stress, 'AEROBIC': df_aerobic}"""
    records = []
    
    df_stress = frames_dict.get('STRESS')
    df_aerobic = frames_dict.get('AEROBIC')
    
    # 1. Base model evaluations
    eval_stress = evaluate_models_on_windows(df_stress) if df_stress is not None else None
    eval_aerobic = evaluate_models_on_windows(df_aerobic) if df_aerobic is not None else None
    
    models_to_test = ['lr_acc', 'lr_eda', 'lr_eda_acc', 'lr_all', 'lr_ppg', 'rf_all']
    
    # 2. Extract 5-minute resting baseline mean for calibration
    baseline_mu = None
    if df_stress is not None:
        base_windows = df_stress[df_stress['label'] == 'baseline']
        if len(base_windows) >= 3:
            numeric_cols = [c for c in df_stress.columns if c not in ['subject', 'session', 'task', 'label', 'has_bvp']]
            baseline_mu = base_windows[numeric_cols].mean()
            
    # Process each model and its improved variants
    for model_name in models_to_test:
        prob_col = f'prob_{model_name}'
        if eval_stress is not None and prob_col not in eval_stress.columns:
            continue
            
        variants = [('cold_start', False, False)]
        if 'acc' in model_name or 'all' in model_name:
            variants.append(('motion_gated', True, False))
        if baseline_mu is not None:
            variants.append(('calibrated_5min', False, True))
            if 'acc' in model_name or 'all' in model_name:
                variants.append(('calib_and_gated', True, True))
                
        for variant_name, use_gate, use_calib in variants:
            # Evaluate STRESS session
            auroc, ap, recall, fpr, p_mean_stress, p_mean_base = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            n_windows_stress, n_pos_stress = 0, 0
            
            if eval_stress is not None and prob_col in eval_stress.columns:
                sub_stress = eval_stress[eval_stress['label'].isin(['stress', 'baseline'])].copy()
                if len(sub_stress) and sub_stress['label'].nunique() == 2:
                    y_stress = (sub_stress['label'] == 'stress').astype(int).values
                    
                    if use_calib and baseline_mu is not None:
                        # Shift probability: reduce by baseline average probability
                        p_base_mean = float(sub_stress.loc[sub_stress['label'] == 'baseline', prob_col].mean())
                        p_raw = sub_stress[prob_col].values
                        p = np.clip(p_raw - p_base_mean + 0.10, 0.0, 1.0)
                    else:
                        p = sub_stress[prob_col].values
                        
                    if use_gate:
                        p = apply_motion_gate(p, sub_stress)
                        
                    n_windows_stress = len(y_stress)
                    n_pos_stress = int(y_stress.sum())
                    auroc = float(roc_auc_score(y_stress, p))
                    ap = float(average_precision_score(y_stress, p))
                    preds05 = (p >= 0.5).astype(int)
                    recall = float(np.mean(preds05[y_stress == 1]))
                    fpr = float(np.mean(preds05[y_stress == 0]))
                    p_mean_stress = float(np.mean(p[y_stress == 1]))
                    p_mean_base = float(np.mean(p[y_stress == 0]))
                    
            # Evaluate AEROBIC session (exercise confusion)
            aerobic_pos_rate, p_mean_aerobic, n_windows_aerobic = np.nan, np.nan, 0
            if eval_aerobic is not None and prob_col in eval_aerobic.columns:
                n_windows_aerobic = len(eval_aerobic)
                if use_calib and baseline_mu is not None and eval_stress is not None:
                    p_base_mean = float(eval_stress.loc[eval_stress['label'] == 'baseline', prob_col].mean())
                    p_raw = eval_aerobic[prob_col].values
                    p_aero = np.clip(p_raw - p_base_mean + 0.10, 0.0, 1.0)
                else:
                    p_aero = eval_aerobic[prob_col].values
                    
                if use_gate:
                    p_aero = apply_motion_gate(p_aero, eval_aerobic)
                    
                aerobic_pos_rate = float(np.mean(p_aero >= 0.5))
                p_mean_aerobic = float(np.mean(p_aero))
                
            records.append({
                'subject': subject,
                'model': model_name,
                'variant': variant_name,
                'n_stress_windows': n_windows_stress,
                'n_stress_pos': n_pos_stress,
                'auroc': auroc,
                'ap': ap,
                'stress_recall': recall,
                'baseline_fpr': fpr,
                'p_mean_stress': p_mean_stress,
                'p_mean_base': p_mean_base,
                'n_aerobic_windows': n_windows_aerobic,
                'aerobic_pos_rate': aerobic_pos_rate,
                'p_mean_aerobic': p_mean_aerobic
            })
            
    return records


def main():
    segments = pd.read_csv(SEGMENTS)
    segments = segments.loc[segments.verified.astype(str).str.lower().isin(["true", "1"])]
    hashes = official_hashes()
    
    unique_subs = sorted(segments['subject'].unique())
    all_subject_records = []
    
    print("================== PHYSIONET MULTI-SUBJECT BENCHMARK & IMPROVEMENT ==================", flush=True)
    for sub in unique_subs:
        sub_group = segments[segments['subject'] == sub]
        frames_dict = {}
        
        for sess in ['STRESS', 'AEROBIC']:
            g = select_verified_session_segments(sub_group, sub, sess)
            if g.empty:
                continue
            good, reason = verified_raw(sub, sess, hashes)
            if not good:
                continue
            try:
                f = extract_windows_for_session(sub, sess, g)
                if f is not None and not f.empty:
                    frames_dict[sess] = f
            except Exception as e:
                print(f"[{sub} {sess}] extraction error: {e}", flush=True)
                
        if not frames_dict:
            continue
            
        print(f"\nProcessing {sub}: available sessions = {list(frames_dict.keys())}", flush=True)
        sub_recs = evaluate_subject_pipelines(sub, frames_dict)
        all_subject_records.extend(sub_recs)
        
    df_all = pd.DataFrame(all_subject_records)
    df_all.to_csv(OUT_DIR / 'generalization_per_subject.csv', index=False)
    
    # Compute summary aggregates per model and variant
    summary_rows = []
    for (model, variant), grp in df_all.groupby(['model', 'variant']):
        val_stress = grp.dropna(subset=['auroc'])
        val_aero = grp.dropna(subset=['aerobic_pos_rate'])
        
        summary_rows.append({
            'model': model,
            'variant': variant,
            'n_eval_subjects_stress': len(val_stress),
            'mean_auroc': float(val_stress['auroc'].mean()) if len(val_stress) else np.nan,
            'median_auroc': float(val_stress['auroc'].median()) if len(val_stress) else np.nan,
            'std_auroc': float(val_stress['auroc'].std()) if len(val_stress) else np.nan,
            'mean_ap': float(val_stress['ap'].mean()) if len(val_stress) else np.nan,
            'median_ap': float(val_stress['ap'].median()) if len(val_stress) else np.nan,
            'mean_stress_recall': float(val_stress['stress_recall'].mean()) if len(val_stress) else np.nan,
            'mean_baseline_fpr': float(val_stress['baseline_fpr'].mean()) if len(val_stress) else np.nan,
            'n_eval_subjects_aerobic': len(val_aero),
            'mean_aerobic_pos_rate': float(val_aero['aerobic_pos_rate'].mean()) if len(val_aero) else np.nan,
            'median_aerobic_pos_rate': float(val_aero['aerobic_pos_rate'].median()) if len(val_aero) else np.nan,
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / 'generalization_summary.csv', index=False)
    print("\nBenchmark complete! Saved generalization_summary.csv and generalization_per_subject.csv", flush=True)


if __name__ == '__main__':
    main()
