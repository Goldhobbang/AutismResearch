"""Experiment D: Personal Baseline Calibration & FPR Reduction Analysis.
Solves the S7 false alarm issue by evaluating Cold Start vs Calibration Available protocols.
Evaluates:
1. Global Fixed 0.5
2. Fold Tuned
3. Personal Adaptive Threshold (5-min baseline)
4. Personal Adaptive Threshold (Full baseline)
5. Personal Adaptive + Hysteresis
6. Feature-level Baseline Delta Model
7. Logit-level Baseline Centering (Platt shift)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'results' / 'phase3' / 'calibration'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_alert_episodes(y_true, y_pred, step_s=15):
    # Alert episode: rising edge 0 -> 1
    alert_starts = (y_pred == 1) & (np.r_[0, y_pred[:-1]] == 0)
    
    neg_mask = (y_true == 0)
    pos_mask = (y_true == 1)
    
    neg_windows = neg_mask.sum()
    neg_hours = (neg_windows * step_s) / 3600.0
    
    # False alert episodes
    fa_episodes = (neg_mask & alert_starts).sum()
    far_per_hour = fa_episodes / neg_hours if neg_hours > 0 else 0.0
    
    # Stress episode detection: did at least 1 alert fire during stress?
    stress_detected = 1 if (pos_mask & (y_pred == 1)).any() else 0
    
    # Delay to first alert in stress
    pos_indices = np.where(pos_mask)[0]
    if len(pos_indices) > 0 and stress_detected:
        first_pos_idx = pos_indices[0]
        pos_alert_indices = np.where(pos_mask & (y_pred == 1))[0]
        first_alert_idx = pos_alert_indices[0]
        detection_delay_s = max(0.0, (first_alert_idx - first_pos_idx) * step_s)
    else:
        detection_delay_s = np.nan
        
    return fa_episodes, far_per_hour, stress_detected, detection_delay_s

def main():
    pred_path = ROOT / 'results' / 'baselines' / 'wesad_20261024' / 'predictions.csv'
    raw_pred = pd.read_csv(pred_path)
    
    # Focus on LR-all baseline
    df_lr = raw_pred[(raw_pred['model'] == 'lr') & (raw_pred['sensors'] == 'all')].copy()
    
    # We also load features for feature-delta model
    feat_path = ROOT / 'data' / 'processed' / 'wesad_features.csv'
    df_feat = pd.read_csv(feat_path)
    qc_cols = [c for c in df_feat.columns if c.startswith('qc_')]
    meta_cols = ['dataset', 'subject', 'session', 'start_s', 'end_s', 'step_s', 'phase', 'y']
    feat_cols = [c for c in df_feat.columns if c not in set(qc_cols + meta_cols) and df_feat[c].dtype in (np.float64, np.int64, float, int)]
    
    # Pre-train feature-delta models per fold
    # For feature-delta: X_delta = X - mean(X_baseline)
    subjects = sorted(df_lr['subject'].unique())
    delta_predictions = {}
    
    for test_sub in subjects:
        # Compute baseline mean per training subject and subtract
        train_dfs = []
        for s in subjects:
            if s == test_sub:
                continue
            s_data = df_feat[(df_feat['subject'] == s) & (df_feat['phase'].isin([1, 2, 3]))].copy()
            s_base = s_data[s_data['phase'] == 1]
            base_mean = s_base[feat_cols].mean()
            s_data[feat_cols] = s_data[feat_cols] - base_mean
            train_dfs.append(s_data)
        df_train_delta = pd.concat(train_dfs, ignore_index=True)
        
        # Test subject: subtract test subject's 5-min baseline mean (first 20 windows of baseline)
        test_data = df_feat[(df_feat['subject'] == test_sub) & (df_feat['phase'].isin([1, 2, 3]))].copy()
        test_base_5min = test_data[test_data['phase'] == 1].iloc[:20]
        test_base_mean = test_base_5min[feat_cols].mean()
        test_data_delta = test_data.copy()
        test_data_delta[feat_cols] = test_data[feat_cols] - test_base_mean
        
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced', random_state=42))
        ])
        pipe.fit(df_train_delta[feat_cols].values, (df_train_delta['phase'] == 2).astype(int).values)
        test_probs = pipe.predict_proba(test_data_delta[feat_cols].values)[:, 1]
        
        for end_s, prob in zip(test_data['end_s'], test_probs):
            delta_predictions[(test_sub, end_s)] = prob

    # Policies evaluation
    records = []
    subject_details = []
    
    policies = [
        'fixed_05', 'tuned_val',
        'calib_prob_5min', 'calib_prob_adaptive_5min',
        'calib_prob_adaptive_full', 'calib_hysteresis_5min',
        'feature_delta_5min', 'logit_shift_5min'
    ]
    
    for pol in policies:
        sub_metrics = []
        for sub, g in df_lr.groupby('subject'):
            g = g.sort_values('end_s').copy()
            valid = g['evaluation_eligible'].astype(bool).values
            probs = g['prob'].values
            thresh = g['threshold'].iloc[0]
            y_true = (g['phase'] == 2).astype(int).values
            
            # Baseline probabilities (Phase 1)
            base_mask = (g['phase'] == 1).values
            base_probs_all = probs[base_mask]
            base_probs_5min = base_probs_all[:20] if len(base_probs_all) >= 20 else base_probs_all
            
            mu_base_5m = np.mean(base_probs_5min) if len(base_probs_5min) > 0 else 0.1
            std_base_5m = np.std(base_probs_5min) if len(base_probs_5min) > 0 else 0.05
            
            mu_base_full = np.mean(base_probs_all) if len(base_probs_all) > 0 else 0.1
            std_base_full = np.std(base_probs_all) if len(base_probs_all) > 0 else 0.05
            
            # Generate binary predictions based on policy
            if pol == 'fixed_05':
                preds = (probs >= 0.5).astype(int)
            elif pol == 'tuned_val':
                preds = (probs >= thresh).astype(int)
            elif pol == 'calib_prob_5min':
                t_sub = mu_base_5m + 1.0 * std_base_5m
                preds = (probs >= t_sub).astype(int)
            elif pol == 'calib_prob_adaptive_5min':
                # Adaptive: clamp threshold between 0.40 and 0.95
                t_sub = np.clip(mu_base_5m + 1.5 * std_base_5m, 0.40, 0.95)
                preds = (probs >= t_sub).astype(int)
            elif pol == 'calib_prob_adaptive_full':
                t_sub = np.clip(mu_base_full + 1.5 * std_base_full, 0.40, 0.95)
                preds = (probs >= t_sub).astype(int)
            elif pol == 'calib_hysteresis_5min':
                t_hi = np.clip(mu_base_5m + 1.5 * std_base_5m, 0.50, 0.95)
                t_lo = np.clip(mu_base_5m + 0.5 * std_base_5m, 0.25, 0.70)
                hyst = []
                state = 0
                for p, v in zip(probs, valid):
                    if not v:
                        state = 0
                    elif state == 0 and p >= t_hi:
                        state = 1
                    elif state == 1 and p < t_lo:
                        state = 0
                    hyst.append(state)
                preds = np.array(hyst)
            elif pol == 'feature_delta_5min':
                # Look up delta model predictions
                delta_p = np.array([delta_predictions.get((sub, end_s), 0.5) for end_s in g['end_s']])
                preds = (delta_p >= 0.5).astype(int)
            elif pol == 'logit_shift_5min':
                # Logit shift: logit = log(p/(1-p)) - logit_base
                eps = 1e-4
                p_c = np.clip(probs, eps, 1 - eps)
                logit = np.log(p_c / (1 - p_c))
                base_p_c = np.clip(base_probs_5min, eps, 1 - eps)
                base_logit = np.mean(np.log(base_p_c / (1 - base_p_c))) if len(base_p_c) else 0.0
                shifted_logit = logit - base_logit
                shifted_prob = 1.0 / (1.0 + np.exp(-shifted_logit))
                preds = (shifted_prob >= 0.5).astype(int)
                
            # Filter to evaluation eligible windows (y in {0, 1})
            y_eval = y_true[valid]
            p_eval = preds[valid]
            
            neg_mask = (y_eval == 0)
            pos_mask = (y_eval == 1)
            
            rec = recall_score(y_eval, p_eval, zero_division=0)
            prec = precision_score(y_eval, p_eval, zero_division=0)
            f1 = f1_score(y_eval, p_eval, zero_division=0)
            ba = balanced_accuracy_score(y_eval, p_eval)
            
            fp = ((y_eval == 0) & (p_eval == 1)).sum()
            fpr = fp / neg_mask.sum() if neg_mask.sum() > 0 else 0.0
            
            # Alert episode metrics
            fa_eps, far_h, stress_det, delay_s = evaluate_alert_episodes(y_eval, p_eval)
            
            sub_metrics.append({
                'policy': pol, 'subject': sub, 'fpr': fpr, 'recall': rec,
                'precision': prec, 'f1': f1, 'balanced_acc': ba,
                'fa_episodes': fa_eps, 'far_per_hour': far_h,
                'stress_detected': stress_det, 'delay_s': delay_s
            })
            
        sub_df = pd.DataFrame(sub_metrics)
        subject_details.append(sub_df)
        
        s7_fpr = sub_df.loc[sub_df['subject'] == 'S7', 'fpr'].values[0]
        s7_f1 = sub_df.loc[sub_df['subject'] == 'S7', 'f1'].values[0]
        worst_fpr = sub_df['fpr'].max()
        worst_sub = sub_df.loc[sub_df['fpr'] == worst_fpr, 'subject'].values[0]
        
        records.append({
            'policy': pol,
            'fpr_mean': sub_df['fpr'].mean(),
            'fpr_std': sub_df['fpr'].std(),
            's7_fpr': s7_fpr,
            's7_f1': s7_f1,
            'worst_fpr': worst_fpr,
            'worst_subject': worst_sub,
            'recall_mean': sub_df['recall'].mean(),
            'precision_mean': sub_df['precision'].mean(),
            'f1_mean': sub_df['f1'].mean(),
            'balanced_acc_mean': sub_df['balanced_acc'].mean(),
            'fa_episodes_mean': sub_df['fa_episodes'].mean(),
            'total_fa_episodes': sub_df['fa_episodes'].sum(),
            'far_per_hour_mean': sub_df['far_per_hour'].mean(),
            'stress_detection_rate': sub_df['stress_detected'].mean(),
            'delay_s_mean': sub_df['delay_s'].mean()
        })
        
        print(f"[{pol:25s}] FPR_mean={sub_df['fpr'].mean():.4f} (S7={s7_fpr:.4f}, Worst={worst_sub}:{worst_fpr:.4f}) | Rec={sub_df['recall'].mean():.4f} F1={sub_df['f1'].mean():.4f} | E-FA={sub_df['fa_episodes'].mean():.2f} (FAR={sub_df['far_per_hour'].mean():.2f}/h)")

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(OUT_DIR / 'calibration_summary.csv', index=False)
    
    details_df = pd.concat(subject_details, ignore_index=True)
    details_df.to_csv(OUT_DIR / 'calibration_per_subject.csv', index=False)
    print(f"\nExperiment D completed! Summary saved to {OUT_DIR / 'calibration_summary.csv'}")

if __name__ == '__main__':
    main()
