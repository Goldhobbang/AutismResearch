"""Experiment B: Physiological Signal-Based Stress Analysis across 12 Sensor Combinations.
Evaluates 15-fold LOSO cross-validation for Wrist and Chest physiological signals.
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
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'results' / 'phase3' / 'physiological'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def define_sensor_feature_groups(df):
    w_eda = [c for c in df.columns if c.startswith('w_eda_')]
    w_bvp = [c for c in df.columns if c.startswith('w_bvp_')]
    w_temp = [c for c in df.columns if c.startswith('w_temp_')]
    w_acc = [c for c in df.columns if c.startswith('w_acc_')]
    
    c_ecg = [c for c in df.columns if c.startswith('c_ecg_')]
    c_resp = [c for c in df.columns if c.startswith('c_resp_')]
    c_emg = [c for c in df.columns if c.startswith('c_emg_')]
    c_eda = [c for c in df.columns if c.startswith('c_eda_')]
    c_temp = [c for c in df.columns if c.startswith('c_temp_')]
    
    groups = {
        '1_w_eda': w_eda,
        '2_w_bvp': w_bvp,
        '3_w_temp': w_temp,
        '4_w_eda_bvp': w_eda + w_bvp,
        '5_w_eda_bvp_temp': w_eda + w_bvp + w_temp,
        '6_c_ecg': c_ecg,
        '7_c_resp': c_resp,
        '8_c_ecg_resp': c_ecg + c_resp,
        '9_chest_physio_all': c_ecg + c_resp + c_emg + c_eda + c_temp,
        '10_wrist_physio_all': w_eda + w_bvp + w_temp,
        '11_wrist_physio_acc': w_eda + w_bvp + w_temp + w_acc,
        '12_wrist_acc_alone': w_acc
    }
    return groups

def main():
    feat_path = ROOT / 'data' / 'processed' / 'wesad_physio_features.csv'
    df = pd.read_csv(feat_path)
    
    # Filter to eligible binary stress windows: Phase 1 & 3 (non-stress, y=0), Phase 2 (stress, y=1)
    df_eval = df[df['phase'].isin([1, 2, 3])].copy()
    y_all = (df_eval['phase'] == 2).astype(int).values
    df_eval['target'] = y_all
    
    subjects = sorted(df_eval['subject'].unique())
    groups = define_sensor_feature_groups(df_eval)
    
    models = {
        'lr': lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced', random_state=42),
        'rf': lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1),
        'lgbm': lambda: LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, class_weight='balanced', n_jobs=-1, verbose=-1)
    }
    
    records = []
    per_sub_records = []
    
    print("================== EXPERIMENT B: PHYSIOLOGICAL SIGNAL ANALYSIS ==================")
    for combo_name, feat_cols in groups.items():
        n_feats = len(feat_cols)
        
        for model_name, model_fn in models.items():
            t0 = time.time()
            sub_metrics = []
            
            oof_y_true = []
            oof_y_pred = []
            oof_probs = []
            
            for sub in subjects:
                train_mask = (df_eval['subject'] != sub)
                test_mask = (df_eval['subject'] == sub)
                
                X_train = df_eval.loc[train_mask, feat_cols].values
                y_train = df_eval.loc[train_mask, 'target'].values
                X_test = df_eval.loc[test_mask, feat_cols].values
                y_test = df_eval.loc[test_mask, 'target'].values
                
                pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('clf', model_fn())
                ])
                pipe.fit(X_train, y_train)
                probs = pipe.predict_proba(X_test)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                oof_y_true.extend(y_test)
                oof_y_pred.extend(preds)
                oof_probs.extend(probs)
                
                rec = recall_score(y_test, preds, zero_division=0)
                prec = precision_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                ba = balanced_accuracy_score(y_test, preds)
                
                # AUROC and AP if both classes present in fold
                if len(np.unique(y_test)) > 1:
                    auroc = roc_auc_score(y_test, probs)
                    ap = average_precision_score(y_test, probs)
                else:
                    auroc, ap = np.nan, np.nan
                    
                neg_mask = (y_test == 0)
                fp = ((y_test == 0) & (preds == 1)).sum()
                fpr = fp / neg_mask.sum() if neg_mask.sum() > 0 else 0.0
                
                sub_metrics.append({
                    'combo': combo_name, 'model': model_name, 'subject': sub,
                    'auroc': auroc, 'ap': ap, 'f1': f1, 'balanced_acc': ba,
                    'precision': prec, 'recall': rec, 'fpr': fpr
                })
                per_sub_records.append(sub_metrics[-1])
                
            elapsed = time.time() - t0
            sub_df = pd.DataFrame(sub_metrics)
            
            # Pooled metrics
            pooled_auroc = roc_auc_score(oof_y_true, oof_probs)
            pooled_ap = average_precision_score(oof_y_true, oof_probs)
            pooled_f1 = f1_score(oof_y_true, oof_y_pred)
            pooled_ba = balanced_accuracy_score(oof_y_true, oof_y_pred)
            
            s7_fpr = sub_df.loc[sub_df['subject'] == 'S7', 'fpr'].values[0]
            s7_ap = sub_df.loc[sub_df['subject'] == 'S7', 'ap'].values[0]
            
            records.append({
                'combination': combo_name,
                'model': model_name,
                'n_features': n_feats,
                'macro_ap_mean': sub_df['ap'].mean(),
                'macro_ap_std': sub_df['ap'].std(),
                'macro_auroc_mean': sub_df['auroc'].mean(),
                'macro_f1_mean': sub_df['f1'].mean(),
                'macro_ba_mean': sub_df['balanced_acc'].mean(),
                'macro_recall_mean': sub_df['recall'].mean(),
                'macro_fpr_mean': sub_df['fpr'].mean(),
                's7_fpr': s7_fpr,
                's7_ap': s7_ap,
                'pooled_ap': pooled_ap,
                'pooled_auroc': pooled_auroc,
                'pooled_f1': pooled_f1,
                'pooled_ba': pooled_ba,
                'fit_time_s': elapsed
            })
            
            print(f"[{combo_name:20s}] {model_name:5s} ({n_feats:2d} feats) | AP={sub_df['ap'].mean():.4f} AUROC={sub_df['auroc'].mean():.4f} F1={sub_df['f1'].mean():.4f} FPR={sub_df['fpr'].mean():.4f} (S7_FPR={s7_fpr:.4f}) [{elapsed:.1f}s]")

    summary_df = pd.DataFrame(records)
    summary_path = OUT_DIR / 'physiological_combinations_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    per_sub_df = pd.DataFrame(per_sub_records)
    per_sub_df.to_csv(OUT_DIR / 'physiological_per_subject.csv', index=False)
    print(f"\nExperiment B completed successfully! Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
