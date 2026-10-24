"""Experiment A: WESAD Multiclass State Classification (3-class, 4-class, and binary reference).
Implements 15-fold LOSO cross-validation across 5 model families and 3 sensor combinations.
Records Macro F1, Balanced Acc, per-class metrics, confusion matrices, and Amusement vs Stress confusion.
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, brier_score_loss
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'results' / 'phase3' / 'multiclass'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Feature sets
def get_feature_columns(df):
    qc_cols = [c for c in df.columns if c.startswith('qc_')]
    meta_cols = ['dataset', 'subject', 'session', 'start_s', 'end_s', 'step_s', 'phase', 'y']
    ignore = set(qc_cols + meta_cols)
    all_feats = [c for c in df.columns if c not in ignore and df[c].dtype in (np.float64, np.int64, float, int)]
    
    eda_feats = [c for c in all_feats if c.startswith('eda_')]
    ppg_feats = [c for c in all_feats if c.startswith('ppg_')]
    acc_feats = [c for c in all_feats if c.startswith('acc_')]
    physio_feats = eda_feats + ppg_feats
    
    return {
        'all': all_feats,
        'acc': acc_feats,
        'physio': physio_feats,
        'eda': eda_feats,
        'ppg': ppg_feats
    }

def get_models(random_state=42):
    return {
        'dummy': lambda: DummyClassifier(strategy='prior'),
        'lr': lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced', random_state=random_state),
        'rf': lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight='balanced', n_jobs=-1),
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
        'lgbm': lambda: LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=random_state, class_weight='balanced', n_jobs=-1, verbose=-1)
    }

def evaluate_loso(df_task, task_name, class_labels, class_names, feat_dict, models):
    subjects = sorted(df_task['subject'].unique())
    n_classes = len(class_labels)
    
    records = []
    oof_predictions = []
    
    for sensor_name, feat_cols in [('all', feat_dict['all']), ('acc', feat_dict['acc']), ('physio', feat_dict['physio'])]:
        for model_name, model_fn in models.items():
            t0 = time.time()
            sub_metrics = []
            
            oof_y_true = []
            oof_y_pred = []
            oof_probs = []
            oof_subjects = []
            
            for sub in subjects:
                train_mask = (df_task['subject'] != sub)
                test_mask = (df_task['subject'] == sub)
                
                X_train = df_task.loc[train_mask, feat_cols].values
                y_train = df_task.loc[train_mask, 'target'].values
                X_test = df_task.loc[test_mask, feat_cols].values
                y_test = df_task.loc[test_mask, 'target'].values
                
                # Preprocessing pipeline
                pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('clf', model_fn())
                ])
                
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                
                if hasattr(pipe, 'predict_proba'):
                    probs = pipe.predict_proba(X_test)
                else:
                    probs = np.zeros((len(y_test), n_classes))
                    
                oof_y_true.extend(y_test)
                oof_y_pred.extend(y_pred)
                oof_probs.extend(probs)
                oof_subjects.extend([sub] * len(y_test))
                
                # Per subject metrics
                acc = accuracy_score(y_test, y_pred)
                ba = balanced_accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                # Stress class metrics (target index for stress)
                stress_idx = class_labels.index(2) # Phase 2 is stress
                stress_rec = recall_score(y_test == stress_idx, y_pred == stress_idx, zero_division=0)
                stress_prec = precision_score(y_test == stress_idx, y_pred == stress_idx, zero_division=0)
                stress_f1 = f1_score(y_test == stress_idx, y_pred == stress_idx, zero_division=0)
                
                # False positive rate for stress on non-stress
                non_stress_mask = (y_test != stress_idx)
                stress_fp = ((y_test != stress_idx) & (y_pred == stress_idx)).sum()
                stress_fpr = stress_fp / non_stress_mask.sum() if non_stress_mask.sum() > 0 else 0.0
                
                sub_metrics.append({
                    'subject': sub, 'accuracy': acc, 'balanced_acc': ba,
                    'macro_f1': macro_f1, 'stress_recall': stress_rec,
                    'stress_precision': stress_prec, 'stress_f1': stress_f1,
                    'stress_fpr': stress_fpr
                })
                
            elapsed = time.time() - t0
            sub_df = pd.DataFrame(sub_metrics)
            
            # Pooled confusion matrix
            cm = confusion_matrix(oof_y_true, oof_y_pred, labels=range(n_classes))
            
            # Amusement vs Stress confusion (Amusement is Phase 3)
            if 3 in class_labels:
                amus_idx = class_labels.index(3)
                amus_total = cm[amus_idx, :].sum()
                amus_pred_as_stress = cm[amus_idx, stress_idx]
                amus_to_stress_rate = amus_pred_as_stress / amus_total if amus_total > 0 else 0.0
            else:
                amus_to_stress_rate = np.nan
                
            records.append({
                'task': task_name,
                'sensors': sensor_name,
                'model': model_name,
                'n_classes': n_classes,
                'macro_f1_mean': sub_df['macro_f1'].mean(),
                'macro_f1_std': sub_df['macro_f1'].std(),
                'balanced_acc_mean': sub_df['balanced_acc'].mean(),
                'balanced_acc_std': sub_df['balanced_acc'].std(),
                'accuracy_mean': sub_df['accuracy'].mean(),
                'stress_f1_mean': sub_df['stress_f1'].mean(),
                'stress_recall_mean': sub_df['stress_recall'].mean(),
                'stress_fpr_mean': sub_df['stress_fpr'].mean(),
                's7_stress_fpr': sub_df.loc[sub_df['subject'] == 'S7', 'stress_fpr'].values[0],
                'amusement_to_stress_rate': amus_to_stress_rate,
                'fit_time_s': elapsed
            })
            
            # Save confusion matrix as json
            cm_dict = {
                'task': task_name, 'sensors': sensor_name, 'model': model_name,
                'class_names': class_names, 'matrix': cm.tolist()
            }
            cm_file = OUT_DIR / f'cm_{task_name}_{sensor_name}_{model_name}.json'
            with open(cm_file, 'w', encoding='utf-8') as f:
                json.dump(cm_dict, f, indent=2)
                
            print(f"[{task_name}] {sensor_name:8s} {model_name:6s} | MacroF1={sub_df['macro_f1'].mean():.4f} BalAcc={sub_df['balanced_acc'].mean():.4f} StressRec={sub_df['stress_recall'].mean():.4f} StressFPR={sub_df['stress_fpr'].mean():.4f} S7_FPR={sub_df.loc[sub_df['subject']=='S7', 'stress_fpr'].values[0]:.4f} ({elapsed:.1f}s)")
            
    return pd.DataFrame(records)

def main():
    feat_path = ROOT / 'data' / 'processed' / 'wesad_features.csv'
    df = pd.read_csv(feat_path)
    
    # Common quality filter
    df_valid = df[df['qc_common_valid'].astype(bool)].copy()
    feat_dict = get_feature_columns(df_valid)
    models = get_models()
    
    all_summaries = []
    
    # ----------------------------------------------------
    # Task 1: A-1 3-class (Baseline=1, Stress=2, Amusement=3)
    # ----------------------------------------------------
    print("\n================== TASK A-1: 3-CLASS (Baseline / Stress / Amusement) ==================")
    df_3class = df_valid[df_valid['phase'].isin([1, 2, 3])].copy()
    # Map phase 1->0, 2->1, 3->2
    phase_to_target_3 = {1: 0, 2: 1, 3: 2}
    df_3class['target'] = df_3class['phase'].map(phase_to_target_3)
    res_3class = evaluate_loso(
        df_3class, '3class',
        class_labels=[1, 2, 3],
        class_names=['Baseline', 'Stress', 'Amusement'],
        feat_dict=feat_dict, models=models
    )
    all_summaries.append(res_3class)
    
    # ----------------------------------------------------
    # Task 2: A-2 4-class (Baseline=1, Stress=2, Amusement=3, Meditation=4)
    # ----------------------------------------------------
    print("\n================== TASK A-2: 4-CLASS (Baseline / Stress / Amusement / Meditation) ==================")
    df_4class = df_valid[df_valid['phase'].isin([1, 2, 3, 4])].copy()
    phase_to_target_4 = {1: 0, 2: 1, 3: 2, 4: 3}
    df_4class['target'] = df_4class['phase'].map(phase_to_target_4)
    res_4class = evaluate_loso(
        df_4class, '4class',
        class_labels=[1, 2, 3, 4],
        class_names=['Baseline', 'Stress', 'Amusement', 'Meditation'],
        feat_dict=feat_dict, models=models
    )
    all_summaries.append(res_4class)
    
    # ----------------------------------------------------
    # Task 3: A-3 Binary (Stress vs Non-stress: Baseline + Amusement)
    # ----------------------------------------------------
    print("\n================== TASK A-3: BINARY (Stress vs Non-stress: Baseline+Amusement) ==================")
    df_bin = df_valid[df_valid['phase'].isin([1, 2, 3])].copy()
    # Stress=1 (target 1), Non-stress=0 (target 0)
    df_bin['target'] = (df_bin['phase'] == 2).astype(int)
    res_bin = evaluate_loso(
        df_bin, 'binary',
        class_labels=[0, 2], # 0 is non-stress, 2 is stress
        class_names=['Non-Stress', 'Stress'],
        feat_dict=feat_dict, models=models
    )
    all_summaries.append(res_bin)
    
    # Combine and save
    final_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = OUT_DIR / 'multiclass_summary.csv'
    final_summary.to_csv(summary_path, index=False)
    print(f"\nAll multiclass experiments completed successfully! Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
