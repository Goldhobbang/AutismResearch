import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'results' / 'phase2' / 'physionet_external'

df = pd.read_csv(OUT_DIR / 'physionet_s01_zero_shot_predictions.csv')
y_true = (df['label'] == 'stress').astype(int).values

models = ['rf_all', 'lr_eda', 'lr_ppg', 'lr_all', 'lr_eda_acc', 'lr_acc']

records = []
for m in models:
    prob = df[f'prob_{m}'].values
    pred05 = df[f'pred05_{m}'].values
    pred_tuned = df[f'pred_tuned_{m}'].values
    
    auroc = float(roc_auc_score(y_true, prob))
    auprc = float(average_precision_score(y_true, prob))
    ba05 = float(balanced_accuracy_score(y_true, pred05))
    rec05 = float(recall_score(y_true, pred05, zero_division=0))
    prec05 = float(precision_score(y_true, pred05, zero_division=0))
    fp05 = int(((y_true == 0) & (pred05 == 1)).sum())
    fpr05 = float(fp05 / (y_true == 0).sum())
    
    records.append({
        'model': m,
        'auroc': auroc,
        'auprc': auprc,
        'balanced_acc_05': ba05,
        'recall_05': rec05,
        'precision_05': prec05,
        'fpr_05': fpr05,
        'mean_prob_stress': float(prob[y_true == 1].mean()),
        'mean_prob_baseline': float(prob[y_true == 0].mean())
    })

summary_df = pd.DataFrame(records)
summary_df.to_csv(OUT_DIR / 'zero_shot_summary.csv', index=False)
print("Saved zero-shot summary to:", OUT_DIR / 'zero_shot_summary.csv')
print(summary_df)
