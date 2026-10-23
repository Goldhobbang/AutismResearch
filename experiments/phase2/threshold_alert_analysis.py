"""Threshold, calibration, and alert policy optimization analysis."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'results' / 'phase2' / 'threshold'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def score_predictions(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ba = balanced_accuracy_score(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=float(prec), recall=float(rec), f1=float(f1),
                balanced_acc=float(ba), fpr=float(fpr))

def apply_policies(g):
    # g is ordered by end_s for one subject/model
    g = g.sort_values('end_s').copy()
    valid = g.evaluation_eligible.astype(bool).values
    probs = g.prob.values
    thresh = g.threshold.iloc[0]
    
    # 1. Fixed 0.5
    g['pred_fixed05'] = (probs >= 0.5).astype(int)
    
    # 2. Tuned validation threshold
    g['pred_tuned'] = (probs >= thresh).astype(int)
    
    # 3. High precision threshold (e.g. max(thresh, 0.7))
    g['pred_conservative'] = (probs >= max(thresh, 0.70)).astype(int)
    
    # 4. Consecutive 2 exceedances
    consec2 = []
    streak = 0
    for p, v in zip(probs, valid):
        streak = streak + 1 if (v and p >= thresh) else 0
        consec2.append(1 if streak >= 2 else 0)
    g['pred_consec2'] = consec2
    
    # 5. Consecutive 2 + 60s cooldown
    alerts_cooldown = []
    streak = 0
    last_alarm = -np.inf
    for end_s, p, v in zip(g.end_s.values, probs, valid):
        streak = streak + 1 if (v and p >= thresh) else 0
        alarm = 1 if (v and streak >= 2 and (end_s - last_alarm >= 60)) else 0
        if alarm:
            last_alarm = end_s
            streak = 0
        alerts_cooldown.append(alarm)
    g['pred_cooldown60'] = alerts_cooldown
    
    # 6. Hysteresis (high=0.7, low=0.3)
    hysteresis = []
    active = 0
    for p, v in zip(probs, valid):
        if not v:
            active = 0
        elif active == 0 and p >= max(thresh, 0.70):
            active = 1
        elif active == 1 and p < 0.30:
            active = 0
        hysteresis.append(active)
    g['pred_hysteresis'] = hysteresis
    
    return g

def main():
    pred_path = ROOT / 'results' / 'wesad' / 'predictions.csv'
    raw_pred = pd.read_csv(pred_path)
    
    # Focus on primary LR-all model
    df_lr = raw_pred[(raw_pred.model == 'lr') & (raw_pred.sensors == 'all')].copy()
    
    # Process policies per subject
    processed = []
    for sub, g in df_lr.groupby('subject'):
        processed.append(apply_policies(g))
    df_policies = pd.concat(processed, ignore_index=True)
    df_policies.to_csv(OUT_DIR / 'policy_predictions.csv', index=False)
    
    # Evaluate each policy per subject on eligible windows
    eligible = df_policies[df_policies.evaluation_eligible.astype(bool)]
    policies = ['pred_fixed05', 'pred_tuned', 'pred_conservative', 'pred_consec2', 'pred_cooldown60', 'pred_hysteresis']
    
    records = []
    for pol in policies:
        for sub, g in eligible.groupby('subject'):
            sc = score_predictions(g.y, g[pol])
            records.append({'policy': pol, 'subject_id': sub, **sc})
            
    policy_df = pd.DataFrame(records)
    policy_df.to_csv(OUT_DIR / 'policy_per_subject.csv', index=False)
    
    # Summary of policies
    summary = []
    for pol, g in policy_df.groupby('policy'):
        summary.append({
            'policy': pol,
            'f1_mean': float(g.f1.mean()),
            'f1_std': float(g.f1.std()),
            'recall_mean': float(g.recall.mean()),
            'precision_mean': float(g.precision.mean()),
            'balanced_acc_mean': float(g.balanced_acc.mean()),
            'fpr_mean': float(g.fpr.mean()),
            'worst_s7_fpr': float(g[g.subject_id == 'S7'].fpr.iloc[0]),
            'worst_s7_f1': float(g[g.subject_id == 'S7'].f1.iloc[0])
        })
    sum_df = pd.DataFrame(summary).sort_values('f1_mean', ascending=False)
    sum_df.to_csv(OUT_DIR / 'policy_summary.csv', index=False)
    print("Policy Evaluation Summary:")
    print(sum_df[['policy', 'f1_mean', 'recall_mean', 'precision_mean', 'fpr_mean', 'worst_s7_fpr']].to_string())
    
    # Pareto Frontier of Thresholds [0.05 : 0.01 : 0.95]
    pareto_records = []
    threshold_grid = np.linspace(0.05, 0.95, 91)
    for t in threshold_grid:
        sub_metrics = []
        for sub, g in eligible.groupby('subject'):
            pred_t = (g.prob >= t).astype(int)
            sub_metrics.append(score_predictions(g.y, pred_t))
        sub_df = pd.DataFrame(sub_metrics)
        pareto_records.append({
            'threshold': float(t),
            'recall_mean': float(sub_df.recall.mean()),
            'precision_mean': float(sub_df.precision.mean()),
            'f1_mean': float(sub_df.f1.mean()),
            'fpr_mean': float(sub_df.fpr.mean()),
            's7_fpr': float(score_predictions(eligible[eligible.subject == 'S7'].y, (eligible[eligible.subject == 'S7'].prob >= t).astype(int))['fpr']),
            's7_recall': float(score_predictions(eligible[eligible.subject == 'S7'].y, (eligible[eligible.subject == 'S7'].prob >= t).astype(int))['recall'])
        })
    pareto_df = pd.DataFrame(pareto_records)
    pareto_df.to_csv(OUT_DIR / 'pareto_threshold_frontier.csv', index=False)
    
    # Plot Pareto Frontier
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(pareto_df.fpr_mean, pareto_df.recall_mean, 'o-', color='#187f91', label='Population Mean')
    axes[0].plot(pareto_df.s7_fpr, pareto_df.s7_recall, 's--', color='#e05d44', label='Subject S7')
    axes[0].set(xlabel='False Positive Rate (FPR)', ylabel='Stress Recall', title='Recall vs FPR Pareto Trade-off Across Thresholds')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot Policy Comparison Bar Chart
    axes[1].bar(sum_df.policy, sum_df.f1_mean, color='#2c5d79', alpha=0.8, label='Stress F1')
    axes[1].plot(sum_df.policy, sum_df.fpr_mean, 'r-o', label='FPR (False Alarm Rate)')
    axes[1].set(ylabel='Score / Rate', title='Alert Policy Comparison (Population Mean)')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'pareto_and_policy_comparison.png', dpi=160)
    plt.close(fig)
    print("Saved plots to", OUT_DIR / 'pareto_and_policy_comparison.png')

if __name__ == '__main__':
    main()
