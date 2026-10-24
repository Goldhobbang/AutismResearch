import pandas as pd
import json

summary = pd.read_csv('results/baselines/wesad_20261024/summary.csv')
pooled = pd.read_csv('results/baselines/wesad_20261024/pooled_oof.csv')
ablation = pd.read_csv('results/baselines/wesad_20261024/ablation.csv')
per_sub = pd.read_csv('results/baselines/wesad_20261024/per_subject.csv')

print('=== SUMMARY (Macro mean across 15 subjects) ===')
for idx, r in summary.iterrows():
    print(f"{r['model']:5s} {r['sensors']:7s} | AUPRC={r['auprc_mean']:.6f} AUROC={r['auroc_mean']:.6f} F1={r['f1_mean']:.6f} BalAcc={r['balanced_acc_mean']:.6f} FPR={r['fpr_mean']:.6f}")

print('\n=== POOLED OOF (Micro across all 2031 valid windows) ===')
for idx, r in pooled.iterrows():
    print(f"{r['model']:5s} {r['sensors']:7s} | AUPRC={r['auprc']:.6f} AUROC={r['auroc']:.6f} F1={r['f1']:.6f} BalAcc={r['balanced_acc']:.6f} FPR={r['fpr']:.6f}")

print('\n=== ABLATION TABLE ===')
for idx, r in ablation.iterrows():
    print(f"{r['sensor_combination']:7s} | AUPRC={r['auprc_mean']:.6f} Delta={r['delta_vs_baseline']:+.6f} Imp={r['improved_subjects']} Wors={r['worsened_subjects']} Tied={r['tied_subjects']}")

print('\n=== LR ALL vs LR ACC PER SUBJECT ===')
lr_all = per_sub[(per_sub['model'] == 'lr') & (per_sub['sensors'] == 'all')].set_index('subject_id')
lr_acc = per_sub[(per_sub['model'] == 'lr') & (per_sub['sensors'] == 'acc')].set_index('subject_id')

delta_df = pd.DataFrame({
    'ap_all': lr_all['auprc'],
    'ap_acc': lr_acc['auprc'],
    'delta_acc_minus_all': lr_acc['auprc'] - lr_all['auprc'],
    'fpr_all': lr_all['fpr'],
    'fpr_acc': lr_acc['fpr'],
    'delta_fpr': lr_acc['fpr'] - lr_all['fpr'],
    'f1_all': lr_all['f1'],
    'f1_acc': lr_acc['f1']
})
print(delta_df)

improved_acc = (delta_df['delta_acc_minus_all'] > 0).sum()
worsened_acc = (delta_df['delta_acc_minus_all'] < 0).sum()
tied_acc = (delta_df['delta_acc_minus_all'] == 0).sum()
print(f"\nCompared to LR all, LR acc AP: improved={improved_acc}, worsened={worsened_acc}, tied={tied_acc}")

improved_all = (delta_df['delta_acc_minus_all'] < 0).sum() # where all > acc
worsened_all = (delta_df['delta_acc_minus_all'] > 0).sum() # where all < acc
print(f"Compared to LR acc, LR all AP: improved={improved_all}, worsened={worsened_all}, tied={tied_acc}")
