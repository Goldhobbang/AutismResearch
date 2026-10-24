import pandas as pd
import numpy as np

df = pd.read_csv('results/phase2/threshold/policy_predictions.csv')

# Only consider evaluation_eligible windows
policies = ['pred_fixed05', 'pred_tuned', 'pred_conservative', 'pred_consec2', 'pred_cooldown60', 'pred_hysteresis']

results = []

for sub, g in df.groupby('subject'):
    g = g.sort_values('end_s').copy()
    g_elig = g[g['evaluation_eligible'].astype(bool)].copy()
    
    neg_windows = (g_elig['y'] == 0).sum()
    pos_windows = (g_elig['y'] == 1).sum()
    
    # Negative monitoring duration in hours (15s step per window)
    neg_hours = (neg_windows * 15.0) / 3600.0
    
    for pol in policies:
        y_true = g_elig['y'].values
        y_pred = g_elig[pol].values
        
        # Window-level metrics
        fp_win = ((y_true == 0) & (y_pred == 1)).sum()
        tp_win = ((y_true == 1) & (y_pred == 1)).sum()
        fn_win = ((y_true == 1) & (y_pred == 0)).sum()
        tn_win = ((y_true == 0) & (y_pred == 0)).sum()
        win_fpr = fp_win / neg_windows if neg_windows > 0 else 0.0
        win_recall = tp_win / pos_windows if pos_windows > 0 else 0.0
        
        # Episode-level metrics:
        # In a continuous sequence, an alert episode starts when y_pred changes from 0 to 1
        # For pred_cooldown60, every y_pred == 1 is already an individual alert event trigger.
        if pol == 'pred_cooldown60':
            alert_starts = (y_pred == 1)
        else:
            # Rising edges (starts of contiguous alert blocks)
            alert_starts = (y_pred == 1) & (np.r_[0, y_pred[:-1]] == 0)
        
        # False alarm episodes during negative windows
        fa_episodes = ((y_true == 0) & alert_starts).sum()
        far_per_hour = fa_episodes / neg_hours if neg_hours > 0 else 0.0
        
        # Stress episode detection: did any alert fire during y_true == 1?
        stress_detected = 1 if ((y_true == 1) & (y_pred == 1)).any() else 0
        
        # Time to first alert in stress (from first positive window)
        pos_indices = np.where(y_true == 1)[0]
        if len(pos_indices) > 0 and stress_detected:
            first_pos_time = g_elig['end_s'].iloc[pos_indices[0]]
            pos_alert_indices = np.where((y_true == 1) & (y_pred == 1))[0]
            first_alert_time = g_elig['end_s'].iloc[pos_alert_indices[0]]
            detection_delay_s = max(0.0, first_alert_time - first_pos_time)
        else:
            detection_delay_s = np.nan
            
        results.append({
            'subject': sub,
            'policy': pol,
            'neg_windows': neg_windows,
            'pos_windows': pos_windows,
            'neg_hours': neg_hours,
            'win_fp': fp_win,
            'win_tp': tp_win,
            'win_fpr': win_fpr,
            'win_recall': win_recall,
            'fa_episodes': fa_episodes,
            'far_per_hour': far_per_hour,
            'stress_detected': stress_detected,
            'detection_delay_s': detection_delay_s
        })

res_df = pd.DataFrame(results)

print("=== MACRO MEAN ACROSS 15 SUBJECTS ===")
summary = res_df.groupby('policy').agg({
    'win_fpr': 'mean',
    'win_recall': 'mean',
    'fa_episodes': ['mean', 'sum'],
    'far_per_hour': 'mean',
    'stress_detected': ['mean', 'sum'],
    'detection_delay_s': 'mean'
})
print(summary)
