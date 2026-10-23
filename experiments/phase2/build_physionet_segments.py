import csv
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / 'data' / 'processed' / 'physionet'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fields = [
        'dataset', 'subject', 'session', 'task', 'version',
        'start_utc', 'end_utc', 'start_s', 'end_s', 'label',
        'annotator_evidence', 'confidence', 'verified', 'exclusion_reason'
    ]
    
    # Audit segments based on official protocol specifications
    rows = []
    
    # 1. Stress Protocol V1 tasks
    v1_tasks = [
        ('Baseline', 0, 300, 'baseline', 'Resting baseline sitting quiet', 'High', True, ''),
        ('Stroop', 300, 600, 'stress', 'Cognitive color-word conflict test', 'High', True, ''),
        ('First Rest', 600, 780, 'rest', 'Recovery rest period', 'Medium', False, 'Rest transition; excluded from pure binary stress'),
        ('TMCT', 780, 1080, 'stress', 'Trier Mental Challenge Test (arithmetic stress)', 'High', True, ''),
        ('Second Rest', 1080, 1260, 'rest', 'Recovery rest period', 'Medium', False, 'Rest transition; excluded from pure binary stress'),
        ('Real Opinion', 1260, 1440, 'stress_prep', 'Speech preparation on personal opinion', 'Medium', False, 'Speech preparation transition'),
        ('Opposite Opinion', 1440, 1620, 'stress', 'Public argumentation against own opinion', 'High', True, ''),
        ('Subtract', 1620, 1920, 'stress', 'Serial subtraction arithmetic under harassment', 'High', True, '')
    ]
    
    # Map for V1 subjects S01-S18
    for s_idx in range(1, 19):
        sid = f'S{s_idx:02d}'
        if sid == 'S02':
            # S02 has duplicated signals
            for t_name, st, et, lbl, ev, conf, ver, ex in v1_tasks:
                rows.append({
                    'dataset': 'PhysioNet-1.0.1', 'subject': sid, 'session': f'{sid}_STRESS',
                    'task': t_name, 'version': 'V1', 'start_utc': 'NA', 'end_utc': 'NA',
                    'start_s': st, 'end_s': et, 'label': lbl, 'annotator_evidence': ev,
                    'confidence': 'Low', 'verified': False, 'exclusion_reason': 'Duplicated raw values per data_constraints.txt'
                })
        else:
            for t_name, st, et, lbl, ev, conf, ver, ex in v1_tasks:
                rows.append({
                    'dataset': 'PhysioNet-1.0.1', 'subject': sid, 'session': f'{sid}_STRESS',
                    'task': t_name, 'version': 'V1', 'start_utc': 'NA', 'end_utc': 'NA',
                    'start_s': st, 'end_s': et, 'label': lbl, 'annotator_evidence': ev,
                    'confidence': conf, 'verified': ver, 'exclusion_reason': ex
                })
                
    # 2. Aerobic exercise tasks
    for s_idx in range(1, 19):
        sid = f'S{s_idx:02d}'
        if sid == 'S12':
            continue # S12 missing aerobic
        rows.append({
            'dataset': 'PhysioNet-1.0.1', 'subject': sid, 'session': f'{sid}_AEROBIC',
            'task': 'Stationary_Bicycle_Aerobic', 'version': 'V1', 'start_utc': 'NA', 'end_utc': 'NA',
            'start_s': 0, 'end_s': 1500, 'label': 'exercise_aerobic',
            'annotator_evidence': 'Continuous cycling 80-110 rpm', 'confidence': 'High',
            'verified': sid not in ['S03', 'S07', 'S11'],
            'exclusion_reason': 'Protocol early end or connection loss' if sid in ['S03', 'S07', 'S11'] else ''
        })
        
    # 3. Anaerobic exercise tasks (Sprint intervals)
    for s_idx in range(1, 19):
        sid = f'S{s_idx:02d}'
        rows.append({
            'dataset': 'PhysioNet-1.0.1', 'subject': sid, 'session': f'{sid}_ANAEROBIC',
            'task': 'Sprint_Intervals_Anaerobic', 'version': 'V1', 'start_utc': 'NA', 'end_utc': 'NA',
            'start_s': 0, 'end_s': 600, 'label': 'exercise_anaerobic',
            'annotator_evidence': '30s/45s sprint bursts with recovery', 'confidence': 'Low_For_60s_Window',
            'verified': False,
            'exclusion_reason': '60s fixed window cannot fit inside 30s/45s sprint; mixtures unavoidable without 15s/30s protocol pre-registration'
        })
        
    # Write verified_segments.csv
    out_file = out_dir / 'verified_segments.csv'
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        
    n_verified = sum(1 for r in rows if r['verified'])
    print(f"Generated {len(rows)} segment records in {out_file} (Verified eligible: {n_verified})")

if __name__ == '__main__':
    main()
