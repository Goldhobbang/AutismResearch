import pickle
import numpy as np
import pandas as pd
from pathlib import Path

raw_dir = Path('data/raw/WESAD')
subjects = sorted([p.name for p in raw_dir.glob('S*') if p.is_dir()])

records = []
for sub in subjects:
    pkl_path = raw_dir / sub / f'{sub}.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    lbl = data['label']
    unique, counts = np.unique(lbl, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    # 700Hz label rate
    row = {'subject': sub, 'total_sec': len(lbl) / 700.0}
    for l in [0, 1, 2, 3, 4, 5, 6, 7]:
        cnt = counts_dict.get(l, 0)
        row[f'label_{l}_sec'] = cnt / 700.0
    records.append(row)

df = pd.DataFrame(records)
print(df.to_string())
print('\nTotal seconds across 15 subjects:')
for l in [1, 2, 3, 4]:
    name = {1: 'Baseline', 2: 'Stress', 3: 'Amusement', 4: 'Meditation'}[l]
    sec = df[f'label_{l}_sec'].sum()
    print(f'Label {l} ({name:10s}): {sec:8.1f}s ({sec/3600:.2f} hours, min={df[f"label_{l}_sec"].min():.1f}s, max={df[f"label_{l}_sec"].max():.1f}s)')
