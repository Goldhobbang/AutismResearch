"""WESAD synchronized pickle loader, wrist signals only.

Loading a pickle executes code: callers must verify original provenance first.
"""
from pathlib import Path
import pickle
import numpy as np

RATES = {'EDA': 4, 'BVP': 64, 'ACC': 32}
LABEL_RATE = 700
EXPECTED_SUBJECTS = {f'S{x}' for x in list(range(2, 12)) + list(range(13, 18))}


def discover(root):
    return sorted(p for p in Path(root).glob('S*/S*.pkl') if p.stem == p.parent.name)


def validate(data, subject=None):
    wrist = data['signal']['wrist']
    labels = np.asarray(data['label']).reshape(-1)
    if not len(labels) or not np.isfinite(labels).all():
        raise ValueError('Missing or non-finite WESAD labels')
    durations = {'label': len(labels) / LABEL_RATE}
    for key, rate in RATES.items():
        signal = np.asarray(wrist[key])
        if not len(signal) or not np.issubdtype(signal.dtype, np.number):
            raise ValueError(f'{key}: empty or nonnumeric')
        if key == 'ACC' and (signal.ndim != 2 or signal.shape[1] != 3):
            raise ValueError('Expected three wrist ACC axes')
        if key != 'ACC' and signal.ndim > 1 and signal.shape[1] != 1:
            raise ValueError(f'Expected single-channel {key}')
        durations[key] = len(signal) / rate
    reported = data.get('subject')
    if reported is not None and subject is not None and str(reported) != subject:
        raise ValueError(f'Subject mismatch: file={subject}, metadata={reported}')
    return {'duration_s': min(durations.values()), 'durations_s': durations,
            'label_values': np.unique(labels).tolist(),
            'duration_spread_s': max(durations.values()) - min(durations.values()),
            'rates_hz': RATES, 'label_rate_hz': LABEL_RATE}


def load(path):
    path = Path(path)
    with path.open('rb') as handle:
        data = pickle.load(handle, encoding='latin1')
    audit = validate(data, path.stem)
    return data, audit
