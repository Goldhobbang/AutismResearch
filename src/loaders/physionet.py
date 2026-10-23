"""Empatica-style CSV parsing; no invented stress/exercise labels.

This loader deliberately requires separately verified segment annotations.
The first line may be a UTC date string, not only a Unix number.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd


def parse_start(value):
    value = str(value).strip().strip('"')
    try:
        return float(value)
    except ValueError:
        timestamp = pd.to_datetime(value, utc=True, errors='raise')
        return float(timestamp.timestamp())


def read_signal(path, expected_rate=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        first, second = next(reader), next(reader)
        start_utc = parse_start(first[0])
        rate = float(second[0])
        rows = list(reader)
    if rate <= 0 or (expected_rate is not None and not np.isclose(rate, expected_rate)):
        raise ValueError(f'Unexpected sampling frequency: {rate}')
    values = np.asarray([[float(x) if x.strip() else np.nan for x in row]
                         for row in rows if row], dtype=float)
    if not len(values):
        raise ValueError('Empty signal CSV')
    return {'start_utc': start_utc, 'rate': rate, 'values': values}


def read_verified_segments(path):
    frame = pd.read_csv(path)
    required = {'subject', 'session', 'start_s', 'end_s', 'label', 'verified'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Segment manifest needs {sorted(required)}')
    if frame.empty:
        raise ValueError('No verified segments: supervised D2 remains blocked')
    verified = frame.verified.astype(str).str.lower().isin(['true', '1', 'yes'])
    if not verified.all():
        raise ValueError('Unverified segment found; exclude explicitly before supervised use')
    if not (frame.end_s > frame.start_s).all():
        raise ValueError('Nonpositive segment duration')
    for _, group in frame.groupby(['subject', 'session']):
        group = group.sort_values('start_s')
        if np.any(group.start_s.to_numpy()[1:] < group.end_s.to_numpy()[:-1]):
            raise ValueError('Overlapping segments require explicit resolution')
    return frame
